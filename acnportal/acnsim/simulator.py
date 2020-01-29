import copy
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

from .events import UnplugEvent
from .interface import Interface, InvalidScheduleError


class Simulator:
    """ Central class of the acnsim package.

    The Simulator class is the central place where everything about a particular simulation is stored including the
    network, scheduling algorithm, and events. It is also where timekeeping is done and orchestrates calling the
    scheduling algorithm, sending pilots to the network, and updating the energy delivered to each EV.

    Args:
        network (ChargingNetwork): The charging network which the simulation will use.
        scheduler (BasicAlgorithm): The scheduling algorithm used in the simulation.
        events (EventQueue): Queue of events which will occur in the simulation.
        start (datetime): Date and time of the first period of the simulation.
        period (int): Length of each time interval in the simulation in minutes. Default: 1
        signals (Dict[str, ...]):
        store_schedule_history (bool): If True, store the scheduler output each time it is run. Note this can use lots
            of memory for long simulations.
    """

    def __init__(self, network, scheduler, events, start, period=1, signals=None,
                 store_schedule_history=False, verbose=True):
        self.network = network
        self.scheduler = scheduler
        self.scheduler.register_interface(Interface(self))
        self.event_queue = events
        self.start = start
        self.period = period
        self.max_recompute = scheduler.max_recompute
        self.signals = signals
        self.verbose = verbose

        # Information storage
        self.pilot_signals = np.zeros((len(self.network.station_ids), self.event_queue.get_last_timestamp() + 1))
        self.charging_rates = np.zeros((len(self.network.station_ids), self.event_queue.get_last_timestamp() + 1))
        self.peak = 0
        self.ev_history = {}
        self.event_history = []
        if store_schedule_history:
            self.schedule_history = {}
        else:
            self.schedule_history = None

        # Local Variables
        self._iteration = 0
        self._resolve = False
        self._last_schedule_update = 0

    @property
    def iteration(self):
        return self._iteration

    def run(self):
        """ Run the simulation until the event queue is empty.

        The run function is the heart of the simulator. It triggers all actions and keeps the simulator moving forward.
        Its actions are (in order):
            1. Get current events from the event queue and execute them.
            2. If necessary run the scheduling algorithm.
            3. Send pilot signals to the network.
            4. Receive back actual charging rates from the network and store the results.

        Returns:
            None
        """
        while not self.event_queue.empty():
            current_events = self.event_queue.get_current_events(self._iteration)
            for e in current_events:
                self.event_history.append(e)
                self._process_event(e)
            if self._resolve or \
                    self.max_recompute is not None and \
                    self._iteration - self._last_schedule_update >= self.max_recompute:
                new_schedule = self.scheduler.run()
                self._update_schedules(new_schedule)
                if self.schedule_history is not None:
                    self.schedule_history[self._iteration] = new_schedule
                self._last_schedule_update = self._iteration
                self._resolve = False
            if self.event_queue.get_last_timestamp() is not None:
                width_increase = max(self.event_queue.get_last_timestamp() + 1, self._iteration + 1)
            else:
                width_increase = self._iteration + 1
            self.pilot_signals = _increase_width(self.pilot_signals, width_increase)
            self.charging_rates = _increase_width(self.charging_rates, width_increase)
            self.network.update_pilots(self.pilot_signals, self._iteration, self.period)
            self._store_actual_charging_rates()
            self._iteration = self._iteration + 1

    def get_active_evs(self):
        """ Return all EVs which are plugged in and not fully charged at the current time.

        Wrapper for self.network.active_evs. See its documentation for more details.

        Returns:
            List[EV]: List of all EVs which are plugged in but not fully charged at the current time.

        """
        evs = copy.deepcopy(self.network.active_evs)
        return evs

    def _process_event(self, event):
        """ Process an event and take appropriate actions.

        Args:
            event (Event): Event to be processed.

        Returns:
            None
        """
        if event.type == 'Plugin':
            self._print('Plugin Event...')
            self.network.plugin(event.ev, event.ev.station_id)
            self.ev_history[event.ev.session_id] = event.ev
            self.event_queue.add_event(UnplugEvent(event.ev.departure, event.ev.station_id, event.ev.session_id))
            self._resolve = True
            self._last_schedule_update = event.timestamp
        elif event.type == 'Unplug':
            self._print('Unplug Event...')
            self.network.unplug(event.station_id)
            self._resolve = True
            self._last_schedule_update = event.timestamp
        elif event.type == 'Recompute':
            self._print('Recompute Event...')
            self._resolve = True

    def _update_schedules(self, new_schedule):
        """ Extend the current self.pilot_signals with the new pilot signal schedule.

        Args:
            new_schedule (Dict[str, List[number]]): Dictionary mappding station ids to a schedule of pilot signals.

        Returns:
            None

        Raises:
            KeyError: Raised when station_id is in the new_schedule but not registered in the Network.
        """
        if len(new_schedule) == 0:
            return

        for station_id in new_schedule:
            if station_id not in self.network.station_ids:
                raise KeyError('Station {0} in schedule but not found in network.'.format(station_id))

        schedule_lengths = set(len(x) for x in new_schedule.values())
        if len(schedule_lengths) > 1:
            raise InvalidScheduleError('All schedules should have the same length.')
        schedule_length = schedule_lengths.pop()

        schedule_matrix = np.array([new_schedule[evse_id] if evse_id in new_schedule else [0] * schedule_length for evse_id in self.network.station_ids])
        if not self.network.is_feasible(schedule_matrix):
            aggregate_currents = self.network.constraint_current(schedule_matrix)
            diff_vec = np.abs(aggregate_currents) - np.tile(self.network.magnitudes + self.network.violation_tolerance, (schedule_length, 1)).T
            max_idx = np.unravel_index(np.argmax(diff_vec), diff_vec.shape)
            max_diff = diff_vec[max_idx]
            max_timeidx = max_idx[1]
            max_constraint = self.network.constraint_index[max_idx[0]]
            warnings.warn(
                f"Invalid schedule provided at iteration {self._iteration}. "
                f"Max violation is {max_diff} A on {max_constraint} "
                f"at time index {max_timeidx}.",
                UserWarning)
        if self._iteration + schedule_length <= self.pilot_signals.shape[1]:
            self.pilot_signals[:, self._iteration:(self._iteration + schedule_length)] = schedule_matrix
        else:
            # We've reached the end of pilot_signals, so double pilot_signal array width
            self.pilot_signals = _increase_width(self.pilot_signals,
                max(self.event_queue.get_last_timestamp() + 1, self._iteration + schedule_length))
            self.pilot_signals[:, self._iteration:(self._iteration + schedule_length)] = schedule_matrix

    def _store_actual_charging_rates(self):
        """ Store actual charging rates from the network in the simulator for later analysis."""
        current_rates = self.network.current_charging_rates
        agg = np.sum(current_rates)
        if self.iteration < self.charging_rates.shape[1]:
            self.charging_rates[:, self.iteration] = current_rates.T
        else:
            self.charging_rates = _increase_width(self.charging_rates, self.event_queue.get_last_timestamp() + 1)
            self.charging_rates[:, self._iteration] = current_rates.T
        self.peak = max(self.peak, agg)

    def _print(self, s):
        if self.verbose:
            print(s)

    def charging_rates_as_df(self):
        """ Return the charging rates as a pandas DataFrame, with EVSE id as columns
        and iteration as index.
        """
        return pd.DataFrame(data=self.charging_rates.T, columns=self.network.station_ids)

    def pilot_signals_as_df(self):
        """ Return the pilot signals as a pandas DataFrame """
        return pd.DataFrame(data=self.pilot_signals.T, columns=self.network.station_ids)

    def index_of_evse(self, station_id):
        """ Return the numerical index of the EVSE given by station_id in the (ordered) dictionary
        of EVSEs.
        """
        if station_id not in self.network.station_ids:
            raise KeyError("EVSE {0} not found in network.".format(station_id))
        return self.network.station_ids.index(station_id)

def _increase_width(a, target_width):
    """ Returns a new 2-D numpy array with target_width number of columns, with the contents
    of a up to the first a.shape[1] columns and 0's thereafter.

    Args:
        a (numpy.Array): 2-D numpy array to be expanded.
        target_width (int): desired number of columns; must be greater than number of columns in a
    Returns:
        numpy.Array
    """
    if target_width <= a.shape[1]:
        return a
    new_matrix = np.zeros((a.shape[0], target_width))
    new_matrix[:, :a.shape[1]] = a
    return new_matrix
