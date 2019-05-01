import copy
from datetime import datetime

from .events import UnplugEvent
from .interface import Interface


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
        max_recomp (int): Maximum number of periods between calling the scheduling algorithm even if no events occur.
            If None, the scheduling algorithm is only called when an event occurs. Default: None.
        signals (Dict[str, ...]):
        store_schedule_history (bool): If True, store the scheduler output each time it is run. Note this can use lots
            of memory for long simulations.
    """

    def __init__(self, network, scheduler, events, start, period=1, max_recomp=None, signals=None,
                 store_schedule_history=False, verbose=True):
        self.network = network
        self.scheduler = scheduler
        self.scheduler.register_interface(Interface(self))
        self.event_queue = events
        self.start = start
        self.period = period
        self.max_recompute = max_recomp
        self.signals = signals
        self.verbose = verbose

        # Information storage
        self.pilot_signals = {station_id: [] for station_id in self.network.space_ids}
        self.charging_rates = {station_id: [] for station_id in self.network.space_ids}
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
                if self.schedule_history is not None:
                    self.schedule_history[self._iteration] = new_schedule
                self._update_schedules(new_schedule)
                self._last_schedule_update = self._iteration
                self._resolve = False
            self._expand_pilots()
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

        schedule_lengths = set(len(x) for x in new_schedule.values())
        if len(schedule_lengths) > 1:
            raise InvalidScheduleError('All schedules should have the same length.')
        schedule_length = schedule_lengths.pop()

        for station_id in new_schedule:
            if station_id not in self.network.space_ids:
                raise KeyError('Station {0} in schedule but not found in network.'.format(station_id))

        for station_id in self.network.space_ids:
            if station_id in new_schedule:
                self.pilot_signals[station_id] = _overwrite_at_index(self._iteration, self.pilot_signals[station_id],
                                                                     new_schedule[station_id])
            else:
                # If a station is not in the new schedule, it shouldn't be charging.
                # Extends the pilot signal for all station_schedule so that they all have the same length.
                self.pilot_signals[station_id] = _overwrite_at_index(self._iteration, self.pilot_signals[station_id],
                                                                     [0] * schedule_length)

    def _expand_pilots(self):
        """ Extends all pilot signals by appending 0's so they at least last past the next time step."""
        for signal in self.pilot_signals.values():
            if len(signal) < self._iteration + 1:
                signal.append(0)

    def _store_actual_charging_rates(self):
        """ Store actual charging rates from the network in the simulator for later analysis."""
        current_rates = self.network.current_charging_rates
        agg = 0
        for station_id, rate in current_rates.items():
            self.charging_rates[station_id].append(rate)
            agg += rate
        self.peak = max(self.peak, agg)

    def _print(self, s):
        if self.verbose:
            print(s)


class InvalidScheduleError(Exception):
    """ Raised when the schedule passed to the simulator is invalid. """
    pass


def _overwrite_at_index(i, prev_list, new_list):
    """ Returns a new list with the contents of prev_list up to index i and of new_list afterward.

    Args:
        i (int): Index of the transition between prev_list and new_list. i is exclusive.
        prev_list (List[]): List which will make up the first part of the new list.
        new_list (List[]): List which will make up the second part of the new list.

    Returns:
        List[]
    """
    if len(prev_list) < i:
        return prev_list + [0] * (i - len(prev_list)) + list(new_list)
    if len(prev_list) == i:
        return prev_list + list(new_list)
    else:
        return prev_list[:i] + list(new_list)
