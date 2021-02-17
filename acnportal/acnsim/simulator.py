import copy
from datetime import datetime
from typing import Dict

import warnings
import json

# noinspection PyProtectedMember
from pydoc import locate

from .events import *
from .events import UnplugEvent
from .interface import Interface
from .interface import InvalidScheduleError
from acnportal.algorithms import BaseAlgorithm
from .base import BaseSimObj


class Simulator(BaseSimObj):
    """ Central class of the acnsim package.

    The Simulator class is the central place where everything about a particular simulation is stored including the
    network, scheduling algorithm, and events. It is also where timekeeping is done and orchestrates calling the
    scheduling algorithm, sending pilots to the network, and updating the energy delivered to each EV.

    Args:
        network (ChargingNetwork): The charging network which the simulation will use.
        scheduler (BaseAlgorithm): The scheduling algorithm used in the simulation.
            If scheduler = None, Simulator.run() cannot be called.
        events (EventQueue): Queue of events which will occur in the simulation.
        start (datetime): Date and time of the first period of the simulation.
        period (float): Length of each time interval in the simulation in minutes. Default: 1
        signals (Dict[str, ...]):
        store_schedule_history (bool): If True, store the scheduler output each time it is run. Note this can use lots
            of memory for long simulations.
        interface_type (type): The class of interface to register with the scheduler.
    """

    period: float

    def __init__(
        self,
        network,
        scheduler,
        events,
        start,
        period: float = 1,
        signals=None,
        store_schedule_history=False,
        verbose=True,
        interface_type=Interface,
    ):
        self.network = network
        self.scheduler = scheduler
        self.max_recompute = None
        self.event_queue = events
        self.start = start
        self.period = period
        self.signals = signals
        self.verbose = verbose

        # Information storage
        width = 1
        if self.event_queue.get_last_timestamp() is not None:
            width = self.event_queue.get_last_timestamp() + 1
        self.pilot_signals = np.zeros((len(self.network.station_ids), width))
        self.charging_rates = np.zeros((len(self.network.station_ids), width))
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
        self._last_schedule_update = None

        # Interface registration is moved here so that copies of this
        # simulator have all attributes.
        if scheduler is not None:
            self.max_recompute = scheduler.max_recompute
            self.scheduler.register_interface(interface_type(self))

    @property
    def iteration(self):
        return self._iteration

    def run(self):
        """
        If scheduler is not None, run the simulation until the event queue is empty.

        The run function is the heart of the simulator. It triggers all actions and keeps the simulator moving forward.
        Its actions are (in order):
            1. Get current events from the event queue and execute them.
            2. If necessary run the scheduling algorithm.
            3. Send pilot signals to the network.
            4. Receive back actual charging rates from the network and store the results.

        Returns:
            None

        Raises:
            TypeError: If called when the scheduler attribute is None.
                The run() method requires a BaseAlgorithm-like
                scheduler to execute.
        """
        if self.scheduler is None:
            raise TypeError("Add a scheduler before attempting to call" " run().")
        while not self.event_queue.empty():
            current_events = self.event_queue.get_current_events(self._iteration)
            for e in current_events:
                self.event_history.append(e)
                self._process_event(e)
            if (
                self._resolve
                or self.max_recompute is not None
                and (
                    self._last_schedule_update is None
                    or self._iteration - self._last_schedule_update
                    >= self.max_recompute
                )
            ):
                new_schedule = self.scheduler.run()
                self._update_schedules(new_schedule)
                if self.schedule_history is not None:
                    self.schedule_history[self._iteration] = new_schedule
                self._last_schedule_update = self._iteration
                self._resolve = False
            if not self.event_queue.empty():
                width_increase = self.event_queue.get_last_timestamp() + 1
            else:
                width_increase = self._iteration + 1
            self.pilot_signals = _increase_width(self.pilot_signals, width_increase)
            self.charging_rates = _increase_width(self.charging_rates, width_increase)
            self.network.update_pilots(self.pilot_signals, self._iteration, self.period)
            self._store_actual_charging_rates()
            self.network.post_charging_update()
            self._iteration = self._iteration + 1

    def step(self, new_schedule):
        """ Step the simulation until the next schedule recompute is
        required.

        The step function executes a single iteration of the run()
        function. However, the step function updates the simulator with
        an input schedule rather than query the scheduler for a new
        schedule when one is required. Also, step will return a flag if
        the simulation is done.

        Args:
            new_schedule (Dict[str, List[number]]): Dictionary mapping
                station ids to a schedule of pilot signals.

        Returns:
            bool: True if the simulation is complete.
        """
        while (
            not self.event_queue.empty()
            and not self._resolve
            and (
                self.max_recompute is None
                or (self._iteration - self._last_schedule_update < self.max_recompute)
            )
        ):
            self._update_schedules(new_schedule)
            if self.schedule_history is not None:
                self.schedule_history[self._iteration] = new_schedule
            self._last_schedule_update = self._iteration
            self._resolve = False
            if self.event_queue.get_last_timestamp() is not None:
                width_increase = max(
                    self.event_queue.get_last_timestamp() + 1, self._iteration + 1
                )
            else:
                width_increase = self._iteration + 1
            self.pilot_signals = _increase_width(self.pilot_signals, width_increase)
            self.charging_rates = _increase_width(self.charging_rates, width_increase)
            self.network.update_pilots(self.pilot_signals, self._iteration, self.period)
            self._store_actual_charging_rates()
            self.network.post_charging_update()
            self._iteration = self._iteration + 1
            current_events = self.event_queue.get_current_events(self._iteration)
            for e in current_events:
                self.event_history.append(e)
                self._process_event(e)
        return self.event_queue.empty()

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
        if event.event_type == "Plugin":
            self._print("Plugin Event...")
            self.network.plugin(event.ev)
            self.ev_history[event.ev.session_id] = event.ev
            self.event_queue.add_event(UnplugEvent(event.ev.departure, event.ev))
            self._resolve = True
            self._last_schedule_update = event.timestamp
        elif event.event_type == "Unplug":
            self._print("Unplug Event...")
            self.network.unplug(event.ev.station_id, event.ev.session_id)
            self._resolve = True
            self._last_schedule_update = event.timestamp
        elif event.event_type == "Recompute":
            self._print("Recompute Event...")
            self._resolve = True

    def _update_schedules(self, new_schedule):
        """ Extend the current self.pilot_signals with the new pilot signal schedule.

        Args:
            new_schedule (Dict[str, List[number]]): Dictionary mapping station ids to a schedule of pilot signals.

        Returns:
            None

        Raises:
            KeyError: Raised when station_id is in the new_schedule but not registered in the Network.
        """
        if len(new_schedule) == 0:
            return

        for station_id in new_schedule:
            if station_id not in self.network.station_ids:
                raise KeyError(
                    "Station {0} in schedule but not found in network.".format(
                        station_id
                    )
                )

        schedule_lengths = set(len(x) for x in new_schedule.values())
        if len(schedule_lengths) > 1:
            raise InvalidScheduleError("All schedules should have the same length.")
        schedule_length = schedule_lengths.pop()

        schedule_matrix = np.array(
            [
                new_schedule[evse_id]
                if evse_id in new_schedule
                else [0] * schedule_length
                for evse_id in self.network.station_ids
            ]
        )
        if not self.network.is_feasible(schedule_matrix):
            aggregate_currents = self.network.constraint_current(schedule_matrix)
            diff_vec = (
                np.abs(aggregate_currents)
                - np.tile(
                    self.network.magnitudes + self.network.violation_tolerance,
                    (schedule_length, 1),
                ).T
            )
            max_idx = np.unravel_index(np.argmax(diff_vec), diff_vec.shape)
            max_diff = diff_vec[max_idx]
            max_timeidx = max_idx[1]
            max_constraint = self.network.constraint_index[max_idx[0]]
            warnings.warn(
                f"Invalid schedule provided at iteration {self._iteration}. "
                f"Max violation is {max_diff} A on {max_constraint} "
                f"at time index {max_timeidx}.",
                UserWarning,
            )
        if self._iteration + schedule_length <= self.pilot_signals.shape[1]:
            self.pilot_signals[
                :, self._iteration : (self._iteration + schedule_length)
            ] = schedule_matrix
        else:
            # We've reached the end of pilot_signals, so double pilot_signal array width
            self.pilot_signals = _increase_width(
                self.pilot_signals,
                max(
                    self.event_queue.get_last_timestamp() + 1,
                    self._iteration + schedule_length,
                ),
            )
            self.pilot_signals[
                :, self._iteration : (self._iteration + schedule_length)
            ] = schedule_matrix

    def _store_actual_charging_rates(self):
        """ Store actual charging rates from the network in the simulator for later analysis."""
        current_rates = self.network.current_charging_rates
        agg = np.sum(current_rates)
        if self.iteration < self.charging_rates.shape[1]:
            self.charging_rates[:, self.iteration] = current_rates.T
        else:
            if not self.event_queue.empty():
                width_increase = self.event_queue.get_last_timestamp() + 1
            else:
                width_increase = self._iteration + 1
            self.charging_rates = _increase_width(self.charging_rates, width_increase)
            self.charging_rates[:, self._iteration] = current_rates.T
        self.peak = max(self.peak, agg)

    def _print(self, s):
        if self.verbose:
            print(s)

    def charging_rates_as_df(self):
        """ Return the charging rates as a pandas DataFrame, with EVSE id as columns
        and iteration as index.

        Returns:
            pandas.DataFrame: A DataFrame containing the charging rates
                of the simulation. Columns are EVSE id, and the index is
                the iteration.
        """
        return pd.DataFrame(
            data=self.charging_rates.T, columns=self.network.station_ids
        )

    def pilot_signals_as_df(self):
        """ Return the pilot signals as a pandas DataFrame

        Returns:
            pandas.DataFrame: A DataFrame containing the pilot signals
                of the simulation. Columns are EVSE id, and the index is
                the iteration.
        """
        return pd.DataFrame(data=self.pilot_signals.T, columns=self.network.station_ids)

    def index_of_evse(self, station_id):
        """ Return the numerical index of the EVSE given by station_id in the (ordered) dictionary
        of EVSEs.
        """
        if station_id not in self.network.station_ids:
            raise KeyError("EVSE {0} not found in network.".format(station_id))
        return self.network.station_ids.index(station_id)

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Implements BaseSimObj._to_dict. Certain simulator attributes are
        not serialized completely as they are not ACN-Sim objects
        (signals and scheduler exist in their own modules).

        If the Python version used is less than 3.7, datetimes cannot be
        accurately loaded. As such, a warning is thrown when the start
        attribute is serialized.

        The signals attribute is only serialized if it is natively
        JSON Serializable, otherwise None is stored.

        Only the scheduler's name is serialized.
        """
        attribute_dict = {}

        # noinspection PyProtectedMember
        registry, context_dict = self.network._to_registry(context_dict=context_dict)
        attribute_dict["network"] = registry["id"]

        registry, context_dict = self.event_queue._to_registry(
            context_dict=context_dict
        )
        attribute_dict["event_queue"] = registry["id"]

        attribute_dict["scheduler"] = (
            f"{self.scheduler.__module__}." f"{self.scheduler.__class__.__name__}"
        )

        attribute_dict["start"] = self.start.strftime("%H:%M:%S.%f %d%m%Y")

        try:
            json.dumps(self.signals)
        except TypeError:
            warnings.warn(
                "Not serializing signals as value types"
                "are not natively JSON serializable.",
                UserWarning,
            )
            attribute_dict["signals"] = None
        else:
            attribute_dict["signals"] = self.signals

        # Serialize non-nested attributes.
        nn_attr_lst = [
            "period",
            "max_recompute",
            "verbose",
            "peak",
            "_iteration",
            "_resolve",
            "_last_schedule_update",
            "schedule_history",
            "pilot_signals",
            "charging_rates",
        ]
        for attr in nn_attr_lst:
            attribute_dict[attr] = getattr(self, attr)

        ev_history = {}
        for session_id, ev in self.ev_history.items():
            # noinspection PyProtectedMember
            registry, context_dict = ev._to_registry(context_dict=context_dict)
            ev_history[session_id] = registry["id"]
        attribute_dict["ev_history"] = ev_history

        event_history = []
        for past_event in self.event_history:
            # noinspection PyProtectedMember
            registry, context_dict = past_event._to_registry(context_dict=context_dict)
            event_history.append(registry["id"])
        attribute_dict["event_history"] = event_history

        return attribute_dict, context_dict

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, BaseSimObj]] = None,
    ) -> Tuple[BaseSimObj, Dict[str, BaseSimObj]]:
        """
        Implements BaseSimObj._from_dict. Certain simulator attributes
        are not loaded completely as they are not ACN-Sim objects
        (signals and scheduler exist in their own modules).

        If the Python version used is less than 3.7, the start attribute
        is stored in ISO format instead of datetime, and a warning is
        thrown.

        The signals attribute is only loaded if it was natively
        JSON Serializable, in the original object, otherwise None is
        set as the signals attribute. The Simulator's signals can be set
        after the Simulator is loaded.

        The scheduler attribute is only accurate if the scheduler's
        constructor takes no arguments, otherwise BaseAlgorithm is
        stored. The Simulator provides a method to set the scheduler
        after the Simulator is loaded.

        """
        # noinspection PyProtectedMember
        network, loaded_dict = BaseSimObj._build_from_id(
            attribute_dict["network"], context_dict, loaded_dict=loaded_dict
        )

        # noinspection PyProtectedMember
        events, loaded_dict = BaseSimObj._build_from_id(
            attribute_dict["event_queue"], context_dict, loaded_dict=loaded_dict
        )

        scheduler_cls = locate(attribute_dict["scheduler"])
        try:
            scheduler = scheduler_cls()
        except TypeError:
            warnings.warn(
                f"Scheduler {attribute_dict['scheduler']} "
                f"requires constructor inputs. Setting "
                f"scheduler to BaseAlgorithm instead."
            )
            scheduler = BaseAlgorithm()

        start = datetime.strptime(attribute_dict["start"], "%H:%M:%S.%f %d%m%Y")

        out_obj = cls(
            network,
            scheduler,
            events,
            start,
            period=attribute_dict["period"],
            signals=attribute_dict["signals"],
            verbose=attribute_dict["verbose"],
        )
        scheduler.register_interface(Interface(out_obj))

        attr_lst = [
            "max_recompute",
            "peak",
            "_iteration",
            "_resolve",
            "_last_schedule_update",
        ]
        for attr in attr_lst:
            setattr(out_obj, attr, attribute_dict[attr])

        if attribute_dict["schedule_history"] is not None:
            out_obj.schedule_history = {
                int(key): value
                for key, value in attribute_dict["schedule_history"].items()
            }
        else:
            out_obj.schedule_history = None

        out_obj.pilot_signals = np.array(attribute_dict["pilot_signals"])
        out_obj.charging_rates = np.array(attribute_dict["charging_rates"])

        ev_history = {}
        for session_id, ev in attribute_dict["ev_history"].items():
            # noinspection PyProtectedMember
            ev_elt, loaded_dict = BaseSimObj._build_from_id(
                ev, context_dict, loaded_dict=loaded_dict
            )
            ev_history[session_id] = ev_elt
        out_obj.ev_history = ev_history

        event_history = []
        for past_event in attribute_dict["event_history"]:
            # noinspection PyProtectedMember
            loaded_event, loaded_dict = BaseSimObj._build_from_id(
                past_event, context_dict, loaded_dict=loaded_dict
            )
            event_history.append(loaded_event)
        out_obj.event_history = event_history

        return out_obj, loaded_dict

    def update_scheduler(self, new_scheduler):
        """ Updates a Simulator's schedule. """
        self.scheduler = new_scheduler
        self.scheduler.register_interface(Interface(self))
        self.max_recompute = new_scheduler.max_recompute


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
    new_matrix[:, : a.shape[1]] = a
    return new_matrix
