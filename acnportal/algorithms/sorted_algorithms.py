# coding=utf-8
"""
Sorting-based scheduling algorithms.
"""
from collections import deque
from copy import copy
from typing import Callable, List, Optional, Dict

import numpy as np

from .upper_bound_estimator import UpperBoundEstimatorBase
from .base_algorithm import BaseAlgorithm
from .utils import infrastructure_constraints_feasible
from .postprocessing import format_array_schedule
from .preprocessing import (
    enforce_pilot_limit,
    apply_upper_bound_estimate,
    apply_minimum_charging_rate,
    remove_finished_sessions,
)
from warnings import warn

from acnportal.acnsim.interface import SessionInfo, InfrastructureInfo, Interface


class SortedSchedulingAlgo(BaseAlgorithm):
    """ Class for sorting based algorithms like First Come First Served (FCFS) and
    Earliest Deadline First (EDF).

    Implements abstract class BaseAlgorithm.

    For this family of algorithms, active EVs are first sorted by some metric, then
    current is allocated to each EV in order. To allocate current we use a binary
    search approach which allocates each EV the maximum current possible subject to the
    constraints and already allocated allotments.

    The argument sort_fn controlled how the EVs are sorted and thus which sorting based
    algorithm is implemented.

    Args:
        sort_fn (Callable[[List[SessionInfo], Interface], List[SessionInfo]]): Function
            which takes in a list of SessionInfo objects and returns a list of the
            same SessionInfo objects but sorted according to some metric.
    """

    _sort_fn: Callable[[List[SessionInfo], Interface], List[SessionInfo]]
    estimate_max_rate: bool
    max_rate_estimator: Optional[UpperBoundEstimatorBase]
    uninterrupted_charging: bool
    allow_overcharging: bool

    def __init__(
        self,
        sort_fn: Callable[[List[SessionInfo], Interface], List[SessionInfo]],
        estimate_max_rate: bool = False,
        max_rate_estimator: Optional[UpperBoundEstimatorBase] = None,
        uninterrupted_charging: bool = False,
        allow_overcharging: bool = False,
    ) -> None:
        super().__init__()
        self._sort_fn = sort_fn
        # Call algorithm each period since it only returns a rate for the
        # next period.
        self.max_recompute = 1
        self.estimate_max_rate = estimate_max_rate
        self.max_rate_estimator = max_rate_estimator
        self.uninterrupted_charging = uninterrupted_charging
        self.allow_overcharging = allow_overcharging

    def register_interface(self, interface: Interface) -> None:
        """ Register interface to the _simulator/physical system.

        This interface is the only connection between the algorithm and what it
            is controlling. Its purpose is to abstract the underlying
            network so that the same algorithms can run on a simulated
            environment or a physical one.

        Args:
            interface (Interface): An interface to the underlying network
                whether simulated or real.

        Returns:
            None
        """
        self._interface = interface
        if self.max_rate_estimator is not None:
            self.max_rate_estimator.register_interface(interface)

    def run_preprocessing(
        self, active_sessions: List[SessionInfo], infrastructure: InfrastructureInfo
    ) -> List[SessionInfo]:
        """ Run a set of preprocessing functions on the active_sessions given to the
        algorithm.

        Args:
            active_sessions (List[SessionInfo]): see BaseAlgorithm
            infrastructure (InfrastructureInfo): Description of the electrical
                infrastructure.

        Returns:
            List[SessionInfo]: A list of processed SessionInfo objects.

        """
        if self.allow_overcharging:
            warn(
                "allow_overcharging is currently not supported. It will be added in a "
                "future release."
            )
        active_sessions: List[SessionInfo] = remove_finished_sessions(
            active_sessions, infrastructure, self.interface.period
        )
        active_sessions = enforce_pilot_limit(active_sessions, infrastructure)
        if self.estimate_max_rate:
            active_sessions: List[SessionInfo] = apply_upper_bound_estimate(
                self.max_rate_estimator, active_sessions
            )
        if self.uninterrupted_charging:
            active_sessions: List[SessionInfo] = apply_minimum_charging_rate(
                active_sessions, infrastructure, self.interface.period
            )
        return active_sessions

    def sorting_algorithm(
        self, active_sessions: List[SessionInfo], infrastructure: InfrastructureInfo
    ) -> np.ndarray:
        """ Schedule EVs by first sorting them by sort_fn, then allocating
            them their maximum feasible rate.

        See class documentation for description of the algorithm.

        Args:
            active_sessions (List[SessionInfo]): see BaseAlgorithm
            infrastructure (InfrastructureInfo): Description of the electrical
                infrastructure.

        Returns:
            np.array[Union[float, int]]: Array of charging rates, where each
                row is the charging rates for a specific session.
        """
        queue: List[SessionInfo] = self._sort_fn(active_sessions, self.interface)
        schedule: np.ndarray = np.zeros(infrastructure.num_stations)

        # Start each EV at its lower bound
        for session in queue:
            station_index: int = infrastructure.get_station_index(session.station_id)
            lb: float = max(0, session.min_rates[0])
            schedule[station_index] = lb

        if not infrastructure_constraints_feasible(schedule, infrastructure):
            raise ValueError(
                "Charging all sessions at their lower bound is not feasible."
            )

        for session in queue:
            station_index = infrastructure.get_station_index(session.station_id)
            ub: float = min(
                session.max_rates[0], self.interface.remaining_amp_periods(session)
            )
            lb: float = max(0, session.min_rates[0])
            if infrastructure.is_continuous[station_index]:
                charging_rate: float = self.max_feasible_rate(
                    station_index, ub, schedule, infrastructure, eps=0.01, lb=lb
                )
            else:
                allowable = [
                    a
                    for a in infrastructure.allowable_pilots[station_index]
                    if lb <= a <= ub
                ]

                if len(allowable) == 0:
                    charging_rate = 0
                else:
                    charging_rate = self.discrete_max_feasible_rate(
                        station_index, allowable, schedule, infrastructure
                    )
            schedule[station_index] = charging_rate
        return schedule

    @staticmethod
    def max_feasible_rate(
        station_index: int,
        ub: float,
        schedule: np.ndarray,
        infrastructure: InfrastructureInfo,
        eps: float = 0.0001,
        lb: float = 0.0,
    ) -> float:
        """ Return the maximum feasible rate less than ub subject to the environment's
        constraints.

        If schedule contains non-zero elements at the given time, these are
        treated as fixed allocations and this function will include them
        when determining the maximum feasible rate for the given EVSE.

        Args:
            station_index (int): Index for the station in the schedule
                vector.
            ub (float): Upper bound on the charging rate. [A]
            schedule (np.array[Union[float, int]]): Array of charging rates, where each
                row is the charging rates for a specific session.
            infrastructure (InfrastructureInfo): Description of the electrical
                infrastructure.
            eps (float): Accuracy to which the max rate should be calculated.
                (When the binary search is terminated.)
            lb (float): Lower bound on the charging rate [A]

        Returns:
            float: maximum feasible rate less than ub subject to the
                environment's constraints. [A]
        """

        def bisection(
            _index: int, _lb: float, _ub: float, _schedule: np.ndarray
        ) -> float:
            """ Use the bisection method to find the maximum feasible charging
                rate for the EV. """
            mid: float = (_ub + _lb) / 2
            _new_schedule = copy(schedule)
            _new_schedule[_index] = mid
            if (_ub - _lb) <= eps:
                return _lb
            elif infrastructure_constraints_feasible(_new_schedule, infrastructure):
                return bisection(_index, mid, _ub, _new_schedule)
            else:
                return bisection(_index, _lb, mid, _new_schedule)

        if not infrastructure_constraints_feasible(schedule, infrastructure):
            raise ValueError("The initial schedule is not feasible.")

        # Test maximum rate to short-circuit bisection
        new_schedule = copy(schedule)
        new_schedule[station_index] = ub
        if infrastructure_constraints_feasible(new_schedule, infrastructure):
            return ub
        return bisection(station_index, lb, ub, schedule)

    @staticmethod
    def discrete_max_feasible_rate(
        station_index: int,
        allowable_pilots: List[float],
        schedule: np.ndarray,
        infrastructure: InfrastructureInfo,
    ) -> float:
        """ Return the maximum feasible allowable rate subject to the
            infrastructure's constraints and the discrete pilot constraints of this
            station.

        If schedule contains non-zero elements at the given time, these are
        treated as fixed allocations and this function will include them
        when determining the maximum feasible rate for the given EVSE.

        Args:
            station_index (int): Index for the station in the schedule
                vector.
            allowable_pilots (List[float]): List of allowable charging rates
                sorted in ascending order.
            schedule (np.array[Union[float, int]]): Array of charging rates, where each
                row is the charging rates for a specific session.
            infrastructure (InfrastructureInfo): Description of the electrical
                infrastructure.

        Returns:
            float: maximum feasible rate less than ub subject to the
                infrastructure's constraints. [A]
        """
        if not infrastructure_constraints_feasible(schedule, infrastructure):
            raise ValueError("The initial schedule is not feasible.")
        new_schedule = copy(schedule)
        feasible_idx = len(allowable_pilots) - 1
        new_schedule[station_index] = allowable_pilots[feasible_idx]
        while not infrastructure_constraints_feasible(new_schedule, infrastructure):
            feasible_idx -= 1
            if feasible_idx < 0:
                new_schedule[station_index] = 0
                break
            else:
                new_schedule[station_index] = allowable_pilots[feasible_idx]
        return new_schedule[station_index]

    # noinspection PyMethodMayBeStatic
    def run_postprocessing(
        self, raw_schedule: np.ndarray, infrastructure: InfrastructureInfo
    ) -> Dict[str, List[float]]:
        """ Run a set of postprocessing functions on the schedule returned by the
        algorithm

        Args:
            raw_schedule (np.ndarray): An unprocessed schedule returned by a step of the
                algorithm.
            infrastructure (InfrastructureInfo): Description of the electrical
                infrastructure.

        Returns:
            Dict[str, List[float]]: Output schedule in a Simulator-accepted form (see
                BaseAlgorithm.schedule).

        """
        return format_array_schedule(raw_schedule, infrastructure)

    def schedule(self, active_sessions: List[SessionInfo]) -> Dict[str, List[float]]:
        """ Schedule EVs by first sorting them by sort_fn, then allocating them their
        maximum feasible rate.

        Implements abstract method schedule from BaseAlgorithm.

        See class documentation for description of the algorithm.

        Args:
            active_sessions (List[SessionInfo]): see BaseAlgorithm

        Returns:
            Dict[str, List[float]]: see BaseAlgorithm
        """
        infrastructure = self.interface.infrastructure_info()
        active_sessions = self.run_preprocessing(active_sessions, infrastructure)
        array_schedule = self.sorting_algorithm(active_sessions, infrastructure)
        return self.run_postprocessing(array_schedule, infrastructure)


class RoundRobin(SortedSchedulingAlgo):
    """ Family of algorithms which allocate charging rates among active EVs
        using a round robin approach.

    Extends SortingAlgorithm.

    For this family of algorithms EVs are first sorted as in SortingAlgorithm.
    The difference however, is that instead of allocating each EV its
    maximum charging rate as we go down the list, we instead give each EV
    one  unit of charge if it is feasible to do so. When it ceases to be
    feasible to give an EV more charge, it is removed from the list.
    This process continues until the list of EVs is empty.

    The argument sort_fn controlled how the EVs are sorted. This controls
    which  EVs will get potential higher charging rates when infrastructure
    constraints become binding.

    Args:
        sort_fn (Callable[[List[SessionInfo], Interface], List[SessionInfo]]): Function
            which takes in a list of EVs and returns a list of the same EVs but sorted
            according to some metric.
        continuous_inc (float): Increment to use when pilot signal is
            continuously controllable.
    """

    continuous_inc: float

    def __init__(
        self,
        sort_fn: Callable[[List[SessionInfo], Interface], List[SessionInfo]],
        estimate_max_rate: bool = False,
        max_rate_estimator: Optional[UpperBoundEstimatorBase] = None,
        uninterrupted_charging: bool = False,
        continuous_inc: float = 0.1,
        allow_overcharging: bool = False,
    ) -> None:
        super().__init__(
            sort_fn,
            estimate_max_rate,
            max_rate_estimator,
            uninterrupted_charging,
            allow_overcharging,
        )
        self.continuous_inc = continuous_inc

    def round_robin(
        self, active_sessions: List[SessionInfo], infrastructure: InfrastructureInfo
    ) -> np.ndarray:
        """ Schedule EVs using a round robin based equal sharing scheme.

        Implements abstract method schedule from BaseAlgorithm.

        See class documentation for description of the algorithm.

        Args:
            active_sessions (List[SessionInfo]): see BaseAlgorithm
            infrastructure (InfrastructureInfo): Description of electrical
                infrastructure.

        Returns:
            np.array[Union[float, int]]: Array of charging rates, where each
                row is the charging rates for a specific session.
        """
        queue = deque(self._sort_fn(active_sessions, self.interface))
        schedule = np.zeros(infrastructure.num_stations)
        rate_idx = np.zeros(infrastructure.num_stations, dtype=int)
        allowable_pilots = infrastructure.allowable_pilots.copy()
        for session in queue:
            i = infrastructure.get_station_index(session.station_id)
            # If pilot signal is continuous discretize it with increments of
            # continuous_inc.
            if infrastructure.is_continuous[i]:
                allowable_pilots[i] = np.arange(
                    session.min_rates[0],
                    session.max_rates[0] + self.continuous_inc / 2,
                    self.continuous_inc,
                )
            ub = min(
                session.max_rates[0],
                infrastructure.max_pilot[i],
                self.interface.remaining_amp_periods(session),
            )
            lb = max(0, session.min_rates[0])
            # Remove any charging rates which are not feasible.
            allowable_pilots[i] = allowable_pilots[i][lb <= allowable_pilots[i]]
            allowable_pilots[i] = allowable_pilots[i][allowable_pilots[i] <= ub]
            # All charging rates should start at their lower bound
            schedule[i] = allowable_pilots[i][0] if len(allowable_pilots[i]) > 0 else 0

        if not infrastructure_constraints_feasible(schedule, infrastructure):
            raise ValueError(
                "Charging all sessions at their lower bound is not feasible."
            )

        while len(queue) > 0:
            session = queue.popleft()
            i = infrastructure.get_station_index(session.station_id)
            if rate_idx[i] < len(allowable_pilots[i]) - 1:
                schedule[i] = allowable_pilots[i][rate_idx[i] + 1]
                if infrastructure_constraints_feasible(schedule, infrastructure):
                    rate_idx[i] += 1
                    queue.append(session)
                else:
                    schedule[i] = allowable_pilots[i][rate_idx[i]]
        return schedule

    def schedule(self, active_sessions: List[SessionInfo]) -> Dict[str, List[float]]:
        """ Schedule EVs by first sorting them by sort_fn, then allocating them their
        maximum feasible rate.

        Implements abstract method schedule from BaseAlgorithm.

        See class documentation for description of the algorithm.

        Args:
            active_sessions (List[SessionInfo]): see BaseAlgorithm

        Returns:
            Dict[str, List[float]]: see BaseAlgorithm
        """
        infrastructure = self.interface.infrastructure_info()
        active_sessions = self.run_preprocessing(active_sessions, infrastructure)
        array_schedule = self.round_robin(active_sessions, infrastructure)
        return self.run_postprocessing(array_schedule, infrastructure)


# -------------------- Sorting Functions --------------------------
# noinspection PyUnusedLocal
def first_come_first_served(
    evs: List[SessionInfo], iface: Interface
) -> List[SessionInfo]:
    """ Sort EVs by arrival time in increasing order.

    Args:
        evs (List[SessionInfo]): List of EVs to be sorted.
        iface (Interface): Interface object. (not used in this case)

    Returns:
        List[SessionInfo]: List of EVs sorted by arrival time in increasing order.
    """
    return sorted(evs, key=lambda x: x.arrival)


# noinspection PyUnusedLocal
def last_come_first_served(
    evs: List[SessionInfo], iface: Interface
) -> List[SessionInfo]:
    """ Sort EVs by arrival time in reverse order.
    Args:
       evs (List[SessionInfo]): List of EVs to be sorted.
       iface (Interface): Interface object. (not used in this case)
    Returns:
       List[SessionInfo]: List of EVs sorted by arrival time in decreasing order.
    """
    return sorted(evs, key=lambda x: x.arrival, reverse=True)


# noinspection PyUnusedLocal
def earliest_deadline_first(
    evs: List[SessionInfo], iface: Interface
) -> List[SessionInfo]:
    """ Sort EVs by estimated departure time in increasing order.

    Args:
        evs (List[SessionInfo]): List of EVs to be sorted.
        iface (Interface): Interface object. (not used in this case)

    Returns:
        List[SessionInfo]: List of EVs sorted by estimated departure time in increasing order.
    """
    return sorted(evs, key=lambda x: x.estimated_departure)


def least_laxity_first(evs: List[SessionInfo], iface: Interface) -> List[SessionInfo]:
    """ Sort EVs by laxity in increasing order.

    Laxity is a measure of the charging flexibility of an EV. Here we define laxity as:
        LAX_i(t) = (estimated_departure_i - t) - (remaining_demand_i(t) / max_rate_i)

    Args:
        evs (List[SessionInfo]): List of EVs to be sorted.
        iface (Interface): Interface object.

    Returns:
        List[SessionInfo]: List of EVs sorted by laxity in increasing order.
    """

    def laxity(ev: SessionInfo) -> float:
        """ Calculate laxity of the EV.

        Args:
            ev (EV): An EV object.

        Returns:
            float: The laxity of the EV.
        """
        lax = (ev.estimated_departure - iface.current_time) - (
            iface.remaining_amp_periods(ev) / iface.max_pilot_signal(ev.station_id)
        )
        return lax

    return sorted(evs, key=laxity)


def largest_remaining_processing_time(
    evs: List[SessionInfo], iface: Interface
) -> List[SessionInfo]:
    """ Sort EVs in decreasing order by the time taken to finish charging them at the
    EVSE's maximum rate.

    Args:
        evs (List[SessionInfo]): List of SessionInfo objects to be sorted.
        iface (Interface): Interface object.

    Returns:
        List[SessionInfo]: List of EVs sorted by remaining processing time in
            decreasing order.
    """

    def remaining_processing_time(ev: SessionInfo) -> float:
        """ Calculate minimum time needed to fully charge the EV based its remaining
        energy request and the EVSE's max charging rate.

        Args:
            ev (SessionInfo): A SessionInfo object.

        Returns:
            float: The minimum remaining processing time of the EV.
        """
        rpt = iface.remaining_amp_periods(ev) / iface.max_pilot_signal(ev.station_id)
        return rpt

    return sorted(evs, key=remaining_processing_time, reverse=True)
