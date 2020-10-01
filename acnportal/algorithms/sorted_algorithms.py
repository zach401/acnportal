from collections import deque
from copy import copy

import numpy as np
from .base_algorithm import BaseAlgorithm
from .utils import infrastructure_constraints_feasible
from .postprocessing import format_array_schedule
from .preprocessing import (
    enforce_pilot_limit,
    apply_upper_bound_estimate,
    apply_minimum_charging_rate,
    remove_finished_sessions
)
from warnings import warn


class SortedSchedulingAlgo(BaseAlgorithm):
    """ Class for sorting based algorithms like First Come First Served (FCFS) and Earliest Deadline First (EDF).

    Implements abstract class BaseAlgorithm.

    For this family of algorithms, active EVs are first sorted by some metric, then current is allocated to each EV in
    order. To allocate current we use a binary search approach which allocates each EV the maximum current possible
    subject to the constraints and already allocated allotments.

    The argument sort_fn controlled how the EVs are sorted and thus which sorting based algorithm is implemented.

    Args:
        sort_fn (Callable[List[EV]]): Function which takes in a list of EVs and returns a list of the same EVs but
            sorted according to some metric.
    """

    def __init__(
        self,
        sort_fn,
        estimate_max_rate=False,
        max_rate_estimator=None,
        uninterrupted_charging=False,
        allow_overcharging=False,
    ):
        super().__init__()
        self._sort_fn = sort_fn
        # Call algorithm each period since it only returns a rate for the
        # next period.
        self.max_recompute = 1
        self.estimate_max_rate = estimate_max_rate
        self.max_rate_estimator = max_rate_estimator
        self.uninterrupted_charging = uninterrupted_charging
        self.allow_overcharging = allow_overcharging

    def register_interface(self, interface):
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

    def sorting_algorithm(self, active_sessions, infrastructure):
        """ Schedule EVs by first sorting them by sort_fn, then allocating
            them their maximum feasible rate.

        Implements abstract method schedule from BaseAlgorithm.

        See class documentation for description of the algorithm.

        Args:
            active_sessions (List[SessionInfo]): see BaseAlgorithm
            infrastructure (InfrastructureInfo): Description of the electical
                infrastructure.

        Returns:
            np.array[Union[float, int]]: Array of charging rates, where each
                row is the charging rates for a specific session.
        """
        queue = self._sort_fn(active_sessions, self.interface)
        schedule = np.zeros(infrastructure.num_stations)

        # Start each EV at its lower bound
        for session in queue:
            station_index = infrastructure.get_station_index(session.station_id)
            lb = max(0, session.min_rates[0])
            schedule[station_index] = lb

        if not infrastructure_constraints_feasible(schedule, infrastructure):
            raise ValueError(
                "Charging all sessions at their lower bound " "is not feasible."
            )

        for session in queue:
            station_index = infrastructure.get_station_index(session.station_id)
            ub = min(
                session.max_rates[0], self.interface.remaining_amp_periods(session)
            )
            lb = max(0, session.min_rates[0])
            # ub = max(lb, session.max_rates[0])
            if infrastructure.is_continuous[station_index]:
                charging_rate = self.max_feasible_rate(
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
        station_index, ub, schedule, infrastructure, eps=0.0001, lb=0.0
    ):
        """ Return the maximum feasible rate less than ub subject to the environment's constraints.

        If schedule contains non-zero elements at the given time, these are
        treated as fixed allocations and this function will include them
        when determining the maximum feasible rate for the given EVSE.

        Args:
            station_index (int): Index for the station in the schedule
                vector.
            ub (float): Upper bound on the charging rate. [A]
            schedule (Dict[str, List[float]]): Dictionary mapping a station_id
                to a list of already fixed charging rates.
            infrastructure (InfrastructureInfo): Description of the electrical
                infrastructure.
            eps (float): Accuracy to which the max rate should be calculated.
                (When the binary search is terminated.)
            schedule (Dict[str, List[float]]): Dictionary mapping a station_id
                to a list of already fixed charging rates.
            infrastructure (InfrastructureInfo): Description of the electrical
                infrastructure.
            eps (float): Accuracy to which the max rate should be calculated.
                (When the binary search is terminated.)
            lb (float): Lower bound on the charging rate [A]

        Returns:
            float: maximum feasible rate less than ub subject to the
                environment's constraints. [A]
        """

        def bisection(_index, _lb, _ub, _schedule):
            """ Use the bisection method to find the maximum feasible charging
                rate for the EV. """
            mid = (_ub + _lb) / 2
            new_schedule = copy(schedule)
            new_schedule[_index] = mid
            if (_ub - _lb) <= eps:
                return _lb
            elif infrastructure_constraints_feasible(new_schedule, infrastructure):
                return bisection(_index, mid, _ub, new_schedule)
            else:
                return bisection(_index, _lb, mid, new_schedule)

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
        station_index, allowable_pilots, schedule, infrastructure
    ):
        """ Return the maximum feasible allowable rate subject to the
            infrastructure's constraints.

        If schedule contains non-zero elements at the given time, these are
        treated as fixed allocations and this function will include them
        when determining the maximum feasible rate for the given EVSE.

        Args:
            station_index (int): Index for the station in the schedule
                vector.
            allowable_pilots (List[float]): List of allowable charging rates
                sorted in ascending order.
            schedule (Dict[str, List[float]]): Dictionary mapping a station_id
                to a list of already fixed charging rates.
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

    def schedule(self, active_sessions):
        """ Schedule EVs by first sorting them by sort_fn, then allocating them their maximum feasible rate.

        Implements abstract method schedule from BaseAlgorithm.

        See class documentation for description of the algorithm.

        Args:
            active_sessions (List[EV]): see BaseAlgorithm

        Returns:
            Dict[str, List[float]]: see BaseAlgorithm
        """
        if self.allow_overcharging:
            warn("allow_overcharging is currently not supported. It will be added in a future release.")
        infrastructure = self.interface.infrastructure_info()
        active_sessions = remove_finished_sessions(active_sessions, infrastructure, self.interface.period)
        active_sessions = enforce_pilot_limit(active_sessions, infrastructure)
        if self.estimate_max_rate:
            active_sessions = apply_upper_bound_estimate(
                self.max_rate_estimator, active_sessions
            )
        if self.uninterrupted_charging:
            active_sessions = apply_minimum_charging_rate(
                active_sessions, infrastructure, self.interface.period
            )
        array_schedule = self.sorting_algorithm(active_sessions, infrastructure)
        return format_array_schedule(array_schedule, infrastructure)


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
    constrains become binding.

    Args:
        sort_fn (Callable[List[EV]]): Function which takes in a list of EVs
            and returns a list of the same EVs but sorted according to some
            metric.
        continuous_inc (float): Increment to use when pilot signal is
            continuously controllable.
    """

    def __init__(
        self,
        sort_fn,
        estimate_max_rate=False,
        max_rate_estimator=None,
        uninterrupted_charging=False,
        continuous_inc=0.1,
        allow_overcharging=False,
    ):
        super().__init__(
            sort_fn, estimate_max_rate, max_rate_estimator, uninterrupted_charging
        )
        self.continuous_inc = continuous_inc

    def round_robin(self, active_sessions, infrastructure):
        """ Schedule EVs using a round robin based equal sharing scheme.

        Implements abstract method schedule from BaseAlgorithm.

        See class documentation for description of the algorithm.

        Args:
            active_sessions (List[EV]): see BaseAlgorithm
            infrastructure (InfrastructureInfo): Description of electrical
                infrastructure.

        Returns:
            Dict[str, List[float]]: see BaseAlgorithm
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
                    session.max_rates[0] + 1e-7,
                    self.continuous_inc,
                )
            ub = min(
                session.max_rates[0],
                infrastructure.max_pilot[i],
                self.interface.remaining_amp_periods(session),
            )
            lb = max(0, session.min_rates[0])
            # Remove any charging rates which are not feasible.
            allowable_pilots[i] = [a for a in allowable_pilots[i] if lb <= a <= ub]
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


# -------------------- Sorting Functions --------------------------
def first_come_first_served(evs, iface):
    """ Sort EVs by arrival time in increasing order.

    Args:
        evs (List[EV]): List of EVs to be sorted.
        iface (Interface): Interface object. (not used in this case)

    Returns:
        List[EV]: List of EVs sorted by arrival time in increasing order.
    """
    return sorted(evs, key=lambda x: x.arrival)


def last_come_first_served(evs, iface):
    """ Sort EVs by arrival time in reverse order.
    Args:
       evs (List[EV]): List of EVs to be sorted.
       iface (Interface): Interface object. (not used in this case)
    Returns:
       List[EV]: List of EVs sorted by arrival time in decreasing order.
    """
    return sorted(evs, key=lambda x: x.arrival, reverse=True)


def earliest_deadline_first(evs, iface):
    """ Sort EVs by departure time in increasing order.

    Args:
        evs (List[EV]): List of EVs to be sorted.
        iface (Interface): Interface object. (not used in this case)

    Returns:
        List[EV]: List of EVs sorted by departure time in increasing order.
    """
    return sorted(evs, key=lambda x: x.departure)


def least_laxity_first(evs, iface):
    """ Sort EVs by laxity in increasing order.

    Laxity is a measure of the charging flexibility of an EV. Here we define laxity as:
        LAX_i(t) = (departure_i - t) - (remaining_demand_i(t) / max_rate_i)

    Args:
        evs (List[EV]): List of EVs to be sorted.
        iface (Interface): Interface object.

    Returns:
        List[EV]: List of EVs sorted by laxity in increasing order.
    """

    def laxity(ev):
        """ Calculate laxity of the EV.

        Args:
            ev (EV): An EV object.

        Returns:
            float: The laxity of the EV.
        """
        lax = (ev.departure - iface.current_time) - (
            iface.remaining_amp_periods(ev) / iface.max_pilot_signal(ev.station_id)
        )
        return lax

    return sorted(evs, key=laxity)


def largest_remaining_processing_time(evs, iface):
    """ Sort EVs in decreasing order by the time taken to finish charging them at the EVSE's maximum rate.

    # Args:
        evs (List[EV]): List of EVs to be sorted.
        iface (Interface): Interface object.

    Returns:
        List[EV]: List of EVs sorted by remaining processing time in decreasing order.
    """

    def remaining_processing_time(ev):
        """ Calculate minimum time needed to fully charge the EV based its remaining energy request and the EVSE's max
            charging rate.

        Args:
            ev (EV): An EV object.

        Returns:
            float: The minimum remaining processing time of the EV.
        """
        rpt = iface.remaining_amp_periods(ev) / iface.max_pilot_signal(ev.station_id)
        return rpt

    return sorted(evs, key=remaining_processing_time, reverse=True)
