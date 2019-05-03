import math
from collections import deque
from copy import copy

import numpy as np
from .base_algorithm import BaseAlgorithm


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

    def __init__(self, sort_fn):
        super().__init__()
        self._sort_fn = sort_fn

    def schedule(self, active_evs):
        """ Schedule EVs by first sorting them by sort_fn, then allocating them their maximum feasible rate.

        Implements abstract method schedule from BaseAlgorithm.

        See class documentation for description of the algorithm.

        Args:
            active_evs (List[EV]): see BaseAlgorithm

        Returns:
            Dict[str, List[float]]: see BaseAlgorithm
        """
        ev_queue = self._sort_fn(active_evs, self.interface)
        schedule = {ev.station_id: [0] for ev in active_evs}
        for ev in ev_queue:
            continuous, allowable_rates = self.interface.allowable_pilot_signals(ev.station_id)
            if continuous:
                charging_rate = self.max_feasible_rate(ev.station_id, allowable_rates[-1], schedule, eps=0.01)
            else:
                charging_rate = self.discrete_max_feasible_rate(ev.station_id, allowable_rates, schedule)
            schedule[ev.station_id][0] = charging_rate
        return schedule

    def max_feasible_rate(self, station_id, ub, schedule, time=0, eps=0.01):
        """ Return the maximum feasible rate less than ub subject to the environment's constraints.

        If schedule contains non-zero elements at the given time, these are treated as fixed allocations and this
        function will include them when determining the maximum feasible rate for the given EVSE.

        Args:
            station_id (str): ID for the station we are finding the maximum feasible rate for.
            ub (float): Upper bound on the charging rate. [A]
            schedule (Dict[str, List[float]]): Dictionary mapping a station_id to a list of already fixed
                charging rates.
            time (int): Time interval for which the max rate should be calculated.
            eps (float): Accuracy to which the max rate should be calculated. (When the binary search is terminated.)

        Returns:
            float: maximum feasible rate less than ub subject to the environment's constraints. [A]
        """
        def bisection(_station_id, _lb, _ub, _schedule):
            """ Use the bisection method to find the maximum feasible charging rate for the EV. """
            mid = (_ub + _lb) / 2
            new_schedule = copy(_schedule)
            new_schedule[_station_id][time] = mid
            if (_ub - _lb) <= eps:
                return _lb
            elif self.interface.is_feasible(new_schedule, time):
                return bisection(_station_id, mid, _ub, new_schedule)
            else:
                return bisection(_station_id, _lb, mid, new_schedule)

        if not self.interface.is_feasible(schedule):
            raise ValueError('The initial schedule is not feasible.')
        return bisection(station_id, 0, ub, schedule)

    def discrete_max_feasible_rate(self, station_id, allowable_rates, schedule, time=0):
        """ Return the maximum feasible allowable rate subject to the environment's constraints.

        If schedule contains non-zero elements at the given time, these are treated as fixed allocations and this
        function will include them when determining the maximum feasible rate for the given EVSE.

        Args:
            station_id (str): ID for the station we are finding the maximum feasible rate for.
            allowable_rates List[float]: List of allowable charging rates sorted in ascending order.
            schedule (Dict[str, List[float]]): Dictionary mapping a station_id to a list of already fixed
                charging rates.
            time (int): Time interval for which the max rate should be calculated.

        Returns:
            float: maximum feasible rate less than ub subject to the environment's constraints. [A]
        """
        if not self.interface.is_feasible(schedule):
            raise ValueError('The initial schedule is not feasible.')
        new_schedule = copy(schedule)
        feasible_idx = len(allowable_rates) - 1
        new_schedule[station_id][time] = allowable_rates[feasible_idx]
        while not self.interface.is_feasible(new_schedule):
            feasible_idx -= 1
            if feasible_idx < 0:
                new_schedule[station_id][time] = 0
                break
            else:
                new_schedule[station_id][time] = allowable_rates[feasible_idx]
        return new_schedule[station_id][time]


class RoundRobin(SortedSchedulingAlgo):
    """ Family of algorithms which allocate charging rates among active EVs using a round robin approach.

    Extends SortingAlgorithm.

    For this family of algorithms EVs are first sorted as in SortingAlgorithm. The difference however, is that instead
    of allocating each EV its maximum charging rate as we go down the list, we instead give each EV one unit of charge
    if it is feasible to do so. When it ceases to be feasible to give an EV more charge, it is removed from the list.
    This process continues until the list of EVs is empty.

    The argument sort_fn controlled how the EVs are sorted. This controls which EVs will get potential higher charging
    rates when infrastructure constrains become binding.

    Args:
        sort_fn (Callable[List[EV]]): Function which takes in a list of EVs and returns a list of the same EVs but
            sorted according to some metric.
    """

    def schedule(self, active_evs):
        """ Schedule EVs using a round robin based equal sharing scheme.

        Implements abstract method schedule from BaseAlgorithm.

        See class documentation for description of the algorithm.

        Args:
            active_evs (List[EV]): see BaseAlgorithm

        Returns:
            Dict[str, List[float]]: see BaseAlgorithm
        """
        continuous_inc = 1

        ev_queue = deque(self._sort_fn(active_evs, self.interface))
        schedule = {ev.station_id: [0] for ev in active_evs}
        rate_idx_map = {ev.station_id: 0 for ev in active_evs}
        allowable_rates = {}
        for ev in ev_queue:
            evse_continuous, evse_rates = self.interface.allowable_pilot_signals(ev.station_id)
            if evse_continuous:
                evse_rates = np.arange(evse_rates[0], evse_rates[1], continuous_inc)
            allowable_rates[ev.station_id] = evse_rates

        while len(ev_queue) > 0:
            ev = ev_queue.popleft()
            if rate_idx_map[ev.station_id] < len(allowable_rates[ev.station_id]) - 1:
                schedule[ev.station_id][0] = allowable_rates[ev.station_id][rate_idx_map[ev.station_id] + 1]
                if self.interface.is_feasible(schedule):
                    rate_idx_map[ev.station_id] += 1
                    ev_queue.append(ev)
                else:
                    schedule[ev.station_id][0] = allowable_rates[ev.station_id][rate_idx_map[ev.station_id]]
        return schedule


# -------------------- Sorting Functions --------------------------
def first_come_first_served(evs, iface):
    """ Sort EVs by arrival time.

    Args:
        evs (List[EV]): List of EVs to be sorted.
        iface (Interface): Interface object.

    Returns:
        List[EV]: List of EVs sorted by arrival time.
    """
    return sorted(evs, key=lambda x: x.arrival)


def earliest_deadline_first(evs, iface):
    """ Sort EVs by departure time.

    Args:
        evs (List[EV]): List of EVs to be sorted.
        iface (Interface): Interface object.

    Returns:
        List[EV]: List of EVs sorted by departure time.
    """
    return sorted(evs, key=lambda x: x.departure)


def least_laxity_first(evs, iface):
    """ Sort EVs by laxity.

    Laxity is a measure of the charging flexibility of an EV. Here we define laxity as:
        LAX_i(t) = (departure_i - t) - (remaining_demand_i(t) / max_rate_i)

    Args:
        evs (List[EV]): List of EVs to be sorted.
        iface (Interface): Interface object.

    Returns:
        List[EV]: List of EVs sorted by laxity.
    """

    def laxity(ev):
        """ Calculate laxity of the EV.

        Args:
            ev (EV): An EV object.

        Returns:
            float: The laxity of the EV.
        """
        lax = (ev.departure - iface.current_time) - \
              (iface.remaining_amp_periods(ev) / iface.max_pilot_signal(ev.station_id))
        return lax

    return sorted(evs, key=lambda x: laxity(x))



