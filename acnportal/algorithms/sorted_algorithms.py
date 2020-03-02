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

    def __init__(self, sort_fn, minimum_charge=False, rampdown=None, peak_limit=float('inf')):
        super().__init__(rampdown)
        self._sort_fn = sort_fn
        self.max_recompute = 1  # Call algorithm each period since it only returns a rate for the next period.
        self.minimum_charge = minimum_charge
        self.peak_limit = peak_limit

    def _find_minimum_charge(self, ev_queue):
        schedule = {ev.station_id: [0] for ev in ev_queue}
        removed = set()
        for ev in ev_queue:
            continuous, allowable_rates = self.interface.allowable_pilot_signals(ev.station_id)
            schedule[ev.station_id][0] = allowable_rates[0] if continuous else allowable_rates[1]
            if schedule[ev.station_id][0] > self.interface.remaining_amp_periods(ev) or\
                    not self.is_feasible(schedule):
                schedule[ev.station_id][0] = 0
                removed.add(ev.station_id)
        return schedule, removed
    
    def is_feasible(self, load_currents, linear=False, violation_tolerance=None, relative_tolerance=None):
        """ Return if a set of current magnitudes for each load are feasible.

        Wraps Network's is_feasible method.

        For a given constraint, the larger of the violation_tolerance
        and relative_tolerance is used to evaluate feasibility.

        Args:
            load_currents (Dict[str, List[number]]): Dictionary mapping load_ids to schedules of charging rates.
            linear (bool): If True, linearize all constraints to a more conservative but easier to compute constraint by
                ignoring the phase angle and taking the absolute value of all load coefficients. Default False.
            violation_tolerance (float): Absolute amount by which
                schedule may violate network constraints. Default
                None, in which case the network's violation_tolerance
                attribute is used.
            relative_tolerance (float): Relative amount by which
                schedule may violate network constraints. Default
                None, in which case the network's relative_tolerance
                attribute is used.

        Returns:
            bool: If load_currents is feasible at time t according to this set of constraints.
        """
        network_constraints = self.interface.is_feasible(load_currents, linear, violation_tolerance, relative_tolerance)
        peak_limit = np.sum(np.array(x) for x in load_currents.values()) <= self.peak_limit
        return network_constraints and peak_limit  

    def schedule(self, active_evs, init_schedule=None):
        """ Schedule EVs by first sorting them by sort_fn, then allocating them their maximum feasible rate.

        Implements abstract method schedule from BaseAlgorithm.

        See class documentation for description of the algorithm.

        Args:
            active_evs (List[EV]): see BaseAlgorithm

        Returns:
            Dict[str, List[float]]: see BaseAlgorithm
        """
        if init_schedule is not None:
            schedule = init_schedule
        if self.minimum_charge:
            active_evs = self.remove_active_evs_less_than_deadband(active_evs)
            ev_queue = self._sort_fn(active_evs, self.interface)
            schedule, _ = self._find_minimum_charge(ev_queue)
        else:
            ev_queue = self._sort_fn(active_evs, self.interface)
            schedule = {ev.station_id: [0] for ev in ev_queue}

        if self.rampdown is not None:
            rampdown_max = self.rampdown.get_maximum_rates(active_evs)
            # Prevent rampdown from pushing the upper bound below the minimum rate.
            if self.minimum_charge:
                for ev in active_evs:
                    rampdown_max[ev.session_id] = max(rampdown_max[ev.session_id], schedule[ev.station_id][0])

        for ev in ev_queue:
            continuous, allowable_rates = self.interface.allowable_pilot_signals(ev.station_id)
            if continuous:
                max_rate_limit = min(allowable_rates[-1], self.interface.remaining_amp_periods(ev))
                if self.rampdown is not None:
                    max_rate_limit = min(rampdown_max[ev.session_id], max_rate_limit)
                charging_rate = self.max_feasible_rate(ev.station_id, max_rate_limit, schedule,
                                                       lb=schedule[ev.station_id][0], eps=0.01)
            else:
                max_rate_limit = self.interface.remaining_amp_periods(ev)
                if self.rampdown is not None:
                    max_rate_limit = min(rampdown_max[ev.session_id], max_rate_limit)
                allowable_rates = [x for x in allowable_rates if schedule[ev.station_id][0] <= x <= max_rate_limit]
                charging_rate = self.discrete_max_feasible_rate(ev.station_id, allowable_rates, schedule)
            schedule[ev.station_id][0] = charging_rate
        return schedule

    def max_feasible_rate(self, station_id, ub, schedule, lb=0, time=0, eps=0.01):
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
            elif self.is_feasible(new_schedule):
                return bisection(_station_id, mid, _ub, new_schedule)
            else:
                return bisection(_station_id, _lb, mid, new_schedule)

        if not self.is_feasible(schedule):
            raise ValueError('The initial schedule is not feasible.')
        return bisection(station_id, lb, ub, schedule)

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
        if not self.is_feasible(schedule):
            raise ValueError('The initial schedule is not feasible.')
        new_schedule = copy(schedule)
        feasible_idx = len(allowable_rates) - 1
        new_schedule[station_id][time] = allowable_rates[feasible_idx]
        while not self.is_feasible(new_schedule):
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

    def schedule(self, active_evs, init_schedule=None):
        """ Schedule EVs using a round robin based equal sharing scheme.

        Implements abstract method schedule from BaseAlgorithm.

        See class documentation for description of the algorithm.

        Args:
            active_evs (List[EV]): see BaseAlgorithm

        Returns:
            Dict[str, List[float]]: see BaseAlgorithm
        """
        continuous_inc = 1
        if self.minimum_charge:
            active_evs = self.remove_active_evs_less_than_deadband(active_evs)

        ev_queue = deque(self._sort_fn(active_evs, self.interface))            
        allowable_rates = {}
        if self.rampdown is not None:
            rd_maxes = self.rampdown.get_maximum_rates(active_evs)
        for ev in ev_queue:
            evse_continuous, evse_rates = self.interface.allowable_pilot_signals(ev.station_id)
            if evse_continuous:
                evse_rates = np.arange(evse_rates[0], evse_rates[1], continuous_inc)
            max_rate_limit = self.interface.remaining_amp_periods(ev)
            if self.rampdown is not None:
                max_rate_limit = max(rd_maxes[ev.session_id], max_rate_limit)
                allowable_rates[ev.station_id] = [x for x in evse_rates if x <= max_rate_limit]
            else:
                allowable_rates[ev.station_id] = evse_rates
            
        if init_schedule is None:
            schedule = {ev.station_id: [0] for ev in active_evs}
            rate_idx_map = {ev.station_id: 0 for ev in active_evs}
        else:
            schedule = init_schedule
            rate_idx_map = {station_id: allowable_rates[station_id].index(sch[0]) for station_id, sch in init_schedule.items()}

        while len(ev_queue) > 0:
            ev = ev_queue.popleft()
            if rate_idx_map[ev.station_id] < len(allowable_rates[ev.station_id]) - 1:
                schedule[ev.station_id][0] = allowable_rates[ev.station_id][rate_idx_map[ev.station_id] + 1]
                if self.is_feasible(schedule):
                    rate_idx_map[ev.station_id] += 1
                    ev_queue.append(ev)
                else:
                    schedule[ev.station_id][0] = allowable_rates[ev.station_id][rate_idx_map[ev.station_id]]
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
        LAX_i(t) = (departure_i - t) - ( _i(t) / max_rate_i)

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
        lax = (ev.departure - iface.current_time) - \
              (iface.remaining_amp_periods(ev) / iface.max_pilot_signal(ev.station_id))
        return lax

    return sorted(evs, key=lambda x: laxity(x))


def largest_remaining_processing_time(evs, iface):
    """ Sort EVs in decreasing order by the time taken to finish charging them at the EVSE's maximum rate.

    Args:
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
        rpt = (iface.remaining_amp_periods(ev) / iface.max_pilot_signal(ev.station_id))
        return rpt

    return sorted(evs, key=lambda x: remaining_processing_time(x), reverse=True)


