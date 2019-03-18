from algorithms.base_algorithm import BaseAlgorithm
import cmath
import math
from copy import copy
from collections import deque, defaultdict
import numpy as np


def get_groupings():
    CC_pod = {'CA-322', 'CA-493', 'CA-496', 'CA-320', 'CA-495', 'CA-321', 'CA-323', 'CA-494'}
    AV_pod = {'CA-324', 'CA-325', 'CA-326', 'CA-327', 'CA-489', 'CA-490', 'CA-491', 'CA-492'}
    AB = {'CA-{0}'.format(i) for i in [308, 508, 303, 513, 310, 506, 316, 500, 318, 498]}
    AB |= CC_pod | AV_pod

    BC = {'CA-{0}'.format(i) for i in [304, 512, 305, 511, 313, 503, 311, 505, 317, 499]}
    CA = {'CA-{0}'.format(i) for i in [307, 509, 309, 507, 306, 510, 315, 501, 319, 497, 312, 504, 314, 502]}

    # Temporarily add ADA spaces to BC, since we don't know where they come from
    BC |= {'CA-148', 'CA-149', 'CA-212', 'CA-213'}

    groups = {}
    groups['ALL'] = list(AB | BC | CA)
    groups['AB'] = list(AB)
    groups['BC'] = list(BC)
    groups['CA'] = list(CA)
    groups['AV-Pod'] = list(AV_pod)
    groups['CC-Pod'] = list(CC_pod)
    return groups


def get_phasor(mag, phase):
    phase_angles = {'AB': math.radians(30),
                    'BC': math.radians(-90),
                    'CA': math.radians(150)}
    return cmath.rect(mag, phase_angles[phase])


def nested_dict_values(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from nested_dict_values(v)
        else:
            yield v


class UncontrolledCharging(BaseAlgorithm):
    def schedule(self, active_evs):
        charging_rate = 32
        schedule = {}
        for ev in active_evs:
            schedule[ev.station_id] = [min(charging_rate, ev.remaining_demand)]
        return schedule


class NaiveCharging(BaseAlgorithm):
    def __init__(self, sort_fn, const_scale=1, agg_max=None):
        super().__init__()
        self.groups = get_groupings()
        self.const_scale=const_scale
        self.sort_fn = sort_fn
        self._init_agg_max = agg_max if agg_max is not None else float('inf')

    @property
    def agg_max(self):
        return self._init_agg_max

    def schedule(self, active_evs):
        currents = {'AB': 0, 'BC': 0, 'CA': 0, 'AV-Pod': 0, 'CC-Pod': 0, 'ALL': 0}
        ev_queue = self.sort_fn(active_evs)
        schedule = {ev.station_id: [0] for ev in active_evs}
        for ev in ev_queue:
            charging_rate = min(ev.max_rate, ev.remaining_demand)
            currents, _ = self.add_rate(ev, charging_rate, schedule, currents)
        return schedule

    def add_rate(self, ev, charging_rate, schedule, currents):
        new_currents = self.get_new_currents(ev.station_id, charging_rate, currents)
        if self.is_feasible(new_currents):
            schedule[ev.station_id][0] += charging_rate
            return new_currents, True
        else:
            return currents, False

    def rate_feasible(self, station_id, proposed_rate, currents):
        return self.is_feasible(self.get_new_currents(station_id, proposed_rate, currents))

    def get_new_currents(self, station_id, charging_rate, currents):
        new_currents = copy(currents)
        new_currents['ALL'] += charging_rate
        if station_id in self.groups['AB']:
            new_currents['AB'] += get_phasor(charging_rate, 'AB')
        elif station_id in self.groups['BC']:
            new_currents['BC'] += get_phasor(charging_rate, 'BC')
        elif station_id in self.groups['CA']:
            new_currents['CA'] += get_phasor(charging_rate, 'CA')

        if station_id in self.groups['CC-Pod']:
            new_currents['CC-Pod'] += charging_rate
        if station_id in self.groups['AV-Pod']:
            new_currents['AV-Pod'] += charging_rate
        return new_currents

    def feasibility_check(self, currents):
        primary_line_const = 180 * self.const_scale
        secondary_line_const = 420 * self.const_scale
        pod_const = 80
        n = 4

        feasible = defaultdict(dict)
        secondary = {}
        secondary['a'] = currents['AB'] - currents['CA']
        secondary['b'] = currents['BC'] - currents['AB']
        secondary['c'] = currents['CA'] - currents['BC']

        primary = {}
        primary['a'] = (1 / n) * (secondary['a'] - secondary['c'])
        primary['b'] = (1 / n) * (secondary['b'] - secondary['a'])
        primary['c'] = (1 / n) * (secondary['c'] - secondary['b'])

        feasible['secondary'] = {phase: abs(secondary[phase]) <= secondary_line_const for phase in secondary}
        feasible['primary'] = {phase: abs(primary[phase]) <= primary_line_const for phase in primary}
        feasible['CC-Pod'] = currents['CC-Pod'] <= pod_const
        feasible['AV-Pod'] = currents['AV-Pod'] <= pod_const
        feasible['ALL'] = currents['ALL'] <= self.agg_max
        return feasible

    def groups_feasibility(self, currents):
        feasible = self.feasibility_check(currents)
        primary_feasible = all(feasible['primary'].values())
        groups_feasible = {}
        groups_feasible['ALL'] = feasible['ALL']
        groups_feasible['AB'] = feasible['ALL'] and primary_feasible and feasible['secondary']['a'] and feasible['secondary']['b']
        groups_feasible['BC'] = feasible['ALL'] and primary_feasible and feasible['secondary']['b'] and feasible['secondary']['c']
        groups_feasible['CA'] = feasible['ALL'] and primary_feasible and feasible['secondary']['a'] and feasible['secondary']['c']
        groups_feasible['CC-Pod'] = groups_feasible['AB'] and feasible['CC-Pod']
        groups_feasible['AV-Pod'] = groups_feasible['AB'] and feasible['AV-Pod']
        return groups_feasible

    def is_feasible(self, currents):
        return all(nested_dict_values(self.feasibility_check(currents)))


class NaiveWithFillUp(NaiveCharging):
    def schedule(self, active_evs):
        currents = {'AB': 0, 'BC': 0, 'CA': 0, 'AV-Pod': 0, 'CC-Pod': 0, 'ALL': 0}
        ev_queue = self.sort_fn(active_evs)
        schedule = {ev.station_id: [0] for ev in active_evs}
        for ev in ev_queue:
            charging_rate = self.binary_search(ev.station_id, 0, ev.max_rate, currents, eps=0.01)
            currents, success = self.add_rate(ev, charging_rate, schedule, currents)
        return schedule

    def binary_search(self, station_id, lb, ub, currents, eps=0.01):
        mid = (ub + lb) / 2
        new_currents = self.get_new_currents(station_id, mid, currents)
        if (ub - lb) <= eps:
            return lb
        elif self.is_feasible(new_currents):
            return self.binary_search(station_id, mid, ub, new_currents, eps)
        else:
            return self.binary_search(station_id, lb, mid, new_currents, eps)


class RoundRobin(NaiveCharging):
    def schedule(self, active_evs):
        currents = {'AB': 0, 'BC': 0, 'CA': 0, 'AV-Pod': 0, 'CC-Pod': 0, 'ALL': 0}
        ev_queue = deque(self.sort_fn(active_evs))
        schedule = {ev.station_id: [0] for ev in active_evs}

        inc = 1
        while len(ev_queue) > 0:
            ev = ev_queue.popleft()
            if schedule[ev.station_id][0] < min(ev.remaining_demand, ev.max_rate):
                charging_rate = min(inc, ev.remaining_demand - schedule[ev.station_id][0])
                currents, success = self.add_rate(ev, charging_rate, schedule, currents)
                if success:
                    ev_queue.append(ev)
        return schedule


class IndividualGreedyCostMin(NaiveWithFillUp):
    def schedule(self, active_evs):
        if len(active_evs) < 1:
            return {}
        T = int(math.ceil(np.max([ev.departure for ev in active_evs]))) + 1
        t = self.interface.get_current_time()

        currents = [{'AB': 0, 'BC': 0, 'CA': 0, 'AV-Pod': 0, 'CC-Pod': 0, 'ALL': 0}]*(T-t)
        ev_queue = self.sort_fn(active_evs)
        schedule = {ev.station_id: [0]*(T-t) for ev in active_evs}
        for ev in ev_queue:
            prices = self.interface.get_prices(t, ev.departure-t)
            pref = np.argsort(prices, kind='mergesort')
            e_del = 0
            for i in pref:
                if e_del >= ev.remaining_demand:
                    break
                charging_rate = self.binary_search(ev.station_id, 0, ev.max_rate, currents[i], eps=0.01)
                currents[i] = self.get_new_currents(ev.station_id, charging_rate, currents[i])
                schedule[ev.station_id][i] = charging_rate
                e_del += charging_rate
        return schedule


# class SmoothedLLF(NaiveCharging):
#     def schedule(self, active_evs):
#         currents = {'AB': 0, 'BC': 0, 'CA': 0, 'AV-Pod': 0, 'CC-Pod': 0, 'ALL': 0}
#         ev_queue = self.sort_fn(active_evs)
#         schedule = {ev.station_id: [0] for ev in active_evs}
#         for ev in ev_queue:
#             charging_rate = self.binary_search(ev.station_id, 0, ev.max_rate, currents, eps=0.01)
#             currents, success = self.add_rate(ev, charging_rate, schedule, currents)
#         return schedule
#
#     def find_currents(self, active_evs, target_level, currents):
#         rates = {}
#         new_currents = deepcopy(currents)
#         for id, ev in active_evs.items():
#             rates[id] = max(0, min((ev.max_rate*(target_level - self.laxity(ev) + 1), ev.max_rate, ev.remaining_demand)))
#             new_currents = self.get_new_currents(id, rates[id], new_currents)
#         return new_currents
#
#     @staticmethod
#     def laxity(ev):
#         return (ev.departure - ev.arrival) - (ev.remaining_demand / ev.max_rate)
#
#     def bisection_level_search(self, active_evs, lb, ub, currents, eps=0.01):
#         if (ub - lb) <= eps:
#
#             return lb
#         mid = (ub - lb) / 2
#         currents = self.find_currents(active_evs, mid, currents)
#         if self.is_feasible(currents):
#             self.bisection_level_search(active_evs, mid, ub, currents, eps)
#         else:
#             self.bisection_level_search(active_evs, lb, mid, currents, eps)



def fcfs(evs):
    return sorted(evs, key=lambda x: x.arrival)


def edf(evs):
    return sorted(evs, key=lambda x: x.departure)


def llf(evs):
    def laxity(ev):
        return (ev.departure - ev.arrival) - (ev.remaining_demand / ev.max_rate)
    return sorted(evs, key=lambda x: laxity(x))

