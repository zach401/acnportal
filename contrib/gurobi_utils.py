from collections import defaultdict
from gurobipy import *
import numpy as np
from collections import defaultdict
AFFINE = 'affine'
PHASE_AWARE = 'phase_aware'


class InfeasibilityException(Exception):
    pass


# Helpful functions
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


def _sq(ex):
    return ex * ex


class OptimizationScheduler:
    def __init__(self, const_type=PHASE_AWARE, const_scale=1, energy_equality=False, verbose=False):
        self.primary_line_const = 180 * const_scale
        self.secondary_line_const = 420 * const_scale
        self.pod_const = 80

        self.groups = get_groupings()
        self.const_type = const_type
        self.energy_equality = energy_equality

        # Create a new model
        self.m = Model('scheduler')
        self.m.setParam('OutputFlag', verbose)

        self.currents = {'ALL': defaultdict(int), 'AB': defaultdict(int), 'BC': defaultdict(int), 'CA': defaultdict(int),
                         'AV-Pod': defaultdict(int), 'CC-Pod': defaultdict(int)}
        self.active_times = set()
        self.schedules = {}
        self.obj = None

    def add_ev(self, ev, ramp_down=float('inf')):
        if ev.departure < 1:
            return

        # Define charging rates variable and include its upper and lower bounds
        max_rate = min(ev.max_rate, ramp_down)
        rates = self.m.addVars(range(max(ev.arrival, 0), ev.departure), lb=0, ub=max_rate)
        for t in range(max(ev.arrival, 0), ev.departure):
            self.active_times.add(t)
        self.schedules[ev.session_id] = rates

        # Define energy constraint
        if self.energy_equality:
            self.m.addConstr(rates.sum() == ev.remaining_demand)
        else:
            self.m.addConstr(rates.sum() <= ev.remaining_demand)

        # Update infrastructure constraints
        groups = self._get_groups(ev)
        self._add_to_currents(rates, groups)

    def add_evs(self, evs, ramp_down=None):
        for ev in evs:
            if ramp_down is not None and ev.station_id in ramp_down:
                self.add_ev(ev, ramp_down=ramp_down[ev.station_id])
            else:
                self.add_ev(ev)

    def _get_groups(self, ev):
        member_groups = []
        for group in self.groups:
            if ev.station_id in self.groups[group]:
                member_groups.append(group)
        return member_groups

    def _add_to_currents(self, r, groups):
        for g in groups:
            for t in r:
                self.currents[g][t] += r[t]

    def _active_times(self):
        # active_t = {}
        # for c in self.currents.values():
        #     active_t |= c.keys()
        # return list(active_t)
        return self.active_times
        
    def _add_affine_infrastructure_constraints(self, turns_ratio=4):
        active_t = self._active_times()

        for t in active_t:
            # Secondary Side
            Ia_sec = self.currents['AB'][t] + self.currents['CA'][t]
            Ib_sec = self.currents['BC'][t] + self.currents['AB'][t]
            Ic_sec = self.currents['CA'][t] + self.currents['BC'][t]

            self.m.addConstr(Ia_sec <= self.secondary_line_const)
            self.m.addConstr(Ib_sec <= self.secondary_line_const)
            self.m.addConstr(Ic_sec <= self.secondary_line_const)

            # Primary Side
            primary_limit_tr = self.primary_line_const * turns_ratio

            Ia_prime = self.currents['AB'][t] + self.currents['BC'][t] + 2 * self.currents['CA'][t]
            Ib_prime = 2 * self.currents['AB'][t] + self.currents['BC'][t] + self.currents['CA'][t]
            Ic_prime = self.currents['AB'][t] + 2 * self.currents['BC'][t] + self.currents['CA'][t]

            self.m.addConstr(Ia_prime <= primary_limit_tr)
            self.m.addConstr(Ib_prime <= primary_limit_tr)
            self.m.addConstr(Ic_prime <= primary_limit_tr)

    def _add_soc_infrastructure_constraints(self, turns_ratio=4):
        # Define phase conversions
        phaseAB = {'re': np.cos(np.deg2rad(30)), 'im': np.sin(np.deg2rad(30))}
        phaseBC = {'re': np.cos(np.deg2rad(-90)), 'im': np.sin(np.deg2rad(-90))}
        phaseCA = {'re': np.cos(np.deg2rad(150)), 'im': np.sin(np.deg2rad(150))}

        active_t = self._active_times()

        for t in active_t:
            # Secondary Side
            Ia_sec_sq = _sq(self.currents['AB'][t]) + _sq(self.currents['CA'][t]) + (self.currents['AB'][t] * self.currents['CA'][t])
            Ib_sec_sq = _sq(self.currents['BC'][t]) + _sq(self.currents['AB'][t]) + (self.currents['BC'][t] * self.currents['AB'][t])
            Ic_sec_sq = _sq(self.currents['CA'][t]) + _sq(self.currents['BC'][t]) + (self.currents['CA'][t] * self.currents['BC'][t])

            secondary_limit_sq = _sq(self.secondary_line_const)
            self.m.addConstr(Ia_sec_sq <= secondary_limit_sq)
            self.m.addConstr(Ib_sec_sq <= secondary_limit_sq)
            self.m.addConstr(Ic_sec_sq <= secondary_limit_sq)

            # Primary Side
            Ia_prime = {part: self.currents['AB'][t] * phaseAB[part] + self.currents['BC'][t] * phaseBC[part] - 2 * self.currents['CA'][t] * phaseCA[part] for part in ['re', 'im']}
            Ib_prime = {part: -2 * self.currents['AB'][t] * phaseAB[part] + self.currents['BC'][t] * phaseBC[part] + self.currents['CA'][t] * phaseCA[part] for part in ['re', 'im']}
            Ic_prime = {part: self.currents['AB'][t] * phaseAB[part] - 2 * self.currents['BC'][t] * phaseBC[part] + self.currents['CA'][t] * phaseCA[part] for part in ['re', 'im']}

            # Add Primary Line Capacity Constraints
            primary_limit_tr_sq = _sq(self.primary_line_const * turns_ratio)
            self.m.addConstr(_sq(Ia_prime['re']) + _sq(Ia_prime['im']) <= primary_limit_tr_sq)
            self.m.addConstr(_sq(Ib_prime['re']) + _sq(Ib_prime['im']) <= primary_limit_tr_sq)
            self.m.addConstr(_sq(Ic_prime['re']) + _sq(Ic_prime['im']) <= primary_limit_tr_sq)
    
    def _add_pod_constraints(self):
        active_t = self._active_times()
        self.m.addConstrs((self.currents['AV-Pod'][t] <= self.pod_const for t in active_t), 'AV-pod-cap')
        self.m.addConstrs((self.currents['CC-Pod'][t] <= self.pod_const for t in active_t), 'CC-pod-cap')

    def compile(self):
        if len(self._active_times()) == 0:
            return
        self.m.setObjective(self.obj, GRB.MAXIMIZE)
        if self.const_type == PHASE_AWARE:
            self._add_soc_infrastructure_constraints()
        elif self.const_type == AFFINE:
            self._add_affine_infrastructure_constraints()
        else:
            raise ValueError('Invalid const_type, should be {0} or {1}'.format(PHASE_AWARE, AFFINE))
        self._add_pod_constraints()

    def solve(self):
        if len(self._active_times()) == 0:
            return None, {}, None
        print('Solving...')
        # self.m.setParam('BarHomogeneous', True)
        self.m.optimize()

        # if self.m.status == GRB.Status.NUMERIC:
        #     self.m.setParam('OutputFlag', True)
        #     self.m.setParam('BarHomogeneous', True)
        #     self.m.setParam('BarConvTol', 1e-2)
        #     self.m.optimize()

        if self.m.status in (GRB.Status.OPTIMAL, GRB.Status.SUBOPTIMAL):
            pilot = {sess_id: dict(self.m.getAttr('x', rates)) for sess_id, rates in self.schedules.items()}
            return self.obj.getValue(), pilot, self.m.runtime
        else:
            raise InfeasibilityException

    # Objectives
    def max_profit_obj(self, unit_revenue, unit_cost, demand_charge, prev_max=0):
        active_t = self._active_times()
        revenue = unit_revenue * quicksum(self.currents['ALL'][t] for t in active_t)
        energy_price = quicksum((self.currents['ALL'][t] * unit_cost[t] for t in active_t))
        peak = self.m.addVar(name='peak')
        self.m.addConstrs((peak >= self.currents['ALL'][t] for t in active_t), 'find_peak')
        self.m.addConstr(peak >= prev_max, 'previous_peak')
        demand_price = demand_charge * peak
        return revenue - energy_price - demand_price

    def max_profit_w_solar_obj(self, unit_revenue, unit_cost, demand_charge, solar_gen, prev_max=0):
        active_t = self._active_times()
        revenue = unit_revenue * quicksum(self.currents['ALL'][t] for t in active_t)

        # Energy Cost
        solar_usage = self.m.addVars(active_t, lb=0)
        self.m.addConstrs((solar_usage[t] <= solar_gen[t] for t in active_t))
        self.m.addConstrs((solar_usage[t] <= self.currents['ALL'][t] for t in active_t))
        energy_cost = quicksum(((self.currents['ALL'][t] - solar_usage[t]) * unit_cost[t] for t in active_t))

        # Demand Cost
        peak = self.m.addVar(name='peak')
        self.m.addConstrs((peak >= (self.currents['ALL'][t] - solar_usage[t]) for t in active_t), 'find_peak')
        self.m.addConstr(peak >= prev_max, 'previous_peak')
        demand_cost = demand_charge * peak
        return revenue - energy_cost - demand_cost

    def quick_charge_obj(self):
        active_t = self._active_times()
        if len(active_t) > 0:
            T = max(active_t)
            quick_charge = quicksum((self.currents['ALL'][t] * (T + 1 - t) for t in active_t))
            return quick_charge
        else:
            return 0

    def tracking_obj(self, signal, alpha=0):
        active_t = self._active_times()
        if alpha > 0:
            signal_tracking = -quicksum((((1-alpha)**t)*_sq(self.currents['ALL'][t] - signal[t]) for t in active_t))
        else:
            signal_tracking = -quicksum((_sq(self.currents['ALL'][t] - signal[t]) for t in active_t))
        return signal_tracking

    def force_concave(self):
        active_t = self._active_times()
        force_concave = quicksum((-_sq(self.currents['ALL'][t]) for t in active_t))
        return force_concave

    def min_capacity_obj(self):
        cap = self.m.addVar(lb=0, name='cap')
        self.primary_line_const = cap / (3 * 277)
        self.secondary_line_const = cap / (3 * 120)
        return -1 * cap

    def smooth_obj(self, prev_rates=None):
        sorted_times = {key: sorted(rates.keys()) for key, rates in self.schedules.items()}
        # Add smoothing for initial rate
        smooth = 0
        if prev_rates is not None:
            smooth += -quicksum(
                _sq(rates[sorted_times[key][0]] - prev_rates[key]) for key, rates in self.schedules.items() if key in prev_rates)
        # Add smoothing for all subsequent rates
        for key, rates in self.schedules.items():
            smooth += -quicksum(_sq(rates[t] - rates[t - 1]) for t in sorted_times[key][1:])
        return smooth


class OptimizationPostProcessor(OptimizationScheduler):
    def __init__(self, min_rate, cutoff, e_del, const_type=PHASE_AWARE, const_scale=1, verbose=False):
        super().__init__(const_type, const_scale, verbose)
        self.cutoff = cutoff
        self.min_rate = min_rate
        self.e_del = e_del

    def add_ev(self, ev, ramp_down=None):
        remaining_demand = ev.remaining_demand - self.e_del[ev.session_id]
        if remaining_demand > self.cutoff:
            # Define charging rates variable and include its upper and lower bounds
            rates = self.m.addVars([0], lb=self.min_rate, ub=min(ev.max_rate, remaining_demand))
            self.active_times.add(0)
            self.schedules[ev.session_id] = rates

            # Update infrastructure constraints
            groups = self._get_groups(ev)
            self._add_to_currents(rates, groups)

    def post_processing_obj(self, targets, alpha=1e-2):
        energy_target = sum(targets.values())
        obj = -_sq(energy_target - self.currents['ALL'][0])
        obj += alpha*quicksum((-_sq(targets[sess_id] - self.schedules[sess_id][0]) for sess_id in self.schedules))
        return obj
