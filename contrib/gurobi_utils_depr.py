from collections import defaultdict
from gurobipy import *
import numpy as np

AFFINE = 'affine'
PHASE_AWARE = 'phase_aware'


class InfeasibilityException(Exception):
    pass


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

# Helpful functions
def ev_active(ev, t):
    return ev.arrival <= t <= ev.departure


def _sq(ex):
    return ex * ex


# Constraints
def _add_rate_constraints(m, rates, EVs_dict, T):
    # Upper Bounds
    m.addConstrs((rates[ev_id[0], ev_id[1], t] <= EVs_dict[ev_id].max_rate
                  for ev_id in EVs_dict for t in range(T) if ev_active(EVs_dict[ev_id], t)))

    # Lower Bounds
    m.addConstrs((rates[ev_id[0], ev_id[1], t] >= 0
                  for ev_id in EVs_dict for t in range(T) if ev_active(EVs_dict[ev_id], t)))

    # Enforce 0 rate for non-active EVs
    m.addConstrs((rates[ev_id[0], ev_id[1], t] == 0 for ev_id in EVs_dict
                  for t in range(T) if not ev_active(EVs_dict[ev_id], t)))


def _add_energy_equality_constraints(m, rates, EVs_dict):
    m.addConstrs((rates.sum(ev_id[0], ev_id[1], '*') == EVs_dict[ev_id].remaining_demand for ev_id in EVs_dict))


def _add_energy_inequality_constraints(m, rates, EVs_dict):
    m.addConstrs((rates.sum(ev_id[0], ev_id[1], '*') <= EVs_dict[ev_id].remaining_demand for ev_id in EVs_dict))


def _add_affine_cap_constraints(m, rates, groups, T, primary_limit, secondary_limit, turns_ratio=4):
    # Secondary Side
    Iab = [rates.sum(groups['AB'], '*', t) for t in range(T)]
    Ibc = [rates.sum(groups['BC'], '*', t) for t in range(T)]
    Ica = [rates.sum(groups['CA'], '*', t) for t in range(T)]

    Ia_sec = [Iab[t] + Ica[t] for t in range(T)]
    Ib_sec = [Ibc[t] + Iab[t] for t in range(T)]
    Ic_sec = [Ica[t] + Ibc[t] for t in range(T)]

    m.addConstrs(Ia_sec[t] <= secondary_limit for t in range(T))
    m.addConstrs(Ib_sec[t] <= secondary_limit for t in range(T))
    m.addConstrs(Ic_sec[t] <= secondary_limit for t in range(T))

    # Primary Side
    primary_limit_tr = primary_limit * turns_ratio

    Ia_prime = [Iab[t] + Ibc[t] + 2 * Ica[t] for t in range(T)]
    Ib_prime = [2 * Iab[t] + Ibc[t] + Ica[t] for t in range(T)]
    Ic_prime = [Iab[t] + 2 * Ibc[t] + Ica[t] for t in range(T)]

    m.addConstrs(Ia_prime[t] <= primary_limit_tr for t in range(T))
    m.addConstrs(Ib_prime[t] <= primary_limit_tr for t in range(T))
    m.addConstrs(Ic_prime[t] <= primary_limit_tr for t in range(T))


def _add_three_phase_cap_constraints(m, rates, groups, T, primary_limit, secondary_limit, turns_ratio=4):
    # Secondary Side
    Iab = [rates.sum(groups['AB'], '*', t) for t in range(T)]
    Ibc = [rates.sum(groups['BC'], '*', t) for t in range(T)]
    Ica = [rates.sum(groups['CA'], '*', t) for t in range(T)]
    Ia_sec_sq = [_sq(Iab[t]) + _sq(Ica[t]) + (Iab[t] * Ica[t]) for t in range(T)]
    Ib_sec_sq = [_sq(Ibc[t]) + _sq(Iab[t]) + (Ibc[t] * Iab[t]) for t in range(T)]
    Ic_sec_sq = [_sq(Ica[t]) + _sq(Ibc[t]) + (Ica[t] * Ibc[t]) for t in range(T)]

    secondary_limit_sq = _sq(secondary_limit)
    m.addConstrs(Ia_sec_sq[t] <= secondary_limit_sq for t in range(T))
    m.addConstrs(Ib_sec_sq[t] <= secondary_limit_sq for t in range(T))
    m.addConstrs(Ic_sec_sq[t] <= secondary_limit_sq for t in range(T))

    # Primary Side
    # Define phase conversions
    phaseAB = {'re': np.cos(np.deg2rad(30)), 'im': np.sin(np.deg2rad(30))}
    phaseBC = {'re': np.cos(np.deg2rad(-90)), 'im': np.sin(np.deg2rad(-90))}
    phaseCA = {'re': np.cos(np.deg2rad(150)), 'im': np.sin(np.deg2rad(150))}

    Ia_prime = {part: [Iab[t] * phaseAB[part] + Ibc[t] * phaseBC[part] - 2 * Ica[t] * phaseCA[part] for t in range(T)]
                for part in ['re', 'im']}
    Ib_prime = {part: [-2 * Iab[t] * phaseAB[part] + Ibc[t] * phaseBC[part] + Ica[t] * phaseCA[part] for t in range(T)]
                for part in ['re', 'im']}
    Ic_prime = {part: [Iab[t] * phaseAB[part] - 2 * Ibc[t] * phaseBC[part] + Ica[t] * phaseCA[part] for t in range(T)]
                for part in ['re', 'im']}

    # Add Primary Line Capacity Constraints
    primary_limit_tr_sq = _sq(primary_limit * turns_ratio)
    m.addConstrs(_sq(Ia_prime['re'][t]) + _sq(Ia_prime['im'][t]) <= primary_limit_tr_sq for t in range(T))
    m.addConstrs(_sq(Ib_prime['re'][t]) + _sq(Ib_prime['im'][t]) <= primary_limit_tr_sq for t in range(T))
    m.addConstrs(_sq(Ic_prime['re'][t]) + _sq(Ic_prime['im'][t]) <= primary_limit_tr_sq for t in range(T))


def gurobi_schedule(obj_function, EVs, cap_const=AFFINE, const_scale=1.0, energy_equality=True):
    primary_line_const = 180 * const_scale
    secondary_line_const = 420 * const_scale
    pod_const = 80  # *const_scale

    groups = get_groupings()

    print('Building Problem...')

    EVs_dict = {(ev.station_id, ev.session_id): ev for ev in EVs}

    T = int(math.ceil(np.max([ev.departure for ev in EVs]))) + 1
    N = len(EVs)
    print(N, T)

    # Create a new model
    m = Model('scheduler')
    m.setParam('OutputFlag', False)

    # Create variables
    rates = m.addVars(EVs_dict.keys(), range(T), name='rates')

    # Set objective
    obj = obj_function(rates, T)
    m.setObjective(obj, GRB.MAXIMIZE)

    # Rate Constraints
    _add_rate_constraints(m, rates, EVs_dict, T)

    # Energy Constraints
    if energy_equality:
        _add_energy_equality_constraints(m, rates, EVs_dict)
    else:
        _add_energy_inequality_constraints(m, rates, EVs_dict)

    # Capacity Constraints
    if cap_const == AFFINE:
        _add_affine_cap_constraints(m, rates, groups, T, primary_line_const, secondary_line_const, turns_ratio=4)
    elif cap_const == PHASE_AWARE:
        _add_three_phase_cap_constraints(m, rates, groups, T, primary_line_const, secondary_line_const, turns_ratio=4)

    # Pod Capacity Constraints
    m.addConstrs((rates.sum(groups['AV-Pod'], '*', t) <= pod_const for t in range(T)), 'AV-pod-cap')
    m.addConstrs((rates.sum(groups['CC-Pod'], '*', t) <= pod_const for t in range(T)), 'CC-pod-cap')

    # Solve
    print('Solving...')
    m.optimize()

    if m.status in (GRB.Status.OPTIMAL, GRB.Status.SUBOPTIMAL):
        rates_mat = np.zeros((N, T))
        solution = m.getAttr('x', rates)
        for i, sess in enumerate(EVs_dict.keys()):
            for t in range(T):
                rates_mat[i, t] = solution[sess[0], sess[1], t]
        pilot = {EVs[i].session_id: rates_mat[i, :] for i in range(len(EVs))}
        return obj.getValue(), pilot, m.runtime
    else:
        raise InfeasibilityException


def gurobi_schedule_w_peak(obj_function, prev_max, EVs, cap_const=AFFINE, const_scale=1.0, energy_equality=True):
    primary_line_const = 180 * const_scale
    secondary_line_const = 420 * const_scale
    pod_const = 80

    groups = get_groupings()

    print('Building Problem...')

    EVs_dict = {(ev.station_id, ev.session_id): ev for ev in EVs}

    T = int(math.ceil(np.max([ev.departure for ev in EVs]))) + 1
    N = len(EVs)
    print(N, T)

    # Create a new model
    m = Model('scheduler')
    # m.setParam('OutputFlag', False)

    # Create variables
    rates = m.addVars(EVs_dict.keys(), range(T), name='rates')
    peak = m.addVar(name='peak')


    # Set objective
    obj = obj_function(rates, peak, T)
    m.setObjective(obj, GRB.MAXIMIZE)

    # Find peak
    m.addConstrs((peak >= rates.sum('*', '*', t) for t in range(T)), 'find_peak')
    m.addConstr(peak >= prev_max, 'previous_peak')

    # Rate Constraints
    _add_rate_constraints(m, rates, EVs_dict, T)

    # Energy Constraints
    if energy_equality:
        _add_energy_equality_constraints(m, rates, EVs_dict)
    else:
        _add_energy_inequality_constraints(m, rates, EVs_dict)

    # Capacity Constraints
    if cap_const == AFFINE:
        _add_affine_cap_constraints(m, rates, groups, T, primary_line_const, secondary_line_const, turns_ratio=4)
    elif cap_const == PHASE_AWARE:
        _add_three_phase_cap_constraints(m, rates, groups, T, primary_line_const, secondary_line_const, turns_ratio=4)

    # Pod Capacity Constraints
    m.addConstrs((rates.sum(groups['AV-Pod'], '*', t) <= pod_const for t in range(T)), 'AV-pod-cap')
    m.addConstrs((rates.sum(groups['CC-Pod'], '*', t) <= pod_const for t in range(T)), 'CC-pod-cap')

    # Solve
    print('Solving...')
    m.optimize()

    if m.status in (GRB.Status.OPTIMAL, GRB.Status.SUBOPTIMAL):
        rates_mat = np.zeros((N, T))
        solution = m.getAttr('x', rates)
        for i, sess in enumerate(EVs_dict.keys()):
            for t in range(T):
                rates_mat[i, t] = solution[sess[0], sess[1], t]
        pilot = {EVs[i].session_id: rates_mat[i, :] for i in range(len(EVs))}
        return obj.getValue(), pilot, m.runtime
    else:
        raise InfeasibilityException


def minimize_transformer(EVs, cap_const=PHASE_AWARE):
    pod_const = 80

    groups = get_groupings()

    print('Building Problem...')

    EVs_dict = {(ev.station_id, ev.session_id): ev for ev in EVs}

    T = int(math.ceil(np.max([ev.departure for ev in EVs]))) + 1
    N = len(EVs)
    print(N, T)

    # Create a new model
    m = Model('scheduler')
    m.setParam('OutputFlag', False)

    # Create variables
    rates = m.addVars(EVs_dict.keys(), range(T), name='rates')
    cap = m.addVar(lb=0, name='cap')

    # Set objective
    obj = 1*cap
    m.setObjective(obj, GRB.MINIMIZE)

    # Rate Constraints
    _add_rate_constraints(m, rates, EVs_dict, T)

    # Energy Constraints
    _add_energy_equality_constraints(m, rates, EVs_dict)

    # Capacity Constraints
    primary_line_const = cap / (3*277)
    secondary_line_const = cap / (3 * 120)
    if cap_const == AFFINE:
        _add_affine_cap_constraints(m, rates, groups, T, primary_line_const, secondary_line_const, turns_ratio=4)
    elif cap_const == PHASE_AWARE:
        _add_three_phase_cap_constraints(m, rates, groups, T, primary_line_const, secondary_line_const, turns_ratio=4)

    # Pod Capacity Constraints
    m.addConstrs((rates.sum(groups['AV-Pod'], '*', t) <= pod_const for t in range(T)), 'AV-pod-cap')
    m.addConstrs((rates.sum(groups['CC-Pod'], '*', t) <= pod_const for t in range(T)), 'CC-pod-cap')

    # Solve
    print('Solving...')
    m.optimize()

    schedule = defaultdict(int)
    if m.status in (GRB.Status.OPTIMAL, GRB.Status.SUBOPTIMAL):
        rates_mat = np.zeros((N, T))
        solution = m.getAttr('x', rates)
        for i, sess in enumerate(EVs_dict.keys()):
            for t in range(T):
                rates_mat[i, t] = solution[sess[0], sess[1], t]
        for i in range(len(EVs)):
            schedule[EVs[i].station_id] += rates_mat[i, :]
        return obj.getValue(), schedule
    else:
        raise InfeasibilityException


# Objective Functions
def greedy_obj(rates, T):
    return quicksum((rates.sum('*', '*', t) * (T-t+1) for t in range(T)))


def max_revenue_obj(unit_revenue, unit_cost, demand_charge):
    def _max_revenue_obj(rates, peak, T):
        revenue = unit_revenue * rates.sum('*', '*', '*')
        energy_price = quicksum((rates.sum('*', '*', t) * unit_cost[t] for t in range(T)))
        demand_price = demand_charge * peak
        obj = revenue - energy_price - demand_price
        return obj
    return _max_revenue_obj


def min_cost_obj(unit_cost, demand_charge):
    def _min_cost_obj(rates, peak, T):
        energy_price = quicksum((rates.sum('*', '*', t) * unit_cost[t] for t in range(T)))
        demand_price = demand_charge * peak
        obj = - (energy_price + demand_price)
        return obj
    return _min_cost_obj


# Regularization
def smooth_obj(obj, rates, prev_rates, active_evs, T, eta=1e-4):
    # Add smoothing for initial rate
    obj += eta * quicksum(_sq(rates[s[0], s[1], 0] - prev_rates[i])
                          for i, s in enumerate(active_evs.keys()))

    # Add smoothing for all subsequent rates
    obj += eta * quicksum(((rates[s[0], s[1], t] - rates[s[0], s[1], t - 1]) *
                           (rates[s[0], s[1], t] - rates[s[0], s[1], t - 1])
                           for t in range(1, T) for s in active_evs.keys()))

