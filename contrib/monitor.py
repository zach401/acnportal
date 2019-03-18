import copy
import math

import numpy as np
import pandas as pd


def create_pod_limits(test_case):
    N = len(test_case)
    CC_pod = {'CA-322', 'CA-493', 'CA-496', 'CA-320', 'CA-495', 'CA-321', 'CA-323', 'CA-494'}
    AV_pod = {'CA-324', 'CA-325', 'CA-326', 'CA-327', 'CA-489', 'CA-490', 'CA-491', 'CA-492'}

    CC_pod_group = [0 for i in range(N)]
    AV_pod_group = [0 for i in range(N)]
    for i in range(N):
        if test_case[i].station_id in CC_pod:
            CC_pod_group[i] = 1
        if test_case[i].station_id in AV_pod:
            AV_pod_group[i] = 1
    return [(np.array(CC_pod_group), 80000), (np.array(AV_pod_group), 80000)]
    # return np.array([CC_pod_group, AV_pod_group]), np.array([80000, 80000])


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
    groups['ALL'] = AB | BC | CA
    groups['AB'] = AB
    groups['BC'] = BC
    groups['CA'] = CA
    groups['AV-Pod'] = AV_pod
    groups['CC-Pod'] = CC_pod
    return groups


def get_session_phase_groups(ids):
    groups = get_groupings()
    session_groups = {}
    session_groups['AB'] = [id for id in ids if id in groups['AB']]
    session_groups['BC'] = [id for id in ids if id in groups['BC']]
    session_groups['CA'] = [id for id in ids if id in groups['CA']]
    session_groups['AV-Pod'] = [id for id in ids if id in groups['AV-Pod']]
    session_groups['CC-Pod'] = [id for id in ids if id in groups['CC-Pod']]
    return session_groups


def get_phase_grouping(test_case):
    N = len(test_case)
    groups = get_groupings()

    AB_group = [0 for i in range(N)]
    BC_group = [0 for i in range(N)]
    CA_group = [0 for i in range(N)]

    for i in range(N):
        if test_case[i].station_id in groups['AB']:
            AB_group[i] = 1
        if test_case[i].station_id in groups['BC']:
            BC_group[i] = 1
        if test_case[i].station_id in groups['CA']:
            CA_group[i] = 1
    return np.array(AB_group), np.array(BC_group), np.array(CA_group)


def post_process(EVs, pilot, processor, processor_kwargs):
    # Need to make a deep copy of EVs since we will be changing the EVs state
    working_EVs = copy.deepcopy(EVs)

    T = int(math.ceil(np.max([ev.departure for ev in working_EVs])))
    post_pilot = {ev.session_id: np.zeros((T,)) for ev in working_EVs}
    delivered = {ev.session_id: 0 for ev in working_EVs}
    for t in range(T):
        active_set = [ev for ev in working_EVs if t >= ev.arrival and t <= ev.departure and ev.remaining_demand() >= 6]
        active_pilots = {ev.session_id: pilot[ev.session_id][t] for ev in active_set}

        post_pilot_t = processor(active_set, active_pilots, **processor_kwargs)

        for ev in active_set:
            post_pilot[ev.session_id][t] = post_pilot_t[ev.session_id]
            ev.delivered += post_pilot_t[ev.session_id]
            # ev.max_rate = min(ev.max_rate, max(ev.remaining_demand(),0))

    return post_pilot


def discretize_rates(rate, inc, min_rate=0, max_rate=32):
    """ Returns a rate rounded down to the next valid discrete rate

    :param rate:        desired rate
    :param inc:         increment between discrete rates
    :param min_rate:    lowest rate available
    :param max_rate:    highest rate available
    :return:            rate rounded down to the next valid discrete rate
    """
    if inc > 0:
        new_rate = int(rate / inc) * inc
    else:
        new_rate = rate
    return max(min(new_rate, max_rate), min_rate)


def get_obj(pilots):
    T = len(pilots[0])
    cost_vec = np.array([t - T for t in range(T)])
    obj = 0
    for key in pilots:
        obj += pilots[key].dot(cost_vec)
    return obj


def sum_cols(df, cols):
    if len(cols) <= 0:
        return None
    acc = df[cols[0]].copy()
    for i in range(1, len(cols)):
        acc += df[cols[i]]
    return acc


def get_phase_currents(rates):
    groups = get_session_phase_groups(list(rates.keys()))
    phase_currents = pd.DataFrame()
    for phase in ['AB', 'BC', 'CA']:
        phase_currents[phase] = sum_cols(rates, groups[phase])
    return phase_currents


def get_line_currents(rates):
    phase_currents = get_phase_currents(rates)

    # Define phase conversions
    re_im = {'re': 1, 'im': 1j}
    phaseAB = {'re': np.cos(np.deg2rad(30)), 'im': np.sin(np.deg2rad(30))}
    phaseBC = {'re': np.cos(np.deg2rad(-90)), 'im': np.sin(np.deg2rad(-90))}
    phaseCA = {'re': np.cos(np.deg2rad(150)), 'im': np.sin(np.deg2rad(150))}

    line_currents = pd.DataFrame()
    line_currents['A'] = sum(
        (re_im[part] * phase_currents['AB'] * phaseAB[part] - re_im[part] * phase_currents['CA'] * phaseCA[part] for
         part in ['re', 'im']))
    line_currents['B'] = sum(
        (re_im[part] * phase_currents['BC'] * phaseBC[part] - re_im[part] * phase_currents['AB'] * phaseAB[part] for
         part in ['re', 'im']))
    line_currents['C'] = sum(
        (re_im[part] * phase_currents['CA'] * phaseCA[part] - re_im[part] * phase_currents['BC'] * phaseBC[part] for
         part in ['re', 'im']))
    return line_currents


def get_primary_currents(rates):
    secondary_line_currents = get_line_currents(rates)
    primary_line_currents = pd.DataFrame()
    primary_line_currents['A'] = (1 / 4) * (secondary_line_currents['A'] - secondary_line_currents['C'])
    primary_line_currents['B'] = (1 / 4) * (secondary_line_currents['B'] - secondary_line_currents['A'])
    primary_line_currents['C'] = (1 / 4) * (secondary_line_currents['C'] - secondary_line_currents['B'])
    return primary_line_currents


def get_pod_currents(rates):
    groups = get_session_phase_groups(list(rates.keys()))
    pod_currents = pd.DataFrame()
    for pod in ['AV-Pod', 'CC-Pod']:
        pod_currents[pod] = sum_cols(rates, groups[pod])
    return pod_currents
