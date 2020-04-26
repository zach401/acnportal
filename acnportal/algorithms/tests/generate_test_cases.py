import numpy as np


def session_generator(num_sessions, arrivals, departures, requested_energy, remaining_energy, max_rates,
                      min_rates=None, station_ids=None):
    sessions = []
    for i in range(num_sessions):
        station_id = station_ids[i] if station_ids is not None else f'{i}'
        session_id = f'{i}'
        min_rate = min_rates[i] if min_rates is not None else 0
        s = {'station_id': station_id,
             'session_id': session_id,
             'requested_energy': requested_energy[i],
             'energy_delivered': requested_energy[i] - remaining_energy[i],
             'arrival': arrivals[i],
             'departure': departures[i],
             'estimated_departure': None,
             'min_rates': min_rate,
             'max_rates': max_rates[i]}
        sessions.append(s)
    return sessions


def single_phase_single_constraint(num_evses, limit, max_pilot=32, min_pilot=8,
                                   allowable_pilots=None, is_continuous=None):
    if allowable_pilots is None:
        allowable_pilots = [[0, 32]] * num_evses
    if is_continuous is None:
        is_continuous = np.ones(num_evses, dtype=bool)
    infrastructure = {'constraint_matrix': np.ones((1, num_evses)),
                      'constraint_limits': np.array([limit]),
                      'phase_angles': np.zeros(num_evses),
                      'voltages': np.repeat(208, num_evses),
                      'constraint_index': ['all'],
                      'station_ids': [f'{i}' for i in range(num_evses)],
                      'max_pilot': np.repeat(max_pilot, num_evses),
                      'min_pilot': np.repeat(min_pilot, num_evses),
                      'allowable_pilots': allowable_pilots,
                      'is_continuous': is_continuous}
    return infrastructure


def three_phase_balanced_network(evses_per_phase, limit, max_pilot=32, min_pilot=8,
                                 allowable_pilots=None, is_continuous=None):
    N = evses_per_phase
    num_evses = 3 * evses_per_phase
    if allowable_pilots is None:
        allowable_pilots = [[0, 32]] * num_evses
    if is_continuous is None:
        is_continuous = np.ones(num_evses, dtype=bool)
    infrastructure = {'constraint_matrix': np.array([[1]  * N + [-1] * N + [0]  * N,
                                                     [0]  * N + [1]  * N + [-1] * N,
                                                     [-1] * N + [0]  * N + [1]  * N]),
                      'constraint_limits': np.repeat(limit, 3),
                      'phase_angles': np.array([0] * N + [-120] * N + [120] * N),
                      'voltages': np.repeat(208, 3*evses_per_phase),
                      'constraint_index': ['AB', 'BC', 'CA'],
                      'station_ids': [f'{i}' for i in range(num_evses)],
                      'max_pilot': np.repeat(max_pilot, num_evses),
                      'min_pilot': np.repeat(min_pilot, num_evses),
                      'allowable_pilots': allowable_pilots,
                      'is_continuous': is_continuous}
    return infrastructure

