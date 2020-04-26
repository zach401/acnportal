import pytest
import numpy as np
from acnportal.algorithms.tests.generate_test_cases import session_generator, single_phase_single_constraint, three_phase_balanced_network
from acnportal.algorithms.tests.test_interface import TestInterface
import random

from acnportal.algorithms import *

CURRENT_TIME = 0
PERIOD = 5


def tiny_single_phase_network(algorithm, limit):
    network = single_phase_single_constraint(2, limit)
    sessions = session_generator(num_sessions=2, arrivals=[0]*2, departures=[12]*2,
                                 requested_energy=[3.3]*2, remaining_energy=[3.3]*2,
                                 max_rates=[32]*2)
    data = {'active_sessions': sessions,
            'infrastructure_info': network,
            'current_time': CURRENT_TIME,
            'period': PERIOD
            }
    interface = TestInterface(data)
    algorithm.register_interface(interface)
    schedule = algorithm.run()
    return {'schedule': schedule,
            'interface': interface}


def large_single_phase_network(algorithm, limit):
    N = 100
    network = single_phase_single_constraint(N, limit)
    sessions = session_generator(num_sessions=N, arrivals=[0]*N, departures=[12]*N,
                                 requested_energy=[3.3]*N, remaining_energy=[3.3]*N,
                                 max_rates=[32]*N)
    data = {'active_sessions': sessions,
            'infrastructure_info': network,
            'current_time': CURRENT_TIME,
            'period': PERIOD
            }
    interface = TestInterface(data)
    algorithm.register_interface(interface)
    schedule = algorithm.run()
    return {'schedule': schedule,
            'interface': interface}


def large_three_phase_network(algorithm, limit):
    num_sessions = 54
    # Network must we WAY over designed to deal with unbalance and ensure all EVs can charge at their maximum rate
    network = three_phase_balanced_network(num_sessions//3, limit)
    sessions = session_generator(num_sessions=num_sessions,
                                 arrivals=[0] * num_sessions,
                                 departures=[2] * num_sessions,
                                 requested_energy=[10] * num_sessions,
                                 remaining_energy=[10] * num_sessions,
                                 max_rates=[32] * num_sessions)
    random.shuffle(sessions)
    data = {'active_sessions': sessions,
            'infrastructure_info': network,
            'current_time': CURRENT_TIME,
            'period': PERIOD
            }
    interface = TestInterface(data)
    algorithm.register_interface(interface)
    schedule = algorithm.run()
    return {'schedule': schedule,
            'interface': interface}


scenarios = {'tiny_single_phase_network': tiny_single_phase_network,
             'large_single_phase_network': large_single_phase_network,
             'large_three_phase_network': large_three_phase_network}

congested_limits = {'tiny_single_phase_network': 48,
                    'large_single_phase_network': 2500,
                    'large_three_phase_network': 30*18}

uncongested_limits = {'tiny_single_phase_network': 64,
                      'large_single_phase_network': 3200,
                      'large_three_phase_network': 64*18}

algorithms = {'FCFS': SortedSchedulingAlgo(first_come_first_served),
              'LLF': SortedSchedulingAlgo(least_laxity_first),
              'EDF': SortedSchedulingAlgo(earliest_deadline_first),
              'LCFS': SortedSchedulingAlgo(last_come_first_served),
              'LRPT': SortedSchedulingAlgo(largest_remaining_processing_time),
              'RR': RoundRobin(first_come_first_served)}

congested_results = {}
for alg_name, alg in algorithms.items():
    for scenario_name, scenario in scenarios.items():
        congested_results[f'congested.{scenario_name}.{alg_name}'] = scenario(alg, congested_limits[scenario_name])

uncongested_results = {}
for alg_name, alg in algorithms.items():
    for scenario_name, scenario in scenarios.items():
        uncongested_results[f'uncongested.{scenario_name}.{alg_name}'] = scenario(alg, uncongested_limits[scenario_name])

all_results = dict(**congested_results, **uncongested_results)
@pytest.mark.parametrize('scenario', all_results.values(), ids=all_results.keys())
class TestBasicAlgorithms:
    def test_all_rates_less_than_evse_limit(self, scenario):
        for station_id, rates in scenario['schedule'].items():
            assert scenario['schedule'][station_id] <= scenario['interface'].max_pilot_signal(station_id)

    def test_all_rates_less_than_session_max_rates(self, scenario):
        interface = scenario['interface']
        sessions = interface.active_sessions()
        for session in sessions:
            station_id = session.station_id
            assert scenario['schedule'][station_id] <= session.max_rates[0]

    def test_energy_requested_not_exceeded(self, scenario):
        interface = scenario['interface']
        sessions = interface.active_sessions()
        for session in sessions:
            station_id = session.station_id
            assert scenario['schedule'][station_id] <= interface.remaining_amp_periods(session)

    def test_infrastructure_limits_satisfied(self, scenario):
        scenario['interface'].is_feasible(scenario['schedule'])


@pytest.mark.parametrize('scenario', uncongested_results.values(), ids=uncongested_results.keys())
def test_all_rates_at_max(scenario):
    # In these scenarios it is possible to charge all EVs at their maximum rate
    for station_id, rates in scenario['schedule'].items():
        assert np.isclose(scenario['schedule'][station_id],
                          scenario['interface'].max_pilot_signal(station_id),
                          atol=1e-4)
