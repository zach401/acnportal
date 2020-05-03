import random
import unittest
from collections import namedtuple

from acnportal.algorithms import *
from acnportal.algorithms.tests.generate_test_cases import *
from acnportal.algorithms.tests.testing_interface import TestingInterface

CURRENT_TIME = 0
PERIOD = 5

# -----------------------------------------------------------------------------
# Algorithms to Test
# -----------------------------------------------------------------------------
algorithms = {
    'FCFS': SortedSchedulingAlgo(first_come_first_served),
    'LLF': SortedSchedulingAlgo(least_laxity_first),
    'EDF': SortedSchedulingAlgo(earliest_deadline_first),
    'LCFS': SortedSchedulingAlgo(last_come_first_served),
    'LRPT': SortedSchedulingAlgo(largest_remaining_processing_time),
    'RR': RoundRobin(first_come_first_served)
}

# -----------------------------------------------------------------------------
# Test Suite
# -----------------------------------------------------------------------------
Scenario = namedtuple('Scenario', ['name', 'schedule',
                                   'interface', 'congested'])


class BaseAlgorithmTest(unittest.TestCase):
    @staticmethod
    def get_interface(limit):
        raise NotImplementedError('BaseAlgorithmTest is an abstract class. '
                                  'Must implement get_interface method.')

    @classmethod
    def setUpClass(cls):
        cls.scenarios = []

    def setUp(self):
        self.scenarios = self.__class__.scenarios

    def test_all_rates_less_than_evse_limit(self):
        for scenario in self.scenarios:
            with self.subTest(msg=f'{scenario.name}'):
                for station_id, rates in scenario.schedule.items():
                    self.assertLessEqual(
                        scenario.schedule[station_id],
                        scenario.interface.max_pilot_signal(station_id))

    def test_all_rates_less_than_session_max_rates(self):
        for scenario in self.scenarios:
            with self.subTest(msg=f'{scenario.name}'):
                sessions = scenario.interface.active_sessions()
                for session in sessions:
                    station_id = session.station_id
                    self.assertLessEqual(
                        scenario.schedule[station_id],
                        session.max_rates[0])

    def test_energy_requested_not_exceeded(self):
        for scenario in self.scenarios:
            with self.subTest(msg=f'{scenario.name}'):
                sessions = scenario.interface.active_sessions()
                for session in sessions:
                    station_id = session.station_id
                    self.assertLessEqual(
                        scenario.schedule[station_id],
                        scenario.interface.remaining_amp_periods(session))

    def test_infrastructure_limits_satisfied(self):
        for scenario in self.scenarios:
            with self.subTest(msg=f'{scenario.name}'):
                self.assertTrue(
                    scenario.interface.is_feasible(scenario.schedule))

    def test_all_rates_at_max(self):
        for scenario in self.scenarios:
            with self.subTest(msg=f'{scenario.name}'):
                if scenario.congested:
                    self.skipTest('Test case is congested, should not expect  '
                                  'all rates to be at max.')
                for station_id, rates in scenario.schedule.items():
                    max_pilot = scenario.interface.max_pilot_signal(station_id)
                    self.assertTrue(np.isclose(rates,
                                               max_pilot,
                                               atol=1e-4))


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestTwoStations(BaseAlgorithmTest):
    @staticmethod
    def two_station(limit):
        network = single_phase_single_constraint(2, limit)
        sessions = session_generator(num_sessions=2,
                                     arrivals=[0] * 2,
                                     departures=[12] * 2,
                                     requested_energy=[3.3] * 2,
                                     remaining_energy=[3.3] * 2,
                                     max_rates=[32] * 2)
        data = {'active_sessions': sessions,
                'infrastructure_info': network,
                'current_time': CURRENT_TIME,
                'period': PERIOD
                }
        return TestingInterface(data)

    @classmethod
    def setUpClass(cls):
        cls.scenarios = []
        for limit in [40, 64]:
            congested = limit < 64
            interface = cls.two_station(limit)
            for algo_name, algo in algorithms.items():
                algo.register_interface(interface)
                schedule = algo.run()
                scenario_name = f'algorithm: {algo_name}, capacity: {limit}'
                cls.scenarios.append(Scenario(scenario_name, schedule,
                                              interface, congested))


class Test100StationsSinglePhase(BaseAlgorithmTest):
    @staticmethod
    def get_interface(limit):
        N = 100
        network = single_phase_single_constraint(N, limit)
        sessions = session_generator(num_sessions=N,
                                     arrivals=[0] * N,
                                     departures=[12] * N,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[3.3] * N,
                                     max_rates=[32] * N)
        data = {'active_sessions': sessions,
                'infrastructure_info': network,
                'current_time': CURRENT_TIME,
                'period': PERIOD
                }
        return TestingInterface(data)

    @classmethod
    def setUpClass(cls):
        cls.scenarios = []
        for limit in [1500, 3200]:
            congested = limit < 3200
            interface = cls.get_interface(limit)
            for algo_name, algo in algorithms.items():
                algo.register_interface(interface)
                schedule = algo.run()
                scenario_name = f'algorithm: {algo_name}, capacity: {limit}'
                cls.scenarios.append(Scenario(scenario_name, schedule,
                                              interface, congested))


class Test102StationsThreePhase(BaseAlgorithmTest):
    @staticmethod
    def get_interface(limit):
        num_sessions = 102
        # Network is WAY over designed to deal with unbalance and ensure
        # all EVs can charge at their maximum rate
        network = three_phase_balanced_network(num_sessions // 3, limit)
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
        return TestingInterface(data)

    @classmethod
    def setUpClass(cls):
        cls.scenarios = []
        for limit in [1500, 3200]:
            congested = limit < 3200
            interface = cls.get_interface(limit)
            for algo_name, algo in algorithms.items():
                algo.register_interface(interface)
                schedule = algo.run()
                scenario_name = f'algorithm: {algo_name}, capacity: {limit}'
                cls.scenarios.append(Scenario(scenario_name, schedule,
                                              interface, congested))


del BaseAlgorithmTest
