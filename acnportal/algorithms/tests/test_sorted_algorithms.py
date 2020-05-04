import random
import unittest
from numpy import testing as nptest
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
                        rates,
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

    def test_all_rates_greater_than_session_min_rates(self):
        for scenario in self.scenarios:
            with self.subTest(msg=f'{scenario.name}'):
                sessions = scenario.interface.active_sessions()
                for session in sessions:
                    station_id = session.station_id
                    self.assertGreaterEqual(
                        scenario.schedule[station_id],
                        session.min_rates[0])

    def test_in_allowable_rates(self):
        for scenario in self.scenarios:
            with self.subTest(msg=f'{scenario.name}'):
                sessions = scenario.interface.active_sessions()
                for session in sessions:
                    station_id = session.station_id
                    is_continuous, allowable = \
                        scenario.interface.allowable_pilot_signals(station_id)
                    if is_continuous:
                        self.assertGreaterEqual(
                            scenario.schedule[station_id],
                            allowable[0])
                        self.assertLessEqual(
                            scenario.schedule[station_id],
                            allowable[1])
                    else:
                        self.assertIn(scenario.schedule[station_id],
                                      allowable)

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
                sessions = scenario.interface.active_sessions()
                infrastructure = scenario.interface.infrastructure_info()
                for session in sessions:
                    i = infrastructure.get_station_index(session.station_id)
                    ub = min(infrastructure.max_pilot[i],
                             session.max_rates[0])
                    rates = scenario.schedule[session.session_id]
                    nptest.assert_almost_equal(rates, ub, decimal=4)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestTwoStations(BaseAlgorithmTest):
    @staticmethod
    def two_station(limit, continuous, session_max_rate, session_min_rate=0):
        if continuous:
            allowable = [[0, 32]] * 2
        else:
            allowable = [[0] + list(range(8, 33))] * 2
        network = single_phase_single_constraint(2,
                                                 limit,
                                                 allowable_pilots=allowable,
                                                 is_continuous=[continuous]*2
                                                 )
        sessions = session_generator(num_sessions=2,
                                     arrivals=[0] * 2,
                                     departures=[12] * 2,
                                     requested_energy=[3.3] * 2,
                                     remaining_energy=[3.3] * 2,
                                     min_rates=[session_min_rate] * 2,
                                     max_rates=[session_max_rate] * 2)
        data = {'active_sessions': sessions,
                'infrastructure_info': network,
                'current_time': CURRENT_TIME,
                'period': PERIOD
                }
        return TestingInterface(data)

    @classmethod
    def setUpClass(cls):
        cls.scenarios = []
        # Test one limit where constraints are binding (40 A),
        # one where constraint will be met exactly by charging at max (64 A),
        # and one where constraints are not binding at all (80 A).
        for limit in [40, 64, 80]:
            # To ensure that both station limits and session limits are
            # satisfied, consider several max_rate parameters.
            # At 16 A, session limit is below station limit (32)
            # At 32 A, session limit equals station limit
            # At 40 A, session limit is greater than station limit.
            for session_max_rate in [16, 32, 40]:
                # Consider both continuous and discrete pilot signals
                for continuous in [True, False]:
                    congested = limit < 64
                    interface = cls.two_station(limit, continuous,
                                                session_max_rate)
                    for algo_name, algo in algorithms.items():
                        algo.register_interface(interface)
                        schedule = algo.run()
                        scenario_name = (f'algorithm: {algo_name}, '
                                         f'capacity: {limit}, '
                                         f'session max: {session_max_rate},'
                                         f'continuous pilot: {continuous}')
                        cls.scenarios.append(Scenario(scenario_name, schedule,
                                                      interface, congested))


class TestTwoStationsMinRates(TestTwoStations):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = []
        limit = 16
        for continuous in [True, False]:
            congested = True
            interface = cls.two_station(limit, continuous, 32, 8)
            for algo_name, algo in algorithms.items():
                algo.register_interface(interface)
                schedule = algo.run()
                scenario_name = (f'algorithm: {algo_name}, '
                                 f'capacity: {limit}, '
                                 f'continuous pilot: {continuous}')
                cls.scenarios.append(Scenario(scenario_name, schedule,
                                              interface, congested))


class Test30StationsSinglePhase(BaseAlgorithmTest):
    @staticmethod
    def get_interface(limit):
        N = 30
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


class Test30StationsThreePhase(BaseAlgorithmTest):
    @staticmethod
    def get_interface(limit):
        num_sessions = 30
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
