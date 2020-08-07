import random
import unittest
from unittest.mock import Mock
from numpy import testing as nptest
from collections import namedtuple

from acnportal.algorithms import *
from acnportal.algorithms.tests.generate_test_cases import *
from acnportal.algorithms.tests.testing_interface import TestingInterface
from acnportal.algorithms import UpperBoundEstimatorBase

CURRENT_TIME = 0
PERIOD = 5

# -----------------------------------------------------------------------------
# Algorithms to Test
# -----------------------------------------------------------------------------
algorithms = {
    "FCFS": SortedSchedulingAlgo(first_come_first_served),
    "LLF": SortedSchedulingAlgo(least_laxity_first),
    "EDF": SortedSchedulingAlgo(earliest_deadline_first),
    "LCFS": SortedSchedulingAlgo(last_come_first_served),
    "LRPT": SortedSchedulingAlgo(largest_remaining_processing_time),
    "RR": RoundRobin(first_come_first_served),
}

# -----------------------------------------------------------------------------
# Test Suite
# -----------------------------------------------------------------------------
Scenario = namedtuple("Scenario", ["name", "interface", "congested"])


class BaseAlgorithmTest(unittest.TestCase):
    def setUp(self):
        self.algo = None

    @staticmethod
    def get_scenarios():
        return []

    def test_output_feasible(self):
        scenarios = self.get_scenarios()
        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                self.algo.register_interface(scenario.interface)
                schedule = self.algo.run()
                self.run_tests(
                    scenario.interface.active_sessions(),
                    schedule,
                    scenario.interface,
                    scenario.congested,
                )

    def run_tests(self, sessions, schedule, interface, congested=True):
        with self.subTest(msg="test_all_rates_less_than_evse_limit"):
            self._test_all_rates_less_than_evse_limit(schedule, interface)

        with self.subTest(msg="test_all_rates_less_than_session_max_rates"):
            self._test_all_rates_less_than_session_max_rates(sessions, schedule)

        with self.subTest("test_in_allowable_rates"):
            self._test_in_allowable_rates(sessions, schedule, interface)

        with self.subTest("test_energy_requested_not_exceeded"):
            self._test_energy_requested_not_exceeded(sessions, schedule, interface)

        with self.subTest("test_infrastructure_limits_satisfied"):
            self._test_infrastructure_limits_satisfied(schedule, interface)

        if not congested:
            with self.subTest("test_all_rates_at_max"):
                self._test_all_rates_at_max(sessions, schedule, interface)

    def _test_all_rates_less_than_evse_limit(self, schedule, interface):
        for station_id, rates in schedule.items():
            self.assertLessEqual(rates, interface.max_pilot_signal(station_id))

    def _test_all_rates_less_than_session_max_rates(self, sessions, schedule):
        for session in sessions:
            station_id = session.station_id
            self.assertLessEqual(schedule[station_id], session.max_rates[0])

    def _test_all_rates_greater_than_session_min_rates(self, sessions, schedule):
        for session in sessions:
            station_id = session.station_id
            self.assertGreaterEqual(schedule[station_id], session.min_rates[0])

    def _test_in_allowable_rates(self, sessions, schedule, interface):
        for session in sessions:
            station_id = session.station_id
            (is_continuous, allowable,) = interface.allowable_pilot_signals(station_id)
            if is_continuous:
                self.assertGreaterEqual(schedule[station_id], allowable[0])
                self.assertLessEqual(schedule[station_id], allowable[1])
            else:
                self.assertIn(schedule[station_id], allowable)

    def _test_energy_requested_not_exceeded(self, sessions, schedule, interface):
        for session in sessions:
            station_id = session.station_id
            self.assertLessEqual(
                schedule[station_id], interface.remaining_amp_periods(session),
            )

    def _test_infrastructure_limits_satisfied(self, schedule, interface):
        self.assertTrue(interface.is_feasible(schedule))

    def _test_all_rates_at_max(self, sessions, schedule, interface):
        infrastructure = interface.infrastructure_info()
        for session in sessions:
            i = infrastructure.get_station_index(session.station_id)
            ub = min(infrastructure.max_pilot[i], session.max_rates[0])
            rates = schedule[session.session_id]
            self.assertTrue(True)
            nptest.assert_almost_equal(rates, ub, decimal=4)


# Two Station Test Case
def two_station(
    limit, continuous, session_max_rate, session_min_rate=0, remaining_energy=None
):
    if continuous:
        allowable = [[0, 32]] * 2
    else:
        allowable = [[0] + list(range(8, 33))] * 2
    if remaining_energy is None:
        remaining_energy = [3.3, 3.3]
    network = single_phase_single_constraint(
        2, limit, allowable_pilots=allowable, is_continuous=[continuous] * 2
    )
    sessions = session_generator(
        num_sessions=2,
        arrivals=[0] * 2,
        departures=[12] * 2,
        requested_energy=[3.3] * 2,
        remaining_energy=remaining_energy,
        min_rates=[session_min_rate] * 2,
        max_rates=[session_max_rate] * 2,
    )
    data = {
        "active_sessions": sessions,
        "infrastructure_info": network,
        "current_time": CURRENT_TIME,
        "period": PERIOD,
    }
    return TestingInterface(data)


class TestTwoStationsBase(BaseAlgorithmTest):
    def setUp(self):
        self.algo = SortedSchedulingAlgo(first_come_first_served)

    @staticmethod
    def get_scenarios():
        scenarios = []
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
                for session_min_rate in [0, 8]:
                    for continuous in [True, False]:
                        congested = limit < 64
                        interface = two_station(
                            limit, continuous, session_max_rate, session_min_rate
                        )
                        scenario_name = (
                            f"capacity: {limit}, "
                            f"session max: {session_max_rate},"
                            f"session min: {session_min_rate},"
                            f"continuous pilot: {continuous}"
                        )
                        scenarios.append(Scenario(scenario_name, interface, congested))
        return scenarios


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestTwoStationsFCFS(TestTwoStationsBase):
    def setUp(self):
        self.algo = SortedSchedulingAlgo(first_come_first_served)


class TestTwoStationsEDF(TestTwoStationsBase):
    def setUp(self):
        self.algo = SortedSchedulingAlgo(earliest_deadline_first)


class TestTwoStationsLLF(TestTwoStationsBase):
    def setUp(self):
        self.algo = SortedSchedulingAlgo(least_laxity_first)


class TestTwoStationsLCFS(TestTwoStationsBase):
    def setUp(self):
        self.algo = SortedSchedulingAlgo(last_come_first_served)


class TestTwoStationsLRPT(TestTwoStationsBase):
    def setUp(self):
        self.algo = SortedSchedulingAlgo(largest_remaining_processing_time)


class TestTwoStationsRR(TestTwoStationsBase):
    def setUp(self):
        self.algo = RoundRobin(first_come_first_served)


# class TestUninterruptedCharging(BaseAlgorithmTest):
#     @classmethod
#     def setUpClass(cls):
#         cls.scenarios = []
#         for limit in [16, 32, 40]:
#             for continuous in [True, False]:
#                 congested = True
#                 interface = two_station(limit, continuous, 32, 0)
#                 for algo_name, algo in algorithms.items():
#                     algo.uninterrupted_charging = True
#                     algo.register_interface(interface)
#                     schedule = algo.run()
#                     scenario_name = (
#                         f"algorithm: {algo_name}, "
#                         f"capacity: {limit}, "
#                         f"continuous pilot: {continuous}"
#                     )
#                     cls.scenarios.append(
#                         Scenario(scenario_name, schedule, interface, congested)
#                     )
#
#     def test_charging_not_interrupted(self):
#         for scenario in self.scenarios:
#             with self.subTest(msg=f"{scenario.name}"):
#                 sessions = scenario.interface.active_sessions()
#                 for session in sessions:
#                     scheduled = scenario.schedule[session.station_id]
#                     minimum_pilot = scenario.interface.min_pilot_signal(
#                         session.station_id
#                     )
#                     self.assertGreaterEqual(scheduled, minimum_pilot)
#
#
# class TestTwoStationsMinRatesInfeasible(unittest.TestCase):
#     def test_sorted_min_rates_infeasible(self):
#         limit = 16
#         max_rate = 32
#         min_rate = 16
#         for continuous in [True, False]:
#             interface = two_station(limit, continuous, max_rate, min_rate)
#             for algo_name, algo in algorithms.items():
#                 scenario_name = (
#                     f"algorithm: {algo_name}, "
#                     f"capacity: {limit}, "
#                     f"continuous pilot: {continuous}"
#                 )
#                 with self.subTest(msg=f"{scenario_name}"):
#                     algo.register_interface(interface)
#                     with self.assertRaisesRegex(
#                         ValueError,
#                         "Charging all sessions at "
#                         "their lower bound is not "
#                         "feasible.",
#                     ):
#                         algo.run()
#
#
# class TestTwoStationsEnergyBinding(BaseAlgorithmTest):
#     @classmethod
#     def setUpClass(cls):
#         cls.scenarios = []
#         limit = 64
#         for continuous in [True, False]:
#             congested = True
#             interface = two_station(limit, continuous, 32, 0, [0.3, 0.05])
#             for algo_name, algo in algorithms.items():
#                 algo.register_interface(interface)
#                 schedule = algo.run()
#                 scenario_name = (
#                     f"algorithm: {algo_name}, "
#                     f"capacity: {limit}, "
#                     f"continuous pilot: {continuous}"
#                 )
#                 cls.scenarios.append(
#                     Scenario(scenario_name, schedule, interface, congested)
#                 )
#
#
# class TestTwoStationsEnergyBindingUninterrupted(BaseAlgorithmTest):
#     @classmethod
#     def setUpClass(cls):
#         cls.scenarios = []
#         limit = 64
#         for continuous in [True, False]:
#             congested = True
#             interface = two_station(limit, continuous, 32, 0, [0.3, 0.05])
#             for algo_name, algo in algorithms.items():
#                 algo.uninterrupted_charging = True
#                 algo.register_interface(interface)
#                 schedule = algo.run()
#                 scenario_name = (
#                     f"algorithm: {algo_name}, "
#                     f"capacity: {limit}, "
#                     f"continuous pilot: {continuous}"
#                 )
#                 cls.scenarios.append(
#                     Scenario(scenario_name, schedule, interface, congested)
#                 )
#
#
# class TestEstimateMaxRate(BaseAlgorithmTest):
#     @classmethod
#     def setUpClass(cls):
#         estimator_mock = UpperBoundEstimatorBase()
#         estimator_mock.get_maximum_rates = Mock(return_value={"0": 16, "1": 12})
#         cls.scenarios = []
#         for limit in [16, 32, 40]:
#             for continuous in [True, False]:
#                 congested = True
#                 interface = two_station(limit, continuous, 32, 0)
#                 for algo_name, algo in algorithms.items():
#                     algo.estimate_max_rate = True
#                     algo.max_rate_estimator = estimator_mock
#                     algo.register_interface(interface)
#                     schedule = algo.run()
#                     scenario_name = (
#                         f"algorithm: {algo_name}, "
#                         f"capacity: {limit}, "
#                         f"continuous pilot: {continuous}"
#                     )
#                     cls.scenarios.append(
#                         Scenario(scenario_name, schedule, interface, congested)
#                     )
#
#     def test_max_rate_estimator_not_exceeded(self):
#         for scenario in self.scenarios:
#             with self.subTest(msg=f"{scenario.name}"):
#                 self.assertLessEqual(scenario.schedule["0"][0], 16)
#                 self.assertLessEqual(scenario.schedule["1"][0], 12)
#
#
# class Test30StationsSinglePhase(BaseAlgorithmTest):
#     @staticmethod
#     def get_interface(limit):
#         N = 30
#         network = single_phase_single_constraint(N, limit)
#         sessions = session_generator(
#             num_sessions=N,
#             arrivals=[0] * N,
#             departures=[12] * N,
#             requested_energy=[3.3] * N,
#             remaining_energy=[3.3] * N,
#             max_rates=[32] * N,
#         )
#         data = {
#             "active_sessions": sessions,
#             "infrastructure_info": network,
#             "current_time": CURRENT_TIME,
#             "period": PERIOD,
#         }
#         return TestingInterface(data)
#
#     @classmethod
#     def setUpClass(cls):
#         cls.scenarios = []
#         for limit in [1500, 3200]:
#             congested = limit < 3200
#             interface = cls.get_interface(limit)
#             for algo_name, algo in algorithms.items():
#                 algo.register_interface(interface)
#                 schedule = algo.run()
#                 scenario_name = f"algorithm: {algo_name}, capacity: {limit}"
#                 cls.scenarios.append(
#                     Scenario(scenario_name, schedule, interface, congested)
#                 )
#
#
# class Test30StationsThreePhase(BaseAlgorithmTest):
#     @staticmethod
#     def get_interface(limit):
#         num_sessions = 30
#         # Network is WAY over designed to deal with unbalance and ensure
#         # all EVs can charge at their maximum rate
#         network = three_phase_balanced_network(num_sessions // 3, limit)
#         sessions = session_generator(
#             num_sessions=num_sessions,
#             arrivals=[0] * num_sessions,
#             departures=[2] * num_sessions,
#             requested_energy=[10] * num_sessions,
#             remaining_energy=[10] * num_sessions,
#             max_rates=[32] * num_sessions,
#         )
#         random.shuffle(sessions)
#         data = {
#             "active_sessions": sessions,
#             "infrastructure_info": network,
#             "current_time": CURRENT_TIME,
#             "period": PERIOD,
#         }
#         return TestingInterface(data)
#
#     @classmethod
#     def setUpClass(cls):
#         cls.scenarios = []
#         for limit in [1500, 3200]:
#             congested = limit < 3200
#             interface = cls.get_interface(limit)
#             for algo_name, algo in algorithms.items():
#                 algo.register_interface(interface)
#                 schedule = algo.run()
#                 scenario_name = f"algorithm: {algo_name}, capacity: {limit}"
#                 cls.scenarios.append(
#                     Scenario(scenario_name, schedule, interface, congested)
#                 )

del BaseAlgorithmTest
del TestTwoStationsBase
# del TestTwoStationsMinRatesFeasibleBase
