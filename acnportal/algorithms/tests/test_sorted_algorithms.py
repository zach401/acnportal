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
Scenario = namedtuple(
    "Scenario",
    ["name", "interface", "assert_at_max", "uninterrupted", "estimate_max_rate"],
)


class BaseAlgorithmTest(unittest.TestCase):
    def setUp(self):
        self.algo = None
        self.max_rate_estimation = {}

    @staticmethod
    def get_scenarios():
        return []

    def test_output_feasible(self):
        scenarios = self.get_scenarios()
        for scenario in scenarios:
            self.algo.register_interface(scenario.interface)
            self.algo.uninterrupted = scenario.uninterrupted
            estimator_mock = UpperBoundEstimatorBase()
            estimator_mock.get_maximum_rates = Mock(
                return_value=self.max_rate_estimation
            )
            self.algo.max_rate_estimator = estimator_mock
            self.algo.estimate_max_rate = scenario.estimate_max_rate

            schedule = self.algo.run()
            self.run_tests(
                scenario.name,
                scenario.interface.active_sessions(),
                schedule,
                scenario.interface,
                scenario.assert_at_max,
                scenario.uninterrupted,
            )

    def run_tests(
        self,
        name,
        sessions,
        schedule,
        interface,
        assert_at_max=False,
        uninterrupted=False,
    ):
        with self.subTest(msg=f"test_all_rates_less_than_evse_limit - {name}"):
            self._test_all_rates_less_than_evse_limit(schedule, interface)

        with self.subTest(msg=f"test_all_rates_less_than_session_max_rates - {name}"):
            self._test_all_rates_less_than_session_max_rates(sessions, schedule)

        with self.subTest(f"test_in_allowable_rates - {name}"):
            self._test_in_allowable_rates(sessions, schedule, interface)

        with self.subTest(f"test_energy_requested_not_exceeded - {name}"):
            self._test_energy_requested_not_exceeded(sessions, schedule, interface)

        with self.subTest(f"test_infrastructure_limits_satisfied - {name}"):
            self._test_infrastructure_limits_satisfied(schedule, interface)

        if assert_at_max:
            with self.subTest(f"test_all_rates_at_max - {name}"):
                self._test_all_rates_at_max(sessions, schedule, interface)

        if uninterrupted:
            with self.subTest(f"test_charging_not_interrupted - {name}"):
                self._test_charging_not_interrupted(sessions, schedule, interface)

        if self.algo.estimate_max_rate:
            with self.subTest(f"test_max_rate_estimator_not_exceeded - {name}"):
                self._test_max_rate_estimator_not_exceeded(sessions, schedule)

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

    def _test_charging_not_interrupted(self, sessions, schedule, interface):
        for session in sessions:
            scheduled = schedule[session.station_id]
            minimum_pilot = interface.min_pilot_signal(session.station_id)
            remaining_energy = interface.remaining_amp_periods(session)

            # Algorithm should not exceed remaining energy in order to meet minimum
            # pilot.
            if minimum_pilot < remaining_energy:
                self.assertGreaterEqual(scheduled, minimum_pilot)

    def _test_max_rate_estimator_not_exceeded(self, sessions, schedule):
        for session in sessions:
            self.assertLessEqual(
                np.array(schedule[session.session_id]),
                self.max_rate_estimation[session.session_id],
            )


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


def big_three_phase_network(num_sessions=30, limit=1000):
    # Network is WAY over designed to deal with unbalance and ensure
    # all EVs can charge at their maximum rate
    network = three_phase_balanced_network(num_sessions // 3, limit)
    sessions = session_generator(
        num_sessions=num_sessions,
        arrivals=[0] * num_sessions,
        departures=[2] * num_sessions,
        requested_energy=[10] * num_sessions,
        remaining_energy=[10] * num_sessions,
        max_rates=[32] * num_sessions,
    )
    random.shuffle(sessions)
    data = {
        "active_sessions": sessions,
        "infrastructure_info": network,
        "current_time": CURRENT_TIME,
        "period": PERIOD,
    }
    return TestingInterface(data)


class TestTwoStationsBase(BaseAlgorithmTest):
    def setUp(self):
        self.algo = None
        self.max_rate_estimation = {"0": 16, "1": 12}

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
                    for session_energy_demands in [[3.3, 3.3], [0.3, 0.05]]:
                        # Consider continuous and discrete EVSEs
                        for continuous in [True, False]:
                            # Consider both interruptable and uninterrupted charging
                            for uninterrupted in [True, False]:
                                for estimate_max_rate in [True, False]:
                                    if (
                                        limit < 64
                                        or session_energy_demands == [0.3, 0.05]
                                        or estimate_max_rate
                                    ):
                                        assert_at_max = False
                                    else:
                                        assert_at_max = True
                                    interface = two_station(
                                        limit,
                                        continuous,
                                        session_max_rate,
                                        session_min_rate,
                                        remaining_energy=session_energy_demands,
                                    )
                                    scenario_name = (
                                        f"capacity: {limit}, "
                                        f"session max: {session_max_rate}, "
                                        f"session min: {session_min_rate}, "
                                        f"continuous pilot: {continuous}, "
                                        f"uninterrupted: {uninterrupted}, "
                                        f"estimate_max_rate: {estimate_max_rate} "
                                    )
                                    scenarios.append(
                                        Scenario(
                                            scenario_name,
                                            interface,
                                            assert_at_max,
                                            uninterrupted,
                                            estimate_max_rate,
                                        )
                                    )
        return scenarios


class TestThirtyStationsBase(BaseAlgorithmTest):
    def setUp(self):
        self.algo = None
        self.max_rate_estimation = {}  # Don't use max_rate_estimation for this test.

    @staticmethod
    def get_scenarios():
        scenarios = []
        for limit in [1500, 3200]:
            assert_at_max = limit > 3200
            interface = big_three_phase_network(limit=limit)
            scenario_name = f"capacity: {limit} "
            scenarios.append(
                Scenario(scenario_name, interface, assert_at_max, False, False)
            )
        return scenarios


class TestTwoStationsMinRatesInfeasible(unittest.TestCase):
    """ Check that error is thrown when minimum rates are not feasible. """

    def test_sorted_min_rates_infeasible(self):
        limit = 16
        max_rate = 32
        min_rate = 16
        for continuous in [True, False]:
            interface = two_station(limit, continuous, max_rate, min_rate)
            for algo_name, algo in algorithms.items():
                scenario_name = (
                    f"algorithm: {algo_name}, "
                    f"capacity: {limit}, "
                    f"continuous pilot: {continuous}"
                )
                with self.subTest(msg=f"{scenario_name}"):
                    algo.register_interface(interface)
                    with self.assertRaisesRegex(
                        ValueError,
                        "Charging all sessions at "
                        "their lower bound is not "
                        "feasible.",
                    ):
                        algo.run()


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestTwoStationsFCFS(TestTwoStationsBase):
    def setUp(self):
        super().setUp()
        self.algo = SortedSchedulingAlgo(first_come_first_served)


class TestThirtyStationsFCFS(TestThirtyStationsBase):
    def setUp(self):
        super().setUp()
        self.algo = SortedSchedulingAlgo(first_come_first_served)


class TestTwoStationsEDF(TestTwoStationsBase):
    def setUp(self):
        super().setUp()
        self.algo = SortedSchedulingAlgo(earliest_deadline_first)


class TestThirtyStationsEDF(TestThirtyStationsBase):
    def setUp(self):
        super().setUp()
        self.algo = SortedSchedulingAlgo(earliest_deadline_first)


class TestTwoStationsLLF(TestTwoStationsBase):
    def setUp(self):
        super().setUp()
        self.algo = SortedSchedulingAlgo(least_laxity_first)


class TestThirtyStationsLLF(TestThirtyStationsBase):
    def setUp(self):
        super().setUp()
        self.algo = SortedSchedulingAlgo(least_laxity_first)


class TestTwoStationsLCFS(TestTwoStationsBase):
    def setUp(self):
        super().setUp()
        self.algo = SortedSchedulingAlgo(last_come_first_served)


class TestThirtyStationsLCFS(TestThirtyStationsBase):
    def setUp(self):
        super().setUp()
        self.algo = SortedSchedulingAlgo(last_come_first_served)


class TestTwoStationsLRPT(TestTwoStationsBase):
    def setUp(self):
        super().setUp()
        self.algo = SortedSchedulingAlgo(largest_remaining_processing_time)


class TestThirtyStationsLRPT(TestThirtyStationsBase):
    def setUp(self):
        super().setUp()
        self.algo = SortedSchedulingAlgo(largest_remaining_processing_time)


class TestTwoStationsRR(TestTwoStationsBase):
    def setUp(self):
        super().setUp()
        self.algo = RoundRobin(first_come_first_served)


class TestThirtyStationsRR(TestThirtyStationsBase):
    def setUp(self):
        super().setUp()
        self.algo = RoundRobin(first_come_first_served)


del BaseAlgorithmTest
del TestTwoStationsBase
del TestThirtyStationsBase
# del TestTwoStationsMinRatesFeasibleBase
