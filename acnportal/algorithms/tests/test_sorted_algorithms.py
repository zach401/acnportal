# coding=utf-8
"""
Tests provided sorting algorithms under many cases.
"""
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
    def setUp(self) -> None:
        """
        Tests that a given algorithm provides feasible schedules to a simulation.
        The elements of feasibility tested here are:

        - A given charging rate is <= the maximum rate of the EVSE it is sent to.

        - A given charging rate is <= the maximum rate of the Session it is charging.

        - A given charging rate is in the allowable rate set of the Session it is
        charging.

        - No session is given more energy than it requested.

        - Infrastructure limits are satisfied.

        - TODO: what is the function of the assert at max rate?

        - Charging is uninterrupted (never goes to zero during the session unless the
        vehicle is done charging) if required.

        - A max rate estimation, if provided, is not exceeded during the session.

        Each algorithm test class has an algorithm (set by overriding this function
        and setting an actual algorithm) and a max_rate_estimation, which provides the
        return value of a mocked UpperBoundEstimator (used if max rate estimation is
        tested).

        The implementation of the _get_scenarios method details which charging
        scenarios should be run in this test class. A scenario is defined by a name,
        interface (usually a testing interface with static simulator data,
        see TestingInterface), and attributes assert_at_max, uninterrupted,
        and estimate_max rate, which dictate additional constraints under which the
        algorithm should operate.

        Returns:
            None.
        """
        self.algo = None
        self.max_rate_estimation = {}

    @staticmethod
    def _get_scenarios() -> List[Scenario]:
        return []

    def test_output_feasible(self) -> None:
        scenarios = self._get_scenarios()
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
            self._run_tests(
                scenario.name,
                scenario.interface.active_sessions(),
                schedule,
                scenario.interface,
                scenario.assert_at_max,
                scenario.uninterrupted,
            )

    def _run_tests(
        self,
        name: str,
        sessions: List[SessionInfo],
        schedule: Dict[str, List[float]],
        interface: Interface,
        assert_at_max: bool = False,
        uninterrupted: bool = False,
    ) -> None:
        with self.subTest(msg=f"test_all_rates_less_than_evse_limit - {name}"):
            self._test_all_rates_less_than_evse_limit(schedule, interface)

        with self.subTest(msg=f"test_all_rates_less_than_session_max_rates - {name}"):
            self._test_all_rates_less_than_session_max_rates(sessions, schedule)

        with self.subTest(
            msg=f"test_all_rates_greater_than_session_min_rates - {name}"
        ):
            self._test_all_rates_greater_than_session_min_rates(
                sessions, interface, schedule
            )

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

    def _test_all_rates_less_than_evse_limit(self, schedule, interface) -> None:
        for station_id, rates in schedule.items():
            self.assertLessEqual(rates, interface.max_pilot_signal(station_id))

    def _test_all_rates_less_than_session_max_rates(self, sessions, schedule) -> None:
        for session in sessions:
            station_id = session.station_id
            self.assertLessEqual(schedule[station_id][0], session.max_rates[0])

    def _test_all_rates_greater_than_session_min_rates(
        self, sessions, interface, schedule
    ) -> None:
        infrastructure = interface.infrastructure_info()
        for session in sessions:
            station_id = session.station_id
            station_index = infrastructure.get_station_index(session.station_id)
            threshold = (
                infrastructure.min_pilot[station_index]
                * infrastructure.voltages[station_index]
                / (60 / interface.period)
                / 1000
            )
            if session.remaining_demand > threshold:
                self.assertGreaterEqual(schedule[station_id][0], session.min_rates[0])
            else:
                self.assertEqual(schedule[station_id][0], 0)

    def _test_in_allowable_rates(self, sessions, schedule, interface) -> None:
        for session in sessions:
            station_id = session.station_id
            (is_continuous, allowable,) = interface.allowable_pilot_signals(station_id)
            if is_continuous:
                self.assertGreaterEqual(schedule[station_id], allowable[0])
                self.assertLessEqual(schedule[station_id], allowable[1])
            else:
                self.assertIn(schedule[station_id], allowable)

    def _test_energy_requested_not_exceeded(
        self, sessions, schedule, interface
    ) -> None:
        for session in sessions:
            station_id = session.station_id
            self.assertLessEqual(
                schedule[station_id], interface.remaining_amp_periods(session),
            )

    def _test_infrastructure_limits_satisfied(self, schedule, interface) -> None:
        self.assertTrue(interface.is_feasible(schedule))

    # noinspection PyMethodMayBeStatic
    def _test_all_rates_at_max(
        self, sessions, schedule, interface
    ) -> None:  # pylint: disable=no-self-use
        infrastructure = interface.infrastructure_info()
        for session in sessions:
            i = infrastructure.get_station_index(session.station_id)
            ub = min(infrastructure.max_pilot[i], session.max_rates[0])
            rates = schedule[session.station_id]
            nptest.assert_almost_equal(rates, ub, decimal=4)

    def _test_charging_not_interrupted(self, sessions, schedule, interface) -> None:
        for session in sessions:
            scheduled = schedule[session.station_id]
            minimum_pilot = interface.min_pilot_signal(session.station_id)
            remaining_energy = interface.remaining_amp_periods(session)

            # Algorithm should not exceed remaining energy in order to meet minimum
            # pilot.
            if minimum_pilot < remaining_energy:
                self.assertGreaterEqual(scheduled, minimum_pilot)

    def _test_max_rate_estimator_not_exceeded(self, sessions, schedule) -> None:
        for session in sessions:
            self.assertLessEqual(
                np.array(schedule[session.station_id]),
                self.max_rate_estimation[session.session_id],
            )


# Two Station Test Case
def two_station(
    limit: float,
    continuous: bool,
    session_max_rate: float,
    session_min_rate: float = 0,
    remaining_energy: Optional[List[float]] = None,
    estimated_departure: Optional[List[float]] = None,
) -> TestingInterface:
    """ Two EVSEs with the same phase, one constraint, and allowable rates from 0 to 32
    if continuous; integers between 8 and 32 if not. Also provides 2 sessions arriving
    and departing at the same time, with the same energy demands. """
    if continuous:
        allowable: List[np.ndarray] = [np.array([0, 32])] * 2
    else:
        allowable: List[np.ndarray] = [np.array([0] + list(range(8, 33)))] * 2
    if remaining_energy is None:
        remaining_energy: List[float] = [3.3, 3.3]
    network: InfrastructureDict = single_phase_single_constraint(
        2, limit, allowable_pilots=allowable, is_continuous=np.array([continuous] * 2)
    )
    sessions: List[SessionDict] = session_generator(
        num_sessions=2,
        arrivals=[0] * 2,
        departures=[11, 12],
        estimated_departures=estimated_departure,
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


def big_three_phase_network(
    num_sessions: int = 30, limit: float = 1000
) -> TestingInterface:
    """
    Network is WAY over designed to deal with unbalance and ensure all EVs can charge
    at their maximum rate.
    """
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
    def setUp(self) -> None:
        """ See BaseAlgorithmTest.setUp. """
        self.algo = None
        self.max_rate_estimation = {"0": 16, "1": 12}

    @staticmethod
    def _get_scenarios() -> List[Scenario]:
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
                    # The latter case below tests when the remaining amp periods is
                    # small enough to trigger a pilot signal going to 0 while there is
                    # still demand remaining.
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
                                    interface: TestingInterface = two_station(
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
    def setUp(self) -> None:
        """ See BaseAlgorithmTest.setUp. """
        self.algo = None
        self.max_rate_estimation = {}  # Don't use max_rate_estimation for this test.

    @staticmethod
    def _get_scenarios() -> List[Scenario]:
        scenarios = []
        for limit in [1500, 3200]:
            interface: TestingInterface = big_three_phase_network(limit=limit)
            scenario_name = f"capacity: {limit} "
            scenarios.append(Scenario(scenario_name, interface, False, False, False))
        return scenarios


# --------------------------------------------------------------------------------------
# Tests
#
# As the functionality tested in each class is evident from the class names,
# the setUp methods are left without docstrings (hence the noinspection
# comments).
# --------------------------------------------------------------------------------------
class TestTwoStationsMinRatesInfeasible(unittest.TestCase):
    """ Check that error is thrown when minimum rates are not feasible. """

    def test_sorted_min_rates_infeasible(self) -> None:
        limit = 16
        max_rate = 32
        min_rate = 16
        for continuous in [True, False]:
            interface: TestingInterface = two_station(
                limit, continuous, max_rate, min_rate
            )
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


class TestTwoStationsFCFS(TestTwoStationsBase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.algo = SortedSchedulingAlgo(first_come_first_served)


class TestThirtyStationsFCFS(TestThirtyStationsBase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.algo = SortedSchedulingAlgo(first_come_first_served)


class TestTwoStationsEDF(TestTwoStationsBase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.algo = SortedSchedulingAlgo(earliest_deadline_first)


class TestThirtyStationsEDF(TestThirtyStationsBase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.algo = SortedSchedulingAlgo(earliest_deadline_first)


class TestTwoStationsLLF(TestTwoStationsBase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.algo = SortedSchedulingAlgo(least_laxity_first)


class TestThirtyStationsLLF(TestThirtyStationsBase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.algo = SortedSchedulingAlgo(least_laxity_first)


class TestTwoStationsLCFS(TestTwoStationsBase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.algo = SortedSchedulingAlgo(last_come_first_served)


class TestThirtyStationsLCFS(TestThirtyStationsBase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.algo = SortedSchedulingAlgo(last_come_first_served)


class TestTwoStationsLRPT(TestTwoStationsBase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.algo = SortedSchedulingAlgo(largest_remaining_processing_time)


class TestThirtyStationsLRPT(TestThirtyStationsBase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.algo = SortedSchedulingAlgo(largest_remaining_processing_time)


class TestTwoStationsRR(TestTwoStationsBase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.algo = RoundRobin(first_come_first_served)


class TestThirtyStationsRR(TestThirtyStationsBase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.algo = RoundRobin(first_come_first_served)

class TestEarliestDeadlineFirstOrder(unittest.TestCase):
    def test_estimated_departure_matches_real(self):
        interface = two_station(limit=100, continuous=True, session_max_rate=32, estimated_departure=[11, 12])
        sorted_sessions = earliest_deadline_first(interface.active_sessions(), interface)
        self.assertEqual(sorted_sessions[0].session_id, "0")
        self.assertEqual(sorted_sessions[1].session_id, "1")


    def test_estimated_departure_does_not_match_real(self):
        interface = two_station(limit=100, continuous=True, session_max_rate=32, estimated_departure=[12, 11])
        sorted_sessions = earliest_deadline_first(interface.active_sessions(), interface)
        self.assertEqual(sorted_sessions[0].session_id, "1")
        self.assertEqual(sorted_sessions[1].session_id, "0")


class TestLeastLaxityFirstOrder(unittest.TestCase):
    def test_estimated_departure_matches_real(self):
        interface = two_station(limit=100, continuous=True, session_max_rate=32, estimated_departure=[11, 12])
        sorted_sessions = least_laxity_first(interface.active_sessions(), interface)
        self.assertEqual(sorted_sessions[0].session_id, "0")
        self.assertEqual(sorted_sessions[1].session_id, "1")


    def test_estimated_departure_does_not_match_real(self):
        interface = two_station(limit=100, continuous=True, session_max_rate=32, estimated_departure=[12, 11])
        sorted_sessions = least_laxity_first(interface.active_sessions(), interface)
        self.assertEqual(sorted_sessions[0].session_id, "1")
        self.assertEqual(sorted_sessions[1].session_id, "0")


del BaseAlgorithmTest
del TestTwoStationsBase
del TestThirtyStationsBase
# del TestTwoStationsMinRatesFeasibleBase
