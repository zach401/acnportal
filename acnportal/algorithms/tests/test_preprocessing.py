# coding=utf-8
""" Tests for Preprocessing algorithms. """
from unittest import TestCase
from unittest.mock import Mock
from numpy import testing as nptest
from acnportal.algorithms.preprocessing import *
from acnportal.algorithms.tests.generate_test_cases import *
from acnportal.acnsim.interface import SessionInfo, InfrastructureInfo

ARRIVAL_TIME = 0
SESSION_DUR = 5
ENERGY_DEMAND = 32 * 5
N = 3
CURRENT_TIME = 0
PERIOD = 5


class TestEnforcePilotLimit(TestCase):
    def test_pilot_greater_than_existing_max(
        self,
    ) -> None:  # pylint: disable=no-self-use
        sessions = session_generator(
            num_sessions=N,
            arrivals=[ARRIVAL_TIME] * N,
            departures=[ARRIVAL_TIME + SESSION_DUR] * N,
            requested_energy=[3.3] * N,
            remaining_energy=[3.3] * N,
            max_rates=[np.repeat(16, SESSION_DUR)] * N,
        )
        sessions = [SessionInfo(**s) for s in sessions]
        infrastructure = InfrastructureInfo(
            **single_phase_single_constraint(num_evses=N, limit=32)
        )
        modified_sessions = enforce_pilot_limit(sessions, infrastructure)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 16)

    def test_pilot_less_than_existing_max(self) -> None:  # pylint: disable=no-self-use
        sessions = session_generator(
            num_sessions=N,
            arrivals=[ARRIVAL_TIME] * N,
            departures=[ARRIVAL_TIME + SESSION_DUR] * N,
            requested_energy=[3.3] * N,
            remaining_energy=[3.3] * N,
            max_rates=[np.repeat(40, SESSION_DUR)] * N,
        )
        sessions = [SessionInfo(**s) for s in sessions]
        infrastructure = InfrastructureInfo(
            **single_phase_single_constraint(num_evses=N, limit=32)
        )
        modified_sessions = enforce_pilot_limit(sessions, infrastructure)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 32)


class TestReconcileMaxMin(TestCase):
    @staticmethod
    def _session_generation_helper(
        max_rates: int, choose_min: bool = True
    ) -> SessionInfo:
        session = SessionInfo(
            station_id="1",
            session_id="1",
            requested_energy=3.3,
            energy_delivered=0,
            arrival=0,
            departure=5,
            max_rates=max_rates,
            min_rates=8,
        )
        return reconcile_max_and_min(session, choose_min=choose_min)

    def test_max_greater_than_min(self) -> None:  # pylint: disable=no-self-use
        modified_session: SessionInfo = self._session_generation_helper(32)
        nptest.assert_almost_equal(modified_session.max_rates, 32)
        nptest.assert_almost_equal(modified_session.min_rates, 8)

    def test_max_equal_min(self) -> None:  # pylint: disable=no-self-use
        modified_session: SessionInfo = self._session_generation_helper(8)
        nptest.assert_almost_equal(modified_session.max_rates, 8)
        nptest.assert_almost_equal(modified_session.min_rates, 8)

    def test_max_less_than_min(self) -> None:  # pylint: disable=no-self-use
        modified_session: SessionInfo = self._session_generation_helper(6)
        nptest.assert_almost_equal(modified_session.max_rates, 8)
        nptest.assert_almost_equal(modified_session.min_rates, 8)

    def test_max_less_than_min_use_max(self) -> None:  # pylint: disable=no-self-use
        modified_session: SessionInfo = self._session_generation_helper(
            6, choose_min=False
        )
        nptest.assert_almost_equal(modified_session.max_rates, 6)
        nptest.assert_almost_equal(modified_session.min_rates, 6)


class TestApplyUpperBoundEstimate(TestCase):
    @staticmethod
    def _session_generation_helper(
        max_rate_list: List[Union[float, np.ndarray]],
        upper_bound_estimate: Dict[str, float],
        expected_max: float,
        expected_min: float,
        min_rate_list: Optional[List[Union[float, np.ndarray]]] = None,
    ) -> None:
        if min_rate_list is not None:
            min_rate_list *= N
        sessions: List[SessionDict] = session_generator(
            num_sessions=N,
            arrivals=[ARRIVAL_TIME] * N,
            departures=[ARRIVAL_TIME + SESSION_DUR] * N,
            requested_energy=[3.3] * N,
            remaining_energy=[3.3] * N,
            max_rates=max_rate_list * N,
            min_rates=min_rate_list,
        )
        sessions: List[SessionInfo] = [SessionInfo(**s) for s in sessions]
        rd = Mock()
        rd.get_maximum_rates = Mock(return_value=upper_bound_estimate)
        modified_sessions = apply_upper_bound_estimate(rd, sessions)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, expected_max)
            nptest.assert_almost_equal(session.min_rates, expected_min)

    def test_default_max_rates_scalars(self) -> None:  # pylint: disable=no-self-use
        self._session_generation_helper(
            max_rate_list=[32],
            upper_bound_estimate={f"{i}": 16 for i in range(N)},
            expected_max=16,
            expected_min=0,
        )

    def test_lower_existing_max_scalars(self) -> None:  # pylint: disable=no-self-use
        self._session_generation_helper(
            max_rate_list=[12],
            upper_bound_estimate={f"{i}": 16 for i in range(N)},
            expected_max=12,
            expected_min=0,
        )

    def test_vector_existing_max_scalar_rampdown(
        self,
    ) -> None:  # pylint: disable=no-self-use
        self._session_generation_helper(
            max_rate_list=[np.repeat(32, SESSION_DUR)],
            upper_bound_estimate={f"{i}": 16 for i in range(N)},
            expected_max=16,
            expected_min=0,
        )

    def test_vector_lower_existing_max_scalar_rampdown(
        self,
    ) -> None:  # pylint: disable=no-self-use
        self._session_generation_helper(
            max_rate_list=[np.repeat(12, SESSION_DUR)],
            upper_bound_estimate={f"{i}": 16 for i in range(N)},
            expected_max=12,
            expected_min=0,
        )

    def test_all_vectors_rampdown_lower(self) -> None:  # pylint: disable=no-self-use
        self._session_generation_helper(
            max_rate_list=[np.repeat(32, SESSION_DUR)],
            upper_bound_estimate={f"{i}": [16] * 5 for i in range(N)},
            expected_max=16,
            expected_min=0,
        )

    def test_all_vectors_existing_lower(self) -> None:  # pylint: disable=no-self-use
        self._session_generation_helper(
            max_rate_list=[np.repeat(12, SESSION_DUR)],
            upper_bound_estimate={f"{i}": [16] * 5 for i in range(N)},
            expected_max=12,
            expected_min=0,
        )

    def test_minimum_rates_binding(self) -> None:  # pylint: disable=no-self-use
        self._session_generation_helper(
            max_rate_list=[np.repeat(12, SESSION_DUR)],
            upper_bound_estimate={f"{i}": 6 for i in range(N)},
            expected_max=8,
            expected_min=8,
            min_rate_list=[np.repeat(8, SESSION_DUR)],
        )

    def test_partial_estimator(self) -> None:  # pylint: disable=no-self-use
        sessions: List[SessionDict] = session_generator(
            num_sessions=N,
            arrivals=[ARRIVAL_TIME] * N,
            departures=[ARRIVAL_TIME + SESSION_DUR] * N,
            requested_energy=[3.3] * N,
            remaining_energy=[3.3] * N,
            max_rates=[32] * N,
        )
        sessions: List[SessionInfo] = [SessionInfo(**s) for s in sessions]
        rd = Mock()
        rd.get_maximum_rates = Mock(
            return_value={f"{i}": [16] * 5 for i in range(N) if i != 1}
        )
        modified_sessions = apply_upper_bound_estimate(rd, sessions)
        for i, session in enumerate(modified_sessions):
            nptest.assert_almost_equal(session.max_rates, (16 if i != 1 else 32))
            nptest.assert_almost_equal(session.min_rates, 0)


class TestApplyMinimumChargingRate(TestCase):
    @staticmethod
    def _session_generation_helper(
        max_rate_list: List[Union[float, np.ndarray]],
        min_rate_list: Optional[List[Union[float, np.ndarray]]] = None,
        remaining_energy: float = 3.3,
    ) -> List[SessionInfo]:
        if min_rate_list is not None:
            min_rate_list *= N
        sessions: List[SessionDict] = session_generator(
            num_sessions=N,
            arrivals=[ARRIVAL_TIME] * N,
            departures=[ARRIVAL_TIME + SESSION_DUR] * N,
            requested_energy=[3.3] * N,
            remaining_energy=[remaining_energy] * N,
            max_rates=max_rate_list * N,
            min_rates=min_rate_list,
        )
        sessions: List[SessionInfo] = [SessionInfo(**s) for s in sessions]
        infrastructure = InfrastructureInfo(**single_phase_single_constraint(N, 32))
        modified_sessions = apply_minimum_charging_rate(
            sessions, infrastructure, PERIOD
        )
        return modified_sessions

    def test_evse_less_than_session_max(self) -> None:  # pylint: disable=no-self-use
        modified_sessions: List[SessionInfo] = self._session_generation_helper(
            max_rate_list=[np.repeat(32, SESSION_DUR)]
        )
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 32)
            nptest.assert_almost_equal(session.min_rates[0], 8)
            nptest.assert_almost_equal(session.min_rates[1:], 0)

    def test_evse_less_than_existing_min(self) -> None:  # pylint: disable=no-self-use
        modified_sessions: List[SessionInfo] = self._session_generation_helper(
            max_rate_list=[np.repeat(32, SESSION_DUR)],
            min_rate_list=[np.repeat(16, SESSION_DUR)],
        )
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 32)
            nptest.assert_almost_equal(session.min_rates, 16)

    def test_evse_min_greater_than_remaining_energy(
        self,
    ) -> None:  # pylint: disable=no-self-use
        modified_sessions: List[SessionInfo] = self._session_generation_helper(
            max_rate_list=[np.repeat(32, SESSION_DUR)], remaining_energy=0.05,
        )
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates[0], 0)
            nptest.assert_almost_equal(session.max_rates[1:], 32)
            nptest.assert_almost_equal(session.min_rates[0], 0)
            nptest.assert_almost_equal(session.min_rates[1:], 0)

    def test_evse_min_greater_than_session_max(
        self,
    ) -> None:  # pylint: disable=no-self-use
        modified_sessions: List[SessionInfo] = self._session_generation_helper(
            max_rate_list=[np.repeat(6, SESSION_DUR)],
        )
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates[0], 8)
            nptest.assert_almost_equal(session.max_rates[1:], 6)
            nptest.assert_almost_equal(session.min_rates[0], 8)
            nptest.assert_almost_equal(session.min_rates[1:], 0)

    def test_apply_min_infeasible(self) -> None:  # pylint: disable=no-self-use
        n = 3
        sessions = session_generator(
            num_sessions=n,
            arrivals=[1, 2, 3],
            departures=[1 + SESSION_DUR, 2 + SESSION_DUR, 3 + SESSION_DUR],
            requested_energy=[3.3] * n,
            remaining_energy=[3.3] * n,
            max_rates=[np.repeat(32, SESSION_DUR)] * n,
        )
        sessions = [SessionInfo(**s) for s in sessions]
        infrastructure = InfrastructureInfo(**single_phase_single_constraint(n, 16))
        modified_sessions = apply_minimum_charging_rate(
            sessions, infrastructure, PERIOD
        )
        for i in range(2):
            nptest.assert_almost_equal(modified_sessions[i].min_rates[0], 8)
            nptest.assert_almost_equal(modified_sessions[i].min_rates[1:], 0)
        # It is not feasible to deliver 8 A to session '2', so max and
        # min should be 0 at time t=0.
        nptest.assert_almost_equal(modified_sessions[2].min_rates, 0)
        nptest.assert_almost_equal(modified_sessions[2].max_rates[0], 0)


class TestRemoveFinishedSessions(TestCase):
    def test_remove_sessions_zero_remaining(self):
        N = 3
        sessions = session_generator(
            num_sessions=N,
            arrivals=[1, 2, 3],
            departures=[1 + SESSION_DUR, 2 + SESSION_DUR, 3 + SESSION_DUR],
            requested_energy=[3.3] * N,
            remaining_energy=[0, 3.3, 2],
            max_rates=[np.repeat(32, SESSION_DUR)] * N,
        )
        infrastructure = InfrastructureInfo(**three_phase_balanced_network(1, limit=100, min_pilot=8))
        sessions = [SessionInfo(**s) for s in sessions]
        modified_sessions = remove_finished_sessions(sessions, infrastructure, 5)
        self.assertEqual(len(modified_sessions), 2)


    def test_remove_sessions_remaining_within_threshold(self):
        N = 3
        sessions = session_generator(
            num_sessions=N,
            arrivals=[1, 2, 3],
            departures=[1 + SESSION_DUR, 2 + SESSION_DUR, 3 + SESSION_DUR],
            requested_energy=[3.3] * N,
            remaining_energy=[0.1, 3.3, 2],
            max_rates=[np.repeat(32, SESSION_DUR)] * N,
        )
        infrastructure = InfrastructureInfo(**three_phase_balanced_network(1, limit=100, min_pilot=8))
        sessions = [SessionInfo(**s) for s in sessions]
        modified_sessions = remove_finished_sessions(sessions, infrastructure, 5)
        self.assertEqual(len(modified_sessions), 2)