from unittest import TestCase
from unittest.mock import Mock
from numpy import testing as nptest
from algorithms.preprocessing import *
from algorithms.tests.generate_test_cases import *
from acnportal.acnsim.interface import SessionInfo, InfrastructureInfo

ARRIVAL_TIME = 0
SESSION_DUR = 5
ENERGY_DEMAND = 32*5
N = 3


class TestEnforcePilotLimit(TestCase):
    def test_pilot_greater_than_existing_max(self):
        sessions = session_generator(num_sessions=N,
                                     arrivals=[ARRIVAL_TIME] * N,
                                     departures=[ARRIVAL_TIME + SESSION_DUR]*N,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[3.3] * N,
                                     max_rates=[np.repeat(16, SESSION_DUR)]*N)
        sessions = [SessionInfo(**s) for s in sessions]
        infrastructure = InfrastructureInfo(
            **single_phase_single_constraint(num_evses=N,
                                             limit=32,
                                             max_pilot=32))
        modified_sessions = enforce_pilot_limit(sessions, infrastructure)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 16)

    def test_pilot_less_than_existing_max(self):
        sessions = session_generator(num_sessions=N,
                                     arrivals=[ARRIVAL_TIME] * N,
                                     departures=[ARRIVAL_TIME + SESSION_DUR]*N,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[3.3] * N,
                                     max_rates=[np.repeat(40, SESSION_DUR)]*N)
        sessions = [SessionInfo(**s) for s in sessions]
        infrastructure = InfrastructureInfo(
            **single_phase_single_constraint(num_evses=N,
                                             limit=32,
                                             max_pilot=32))
        modified_sessions = enforce_pilot_limit(sessions, infrastructure)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 32)


class TestReconcileMaxMin(TestCase):
    def test_max_greater_than_min(self):
        session = SessionInfo(station_id='1',
                              session_id='1',
                              requested_energy=3.3,
                              energy_delivered=0,
                              arrival=0,
                              departure=5,
                              max_rates=32,
                              min_rates=8)
        modified_session = reconcile_max_and_min(session)
        nptest.assert_almost_equal(modified_session.max_rates, 32)
        nptest.assert_almost_equal(modified_session.min_rates, 8)

    def test_max_equal_min(self):
        session = SessionInfo(station_id='1',
                              session_id='1',
                              requested_energy=3.3,
                              energy_delivered=0,
                              arrival=0,
                              departure=5,
                              max_rates=8,
                              min_rates=8)
        modified_session = reconcile_max_and_min(session)
        nptest.assert_almost_equal(modified_session.max_rates, 8)
        nptest.assert_almost_equal(modified_session.min_rates, 8)

    def test_max_less_than_min(self):
        session = SessionInfo(station_id='1',
                              session_id='1',
                              requested_energy=3.3,
                              energy_delivered=0,
                              arrival=0,
                              departure=5,
                              max_rates=6,
                              min_rates=8)
        modified_session = reconcile_max_and_min(session)
        nptest.assert_almost_equal(modified_session.max_rates, 8)
        nptest.assert_almost_equal(modified_session.min_rates, 8)


class TestApplyUpperBoundEstimate(TestCase):
    def test_default_max_rates_scalars(self):
        sessions = session_generator(num_sessions = N,
                                     arrivals=[ARRIVAL_TIME] * N,
                                     departures=[ARRIVAL_TIME + SESSION_DUR]*N,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[3.3] * N,
                                     max_rates=[32] * N)
        sessions = [SessionInfo(**s) for s in sessions]
        rd = Mock()
        rd.get_maximum_rates = Mock(return_value={f'{i}': 16 for i in range(N)})
        modified_sessions = apply_upper_bound_estimate(rd, sessions)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 16)
            nptest.assert_almost_equal(session.min_rates, 0)

    def test_lower_existing_max_scalars(self):
        sessions = session_generator(num_sessions=N,
                                     arrivals=[ARRIVAL_TIME] * N,
                                     departures=[ARRIVAL_TIME + SESSION_DUR]*N,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[3.3] * N,
                                     max_rates=[12] * N)
        sessions = [SessionInfo(**s) for s in sessions]
        rd = Mock()
        rd.get_maximum_rates = Mock(return_value={f'{i}': 16 for i in range(N)})
        modified_sessions = apply_upper_bound_estimate(rd, sessions)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 12)
            nptest.assert_almost_equal(session.min_rates, 0)

    def test_vector_existing_max_scalar_rampdown(self):
        sessions = session_generator(num_sessions=N,
                                     arrivals=[ARRIVAL_TIME] * N,
                                     departures=[ARRIVAL_TIME + SESSION_DUR]*N,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[3.3] * N,
                                     max_rates=[np.repeat(32, SESSION_DUR)] * N)
        sessions = [SessionInfo(**s) for s in sessions]
        rd = Mock()
        rd.get_maximum_rates = Mock(return_value={f'{i}': 16 for i in range(N)})
        modified_sessions = apply_upper_bound_estimate(rd, sessions)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 16)
            nptest.assert_almost_equal(session.min_rates, 0)

    def test_vector_lower_existing_max_scalar_rampdown(self):
        sessions = session_generator(num_sessions=N,
                                     arrivals=[ARRIVAL_TIME] * N,
                                     departures=[ARRIVAL_TIME + SESSION_DUR]*N,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[3.3] * N,
                                     max_rates=[np.repeat(12, SESSION_DUR)] * N)
        sessions = [SessionInfo(**s) for s in sessions]
        rd = Mock()
        rd.get_maximum_rates = Mock(return_value={f'{i}': 16 for i in range(N)})
        modified_sessions = apply_upper_bound_estimate(rd, sessions)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 12)
            nptest.assert_almost_equal(session.min_rates, 0)

    def test_all_vectors_rampdown_lower(self):
        sessions = session_generator(num_sessions=N,
                                     arrivals=[ARRIVAL_TIME] * N,
                                     departures=[ARRIVAL_TIME + SESSION_DUR]*N,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[3.3] * N,
                                     max_rates=[np.repeat(32, SESSION_DUR)] * N)
        sessions = [SessionInfo(**s) for s in sessions]
        rd = Mock()
        rd.get_maximum_rates = Mock(
            return_value={f'{i}': [16]*5 for i in range(N)})
        modified_sessions = apply_upper_bound_estimate(rd, sessions)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 16)
            nptest.assert_almost_equal(session.min_rates, 0)

    def test_all_vectors_existing_lower(self):
        sessions = session_generator(num_sessions=N,
                                     arrivals=[ARRIVAL_TIME] * N,
                                     departures=[ARRIVAL_TIME + SESSION_DUR]*N,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[3.3] * N,
                                     max_rates=[np.repeat(12, SESSION_DUR)] * N)
        sessions = [SessionInfo(**s) for s in sessions]
        rd = Mock()
        rd.get_maximum_rates = Mock(
            return_value={f'{i}': [16]*5 for i in range(N)})
        modified_sessions = apply_upper_bound_estimate(rd, sessions)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 12)
            nptest.assert_almost_equal(session.min_rates, 0)

    def test_minimum_rates_binding(self):
        sessions = session_generator(num_sessions=N,
                                     arrivals=[ARRIVAL_TIME] * N,
                                     departures=[ARRIVAL_TIME + SESSION_DUR]*N,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[3.3] * N,
                                     max_rates=[np.repeat(12, SESSION_DUR)] * N,
                                     min_rates=[np.repeat(8, SESSION_DUR)] * N)
        sessions = [SessionInfo(**s) for s in sessions]
        rd = Mock()
        rd.get_maximum_rates = Mock(return_value={f'{i}': 6 for i in range(N)})
        modified_sessions = apply_upper_bound_estimate(rd, sessions)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 8)
            nptest.assert_almost_equal(session.min_rates, 8)


class TestApplyMinimumChargingRate(TestCase):
    def test_evse_less_than_session_max(self):
        sessions = session_generator(num_sessions=N,
                                     arrivals=[ARRIVAL_TIME] * N,
                                     departures=[ARRIVAL_TIME + SESSION_DUR]*N,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[3.3] * N,
                                     max_rates=[np.repeat(32, SESSION_DUR)]*N)
        sessions = [SessionInfo(**s) for s in sessions]
        infrastructure = InfrastructureInfo(
            **single_phase_single_constraint(N, 32, 32, 8))
        modified_sessions = apply_minimum_charging_rate(sessions,
                                                        infrastructure, 5)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 32)
            nptest.assert_almost_equal(session.min_rates[0], 8)
            nptest.assert_almost_equal(session.min_rates[1:], 0)

    def test_evse_less_than_existing_min(self):
        sessions = session_generator(num_sessions=N,
                                     arrivals=[ARRIVAL_TIME] * N,
                                     departures=[ARRIVAL_TIME + SESSION_DUR]*N,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[3.3] * N,
                                     max_rates=[np.repeat(32, SESSION_DUR)]*N,
                                     min_rates=[np.repeat(16, SESSION_DUR)]*N)
        sessions = [SessionInfo(**s) for s in sessions]
        infrastructure = InfrastructureInfo(
            **single_phase_single_constraint(N, 32, 32, 8))
        modified_sessions = apply_minimum_charging_rate(sessions,
                                                        infrastructure, 5)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates, 32)
            nptest.assert_almost_equal(session.min_rates, 16)

    def test_evse_min_greater_than_remaining_energy(self):
        sessions = session_generator(num_sessions=N,
                                     arrivals=[ARRIVAL_TIME] * N,
                                     departures=[ARRIVAL_TIME + SESSION_DUR]*N,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[0.05] * N,
                                     max_rates=[np.repeat(32, SESSION_DUR)]*N)
        sessions = [SessionInfo(**s) for s in sessions]
        infrastructure = InfrastructureInfo(
            **single_phase_single_constraint(N, 32, 32, 8))
        modified_sessions = apply_minimum_charging_rate(sessions,
                                                        infrastructure, 5)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates[0], 0)
            nptest.assert_almost_equal(session.max_rates[1:], 32)
            nptest.assert_almost_equal(session.min_rates[0], 0)
            nptest.assert_almost_equal(session.min_rates[1:], 0)

    def test_evse_min_greater_than_session_max(self):
        sessions = session_generator(num_sessions=N,
                                     arrivals=[ARRIVAL_TIME] * N,
                                     departures=[ARRIVAL_TIME + SESSION_DUR]*N,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[3.3] * N,
                                     max_rates=[np.repeat(6, SESSION_DUR)]*N)
        sessions = [SessionInfo(**s) for s in sessions]
        infrastructure = InfrastructureInfo(
            **single_phase_single_constraint(N, 32, 32, 8))
        modified_sessions = apply_minimum_charging_rate(sessions,
                                                        infrastructure, 5)
        for session in modified_sessions:
            nptest.assert_almost_equal(session.max_rates[0], 8)
            nptest.assert_almost_equal(session.max_rates[1:], 6)
            nptest.assert_almost_equal(session.min_rates[0], 8)
            nptest.assert_almost_equal(session.min_rates[1:], 0)

    def test_apply_min_infeasible(self):
        N = 3
        sessions = session_generator(num_sessions=N,
                                     arrivals=[1, 2, 3],
                                     departures=[1 + SESSION_DUR,
                                                 2 + SESSION_DUR,
                                                 3 + SESSION_DUR],
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[3.3] * N,
                                     max_rates=[np.repeat(32, SESSION_DUR)]*N)
        sessions = [SessionInfo(**s) for s in sessions]
        infrastructure = InfrastructureInfo(
            **single_phase_single_constraint(N, 16, 32, 8))
        modified_sessions = apply_minimum_charging_rate(sessions,
                                                        infrastructure, 5)
        for i in range(2):
            nptest.assert_almost_equal(modified_sessions[i].min_rates[0], 8)
            nptest.assert_almost_equal(modified_sessions[i].min_rates[1:], 0)
        # It is not feasible to deliver 8 A to session '2', so max and
        # min should be 0 at time t=0.
        nptest.assert_almost_equal(modified_sessions[2].min_rates, 0)
        nptest.assert_almost_equal(modified_sessions[2].max_rates[0], 0)


class TestIncRemainingEnergyToMinAllowable(TestCase):
    def test_energy_unchanged_when_above_min(self):
        N = 3
        period = 5
        min_energy = 12 * 208 / 1000 / 12
        sessions = session_generator(num_sessions=N,
                                     arrivals=[0] * N,
                                     departures=[3] * 3,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[min_energy]* N,
                                     max_rates=[np.repeat(32, SESSION_DUR)]
                                               * N,
                                     min_rates=[np.repeat(12, SESSION_DUR)]
                                               * N)
        sessions = [SessionInfo(**s) for s in sessions]
        infrastructure = InfrastructureInfo(
            **single_phase_single_constraint(N, 64, 32, 8))
        modified_sessions = inc_remaining_energy_to_min_allowable(sessions,
                                                                  infrastructure,
                                                                  period)
        for session in modified_sessions:
            self.assertAlmostEqual(session.remaining_demand, min_energy)

    def test_energy_unchanged_at_min(self):
        N = 3
        period = 5
        min_energy = 12 * 208 / 1000 / 12
        sessions = session_generator(num_sessions=N,
                                     arrivals=[0] * N,
                                     departures=[3] * 3,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[min_energy]* N,
                                     max_rates=[np.repeat(32, SESSION_DUR)]
                                               * N,
                                     min_rates=[np.repeat(12, SESSION_DUR)]
                                               * N)
        sessions = [SessionInfo(**s) for s in sessions]
        infrastructure = InfrastructureInfo(
            **single_phase_single_constraint(N, 64, 32, 8))
        modified_sessions = inc_remaining_energy_to_min_allowable(sessions,
                                                                  infrastructure,
                                                                  period)
        for session in modified_sessions:
            self.assertAlmostEqual(session.remaining_demand, min_energy)

    def test_energy_increased_below_min(self):
        N = 3
        period = 5
        min_energy = 12 * 208 / 1000 / 12
        sessions = session_generator(num_sessions=N,
                                     arrivals=[0] * N,
                                     departures=[3] * 3,
                                     requested_energy=[3.3] * N,
                                     remaining_energy=[min_energy - 0.1]
                                                      * N,
                                     max_rates=[np.repeat(32,
                                                          SESSION_DUR)] * N,
                                     min_rates=[np.repeat(12,
                                                          SESSION_DUR)] * N)
        sessions = [SessionInfo(**s) for s in sessions]
        infrastructure = InfrastructureInfo(
            **single_phase_single_constraint(N, 64, 32, 8))
        modified_sessions = inc_remaining_energy_to_min_allowable(sessions,
                                                                  infrastructure,
                                                                  period)
        for session in modified_sessions:
            self.assertGreaterEqual(session.remaining_demand, min_energy)
