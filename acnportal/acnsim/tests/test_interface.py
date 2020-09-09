from typing import List
from unittest import TestCase
from unittest.mock import create_autospec
import numpy.testing as nptest
import numpy as np

from acnportal.acnsim import (
    Simulator,
    Interface,
    InvalidScheduleError,
    FiniteRatesEVSE,
    DeadbandEVSE,
)
from acnportal.acnsim.interface import SessionInfo, InfrastructureInfo, Constraint
from acnportal.acnsim.models import EVSE
from acnportal.acnsim.network import ChargingNetwork


class TestSessionInfo(TestCase):
    def test_valid_inputs_w_defaults(self):
        s = SessionInfo("PS-001", "01", 10, 4, 5, 60, 63)
        self.assertEqual(s.station_id, "PS-001")
        self.assertEqual(s.session_id, "01")
        self.assertEqual(s.requested_energy, 10)
        self.assertEqual(s.energy_delivered, 4)
        self.assertEqual(s.arrival, 5)
        self.assertEqual(s.departure, 60)
        self.assertEqual(s.estimated_departure, 63)
        self.assertEqual(s.current_time, 0)
        nptest.assert_array_equal(s.min_rates, 0)
        nptest.assert_array_equal(s.max_rates, float("inf"))
        self.assertEqual(s.remaining_demand, 6)
        self.assertEqual(s.arrival_offset, 5)
        self.assertEqual(s.remaining_time, 55)

    def test_valid_inputs_nonzero_current_time_greater_than_arrival(self):
        s = SessionInfo("PS-001", "01", 10, 4, 5, 60, 63, 6)
        self.assertEqual(s.current_time, 6)
        self.assertEqual(s.arrival_offset, 0)
        self.assertEqual(s.remaining_time, 54)

    def test_valid_inputs_nonzero_current_time_less_than_arrival(self):
        s = SessionInfo("PS-001", "01", 10, 4, 5, 60, 63, 4)
        self.assertEqual(s.current_time, 4)
        self.assertEqual(s.arrival_offset, 1)
        self.assertEqual(s.remaining_time, 55)

    def test_proper_length_min_rates(self):
        s = SessionInfo(
            "PS-001", "01", 10, 4, 5, 10, 12, current_time=0, min_rates=[8.0] * 5
        )
        self.assertEqual(len(s.min_rates), 5)
        nptest.assert_array_equal(s.min_rates, 8.0)

    def test_min_rates_too_short(self):
        with self.assertRaises(ValueError):
            SessionInfo(
                "PS-001", "01", 10, 4, 5, 10, 12, current_time=0, min_rates=[8.0] * 4
            )

    def test_min_rates_too_long(self):
        with self.assertRaises(ValueError):
            SessionInfo(
                "PS-001", "01", 10, 4, 5, 10, 12, current_time=0, min_rates=[8.0] * 6
            )

    def test_proper_length_max_rates(self):
        s = SessionInfo(
            "PS-001", "01", 10, 4, 5, 10, 12, current_time=0, max_rates=[8.0] * 5
        )
        self.assertEqual(len(s.max_rates), 5)
        nptest.assert_array_equal(s.max_rates, 8.0)

    def test_max_rates_too_short(self):
        with self.assertRaises(ValueError):
            SessionInfo(
                "PS-001", "01", 10, 4, 5, 10, 12, current_time=0, max_rates=[8.0] * 4
            )

    def test_max_rates_too_long(self):
        with self.assertRaises(ValueError):
            SessionInfo(
                "PS-001", "01", 10, 4, 5, 10, 12, current_time=0, max_rates=[8.0] * 6
            )


class TestInfrastructureInfo(TestCase):
    def test_inputs_consistent(self):
        M, N = 6, 5
        infra = InfrastructureInfo(
            np.ones((M, N)),
            np.ones((M,)),
            np.ones((N,)),
            np.ones((N,)),
            [f"C-{i}" for i in range(M)],
            [f"S-{i}" for i in range(N)],
            np.ones((N,)),
            np.zeros((N,)),
            [np.array([1, 2, 3, 4])] * N,
            np.zeros((N,)),
        )
        self.assertEqual(infra.constraint_matrix.shape, (M, N))
        self.assertEqual(len(infra.constraint_limits), M)
        self.assertEqual(len(infra.phases), N)
        self.assertEqual(len(infra.voltages), N)
        self.assertEqual(len(infra.constraint_ids), M)
        self.assertEqual(len(infra.station_ids), N)
        self.assertEqual(len(infra.max_pilot), N)
        self.assertEqual(len(infra.min_pilot), N)
        self.assertEqual(len(infra.allowable_pilots), N)
        self.assertEqual(len(infra.is_continuous), N)

    def test_inputs_allowable_pilot_defaults(self):
        M, N = 6, 5
        infra = InfrastructureInfo(
            np.ones((M, N)),
            np.ones((M,)),
            np.ones((N,)),
            np.ones((N,)),
            [f"C-{i}" for i in range(M)],
            [f"S-{i}" for i in range(N)],
            np.ones((N,)),
            np.zeros((N,)),
            is_continuous=np.zeros((N,)),
        )
        self.assertEqual(len(infra.allowable_pilots), N)

    def test_inputs_is_continuous_default(self):
        M, N = 6, 5
        infra = InfrastructureInfo(
            np.ones((M, N)),
            np.ones((M,)),
            np.ones((N,)),
            np.ones((N,)),
            [f"C-{i}" for i in range(M)],
            [f"S-{i}" for i in range(N)],
            np.ones((N,)),
            np.zeros((N,)),
            [np.array([1, 2, 3, 4])] * N,
        )
        self.assertEqual(len(infra.is_continuous), N)

    def test_num_stations_mismatch(self):
        M, N = 5, 6
        for i in range(8):
            for error in [-1, 1]:
                errors = [0] * 8
                errors[i] = error
                with self.assertRaises(ValueError):
                    InfrastructureInfo(
                        np.ones((M, N + errors[0])),
                        np.ones((M,)),
                        np.ones((N + errors[1],)),
                        np.ones((N + errors[2],)),
                        [f"C-{i}" for i in range(M)],
                        [f"S-{i}" for i in range(N + errors[3])],
                        np.ones((N + errors[4],)),
                        np.zeros((N + errors[5],)),
                        [np.array([1, 2, 3, 4])] * (N + errors[6]),
                        np.zeros((N + errors[7],)),
                    )

    def test_num_constraints_mismatch(self):
        M, N = 5, 6
        for i in range(3):
            for error in [-1, 1]:
                errors = [0] * 3
                errors[i] = error
                with self.assertRaises(ValueError):
                    InfrastructureInfo(
                        np.ones((M + errors[0], N)),
                        np.ones((M + errors[1],)),
                        np.ones((N,)),
                        np.ones((N,)),
                        [f"C-{i}" for i in range(M + errors[2])],
                        [f"S-{i}" for i in range(N)],
                        np.ones((N,)),
                        np.zeros((N,)),
                        [np.array([1, 2, 3, 4])] * (N),
                        np.zeros((N,)),
                    )


class TestInterface(TestCase):
    def setUp(self):
        self.simulator = create_autospec(Simulator)
        self.network = ChargingNetwork()
        self.simulator.network = self.network
        self.interface = Interface(self.simulator)
        evse1 = EVSE("PS-001")
        self.network.register_evse(evse1, 120, -30)
        evse2 = EVSE("PS-002")
        evse3 = DeadbandEVSE("PS-003")
        self.network.register_evse(evse3, 360, 150)
        self.network.register_evse(evse2, 240, 90)
        # Include a FiniteRatesEVSE for more thorough testing.
        self.allowable_rates: List[int] = [0, 8, 16, 24, 32]
        evse4: FiniteRatesEVSE = FiniteRatesEVSE("PS-004", self.allowable_rates)
        self.network.register_evse(evse4, 120, -30)
        self.network.constraint_matrix = np.eye(4)
        self.network.magnitudes = np.ones((4, 1))
        self.network.constraint_index = ["C1", "C2", "C3", "C4"]

    def test_init(self):
        self.assertIs(self.interface._simulator, self.simulator)

    def test_active_evs(self):
        with self.assertWarns(UserWarning):
            _ = self.interface.active_evs
        self.simulator.get_active_evs.assert_called_once()

    def test_last_applied_pilot_signals_low_iteration(self):
        self.simulator.iteration = 1
        self.assertEqual(self.interface.last_applied_pilot_signals, {})

    def test_allowable_pilot_signals(self):
        self.assertEqual(
            self.interface.allowable_pilot_signals("PS-001"), (True, [0, float("inf")])
        )

    def test_allowable_pilot_signals_deadband(self) -> None:
        self.assertEqual(
            self.interface.allowable_pilot_signals("PS-003"), (True, [6, float("inf")])
        )

    def test_allowable_pilot_signals_finite_rates(self) -> None:
        self.assertEqual(
            self.interface.allowable_pilot_signals("PS-004"),
            (False, self.allowable_rates),
        )

    def test_max_pilot_signal(self) -> None:
        self.assertEqual(self.interface.max_pilot_signal("PS-001"), float("inf"))

    def test_max_pilot_signal_deadband(self) -> None:
        self.assertEqual(self.interface.max_pilot_signal("PS-003"), float("inf"))

    def test_max_pilot_signal_finite_rates(self) -> None:
        self.assertEqual(self.interface.max_pilot_signal("PS-004"), 32)

    def test_min_pilot_signal(self) -> None:
        self.assertEqual(self.interface.min_pilot_signal("PS-001"), 0)

    def test_min_pilot_signal_deadband(self) -> None:
        self.assertEqual(self.interface.min_pilot_signal("PS-003"), 0)

    def test_min_pilot_signal_finite_rates(self) -> None:
        self.assertEqual(self.interface.min_pilot_signal("PS-004"), 8)

    def test_evse_voltage(self):
        self.assertEqual(self.interface.evse_voltage("PS-002"), 240)

    def test_evse_phase(self) -> None:
        self.assertEqual(self.interface.evse_phase("PS-002"), 90)

    def test_get_constraints(self) -> None:
        constraint_info: Constraint = self.interface.get_constraints()
        nptest.assert_equal(
            constraint_info.constraint_matrix, self.network.constraint_matrix
        )
        nptest.assert_equal(constraint_info.magnitudes, self.network.magnitudes)
        self.assertEqual(
            constraint_info.constraint_index, self.network.constraint_index
        )
        self.assertEqual(constraint_info.evse_index, self.network.station_ids)

    def test_is_feasible_empty_schedule(self):
        self.assertTrue(self.interface.is_feasible({}))

    def test_is_feasible_unequal_schedules(self):
        with self.assertRaises(InvalidScheduleError):
            self.interface.is_feasible(
                {"PS-001": [1, 2], "PS-002": [3, 4, 5], "PS-003": [4, 5]}
            )

    def test_is_feasible(self):
        # Mock network's is_feasible function to check its call signature later
        self.network.is_feasible = create_autospec(self.network.is_feasible)
        self.interface.is_feasible({"PS-001": [1, 2], "PS-002": [4, 5]})
        network_is_feasible_args = self.network.is_feasible.call_args
        # Check that the call to the network's is_feasible method has the correct
        # arguments.
        np.testing.assert_allclose(
            network_is_feasible_args[0][0], np.array([[1, 2], [0, 0], [4, 5], [0, 0]])
        )
        # Network's is_feasible method has its second argument (linear) defaulting to
        # False. Check this is the case.
        self.assertEqual(network_is_feasible_args[0][1], False)
        # Network's is_feasible method has its third argument (violation_tolerance)
        # defaulting to None. Check this is the case.
        self.assertIsNone(network_is_feasible_args[0][2])
        self.assertIsNone(network_is_feasible_args[0][3])

    def test_is_feasible_with_options(self):
        # Mock network's is_feasible function to check its call signature later
        self.network.is_feasible = create_autospec(self.network.is_feasible)
        self.interface.is_feasible(
            {"PS-001": [1, 2], "PS-002": [4, 5]},
            linear=True,
            violation_tolerance=1e-3,
            relative_tolerance=1e-5,
        )
        network_is_feasible_args = self.network.is_feasible.call_args
        # Check that the call to the network's is_feasible method has the correct
        # arguments.
        np.testing.assert_allclose(
            network_is_feasible_args[0][0], np.array([[1, 2], [0, 0], [4, 5], [0, 0]])
        )
        self.assertEqual(network_is_feasible_args[0][1], True)
        self.assertEqual(network_is_feasible_args[0][2], 1e-3)
        self.assertEqual(network_is_feasible_args[0][3], 1e-5)
