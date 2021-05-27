# coding=utf-8
""" Tests for objects of the interface module. """
from typing import List
from unittest import TestCase
from unittest.mock import create_autospec, patch, Mock
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
    def test_valid_inputs_w_defaults(self) -> None:
        s = SessionInfo("PS-001", "01", 10, 4, 5, 60)
        self.assertEqual(s.station_id, "PS-001")
        self.assertEqual(s.session_id, "01")
        self.assertEqual(s.requested_energy, 10)
        self.assertEqual(s.energy_delivered, 4)
        self.assertEqual(s.arrival, 5)
        self.assertEqual(s.departure, 60)
        self.assertEqual(s.estimated_departure, 60)
        self.assertEqual(s.current_time, 0)
        nptest.assert_array_equal(s.min_rates, 0)
        self.assertEqual(s.min_rates.shape, (55,))
        nptest.assert_array_equal(s.max_rates, float("inf"))
        self.assertEqual(s.max_rates.shape, (55,))
        self.assertEqual(s.remaining_demand, 6)
        self.assertEqual(s.arrival_offset, 5)
        self.assertEqual(s.remaining_time, 55)

    def test_kwarg_order(self) -> None:
        """ Tests that the order of overflow of kwargs for SessionInfo is maintained.
        """
        s = SessionInfo("PS-001", "01", 10, 4, 5, 60, 63, 6, [8.0] * 54, [32.0] * 54)
        self.assertEqual(s.estimated_departure, 63)
        self.assertEqual(s.current_time, 6)
        nptest.assert_array_equal(s.min_rates, 8.0)
        self.assertEqual(s.min_rates.shape, (54,))
        nptest.assert_array_equal(s.max_rates, 32.0)
        self.assertEqual(s.max_rates.shape, (54,))

    def test_valid_inputs_distinct_estimated_departure(self) -> None:
        s = SessionInfo("PS-001", "01", 10, 4, 5, 60, estimated_departure=63)
        self.assertEqual(s.departure, 60)
        self.assertEqual(s.estimated_departure, 63)

    def test_valid_inputs_nonzero_current_time_greater_than_arrival(self) -> None:
        s = SessionInfo(
            "PS-001", "01", 10, 4, 5, 60, estimated_departure=63, current_time=6
        )
        self.assertEqual(s.current_time, 6)
        self.assertEqual(s.arrival_offset, 0)
        self.assertEqual(s.remaining_time, 54)

    def test_valid_inputs_nonzero_current_time_less_than_arrival(self) -> None:
        s = SessionInfo(
            "PS-001", "01", 10, 4, 5, 60, estimated_departure=63, current_time=4
        )
        self.assertEqual(s.current_time, 4)
        self.assertEqual(s.arrival_offset, 1)
        self.assertEqual(s.remaining_time, 55)

    def test_proper_length_min_rates(self) -> None:
        s = SessionInfo(
            "PS-001", "01", 10, 4, 5, 10, estimated_departure=12, min_rates=[8.0] * 5
        )
        self.assertEqual(s.min_rates.shape, (5,))
        nptest.assert_array_equal(s.min_rates, 8.0)

    def test_min_rates_too_short(self) -> None:
        with self.assertRaises(ValueError):
            SessionInfo(
                "PS-001",
                "01",
                10,
                4,
                5,
                10,
                estimated_departure=12,
                min_rates=[8.0] * 4,
            )

    def test_min_rates_too_long(self) -> None:
        with self.assertRaises(ValueError):
            SessionInfo(
                "PS-001",
                "01",
                10,
                4,
                5,
                10,
                estimated_departure=12,
                min_rates=[8.0] * 6,
            )

    def test_proper_length_max_rates(self) -> None:
        s = SessionInfo(
            "PS-001", "01", 10, 4, 5, 10, estimated_departure=12, max_rates=[8.0] * 5
        )
        self.assertEqual(s.max_rates.shape, (5,))
        nptest.assert_array_equal(s.max_rates, 8.0)

    def test_max_rates_too_short(self) -> None:
        with self.assertRaises(ValueError):
            SessionInfo(
                "PS-001",
                "01",
                10,
                4,
                5,
                10,
                estimated_departure=12,
                max_rates=[8.0] * 4,
            )

    def test_max_rates_too_long(self) -> None:
        with self.assertRaises(ValueError):
            SessionInfo(
                "PS-001",
                "01",
                10,
                4,
                5,
                10,
                estimated_departure=12,
                max_rates=[8.0] * 6,
            )


class TestInfrastructureInfo(TestCase):
    def test_inputs_consistent(self) -> None:
        m, n = 6, 5
        # Here, the last two arguments are kwargs, left unspecified to test that the
        # order of overflow is unchanged.
        infra = InfrastructureInfo(
            np.ones((m, n)),
            np.ones((m,)),
            np.ones((n,)),
            np.ones((n,)),
            [f"C-{i}" for i in range(m)],
            [f"S-{i}" for i in range(n)],
            np.ones((n,)),
            np.zeros((n,)),
            [np.array([1, 2, 3, 4])] * n,  # allowable_pilots
            np.zeros((n,)),  # is_continuous
        )
        self.assertEqual(infra.constraint_matrix.shape, (m, n))
        self.assertEqual(infra.constraint_limits.shape, (m,))
        self.assertEqual(infra.phases.shape, (n,))
        self.assertEqual(infra.voltages.shape, (n,))
        self.assertEqual(len(infra.constraint_ids), m)
        self.assertEqual(len(infra.station_ids), n)
        self.assertEqual(infra.max_pilot.shape, (n,))
        self.assertEqual(infra.min_pilot.shape, (n,))
        self.assertEqual(len(infra.allowable_pilots), n)
        self.assertEqual(infra.is_continuous.shape, (n,))
        self.assertEqual(infra.is_continuous.dtype, "bool")

    def test_inputs_allowable_pilot_defaults(self) -> None:
        m, n = 6, 5
        infra = InfrastructureInfo(
            np.ones((m, n)),
            np.ones((m,)),
            np.ones((n,)),
            np.ones((n,)),
            [f"C-{i}" for i in range(m)],
            [f"S-{i}" for i in range(n)],
            np.ones((n,)),
            np.zeros((n,)),
            is_continuous=np.zeros((n,)),
        )
        self.assertEqual(len(infra.allowable_pilots), n)
        self.assertEqual(infra.allowable_pilots, [None] * n)

    def test_inputs_is_continuous_default(self) -> None:
        m, n = 6, 5
        infra = InfrastructureInfo(
            np.ones((m, n)),
            np.ones((m,)),
            np.ones((n,)),
            np.ones((n,)),
            [f"C-{i}" for i in range(m)],
            [f"S-{i}" for i in range(n)],
            np.ones((n,)),
            np.zeros((n,)),
            allowable_pilots=[np.array([1, 2, 3, 4])] * n,
        )
        nptest.assert_array_equal(infra.is_continuous, True)
        self.assertEqual(infra.is_continuous.shape, (n,))
        self.assertEqual(infra.is_continuous.dtype, "bool")

    def test_num_stations_mismatch(self) -> None:
        m, n = 5, 6
        for i in range(8):
            for error in [-1, 1]:
                errors = [0] * 8
                errors[i] = error
                with self.assertRaises(ValueError):
                    InfrastructureInfo(
                        np.ones((m, n + errors[0])),
                        np.ones((m,)),
                        np.ones((n + errors[1],)),
                        np.ones((n + errors[2],)),
                        [f"C-{i}" for i in range(m)],
                        [f"S-{i}" for i in range(n + errors[3])],
                        np.ones((n + errors[4],)),
                        np.zeros((n + errors[5],)),
                        allowable_pilots=[np.array([1, 2, 3, 4])] * (n + errors[6]),
                        is_continuous=np.zeros((n + errors[7],)),
                    )

    def test_num_constraints_mismatch(self) -> None:
        m, n = 5, 6
        for i in range(3):
            for error in [-1, 1]:
                errors = [0] * 3
                errors[i] = error
                with self.assertRaises(ValueError):
                    InfrastructureInfo(
                        np.ones((m + errors[0], n)),
                        np.ones((m + errors[1],)),
                        np.ones((n,)),
                        np.ones((n,)),
                        [f"C-{i}" for i in range(m + errors[2])],
                        [f"S-{i}" for i in range(n)],
                        np.ones((n,)),
                        np.zeros((n,)),
                        allowable_pilots=[np.array([1, 2, 3, 4])] * n,
                        is_continuous=np.zeros((n,)),
                    )

    def test_num_stations_num_constraints_mismatch(self) -> None:
        m, n = 5, 6
        with self.assertRaises(ValueError):
            InfrastructureInfo(
                np.ones((m + 1, n)),
                np.ones((m,)),
                np.ones((n,)),
                np.ones((n,)),
                [f"C-{i}" for i in range(m - 1)],
                [f"S-{i}" for i in range(n)],
                np.ones((n,)),
                np.zeros((n + 1,)),
                allowable_pilots=[np.array([1, 2, 3, 4])] * n,
                is_continuous=np.zeros((n - 1,)),
            )


class TestInterface(TestCase):
    def setUp(self) -> None:
        """ Run this setup function once before each test. """
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

    def test_init(self) -> None:
        self.assertIs(self.interface._simulator, self.simulator)

    def test_violation_tolerance(self) -> None:
        self.assertEqual(
            self.interface._violation_tolerance,
            self.interface._simulator.network.violation_tolerance,
        )

    def test_relative_tolerance(self) -> None:
        self.assertEqual(
            self.interface._relative_tolerance,
            self.interface._simulator.network.relative_tolerance,
        )

    def test_active_evs(self) -> None:
        with self.assertWarns(UserWarning):
            _ = self.interface.active_evs
        self.simulator.get_active_evs.assert_called_once()

    def test_last_applied_pilot_signals_low_iteration(self) -> None:
        self.simulator.iteration = 1
        self.assertEqual(self.interface.last_applied_pilot_signals, {})

    def test_allowable_pilot_signals(self) -> None:
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

    def test_evse_voltage(self) -> None:
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

    def test_is_feasible_empty_schedule(self) -> None:
        self.assertTrue(self.interface.is_feasible({}))

    def test_is_feasible_unequal_schedules(self) -> None:
        with self.assertRaises(InvalidScheduleError):
            self.interface.is_feasible(
                {"PS-001": [1, 2], "PS-002": [3, 4, 5], "PS-003": [4, 5]}
            )

    def test_is_feasible(self) -> None:
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
        self.assertEqual(network_is_feasible_args[0][2], 1e-5)
        self.assertEqual(network_is_feasible_args[0][3], 1e-7)

    def test_is_feasible_with_options(self) -> None:
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
