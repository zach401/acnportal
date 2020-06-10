from unittest import TestCase
from unittest.mock import create_autospec

import numpy as np

from acnportal.acnsim import Simulator, Interface, InvalidScheduleError
from acnportal.acnsim.models import EVSE
from acnportal.acnsim.network import ChargingNetwork


class TestInterface(TestCase):
    def setUp(self):
        self.simulator = create_autospec(Simulator)
        self.network = ChargingNetwork()
        self.simulator.network = self.network
        self.interface = Interface(self.simulator)
        evse1 = EVSE("PS-001")
        self.network.register_evse(evse1, 120, -30)
        evse2 = EVSE("PS-002")
        evse3 = EVSE("PS-003")
        self.network.register_evse(evse3, 360, 150)
        self.network.register_evse(evse2, 240, 90)

    def test_init(self):
        self.assertIs(self.interface._simulator, self.simulator)

    def test_active_evs(self):
        _ = self.interface.active_evs
        self.simulator.get_active_evs.assert_called_once()

    def test_last_applied_pilot_signals_low_iteration(self):
        self.simulator.iteration = 1
        self.assertEqual(self.interface.last_applied_pilot_signals, {})

    def test_allowable_pilot_signals(self):
        self.assertEqual(
            self.interface.allowable_pilot_signals("PS-001"), (True, [0, float("inf")])
        )

    def test_evse_voltage(self):
        self.assertEqual(self.interface.evse_voltage("PS-002"), 240)

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
        # Check that the call to the network's is_feasible method has the correct arguments
        np.testing.assert_allclose(
            network_is_feasible_args[0][0], np.array([[1, 2], [0, 0], [4, 5]])
        )
        # Network's is_feasible method has its second argument (linear) defaulting to False. Check this is the case.
        self.assertEqual(network_is_feasible_args[0][1], False)
        # Network's is_feasible method has its third argument (violation_tolerance) defaulting to None. Check this is the case.
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
        # Check that the call to the network's is_feasible method has the correct arguments
        np.testing.assert_allclose(
            network_is_feasible_args[0][0], np.array([[1, 2], [0, 0], [4, 5]])
        )
        self.assertEqual(network_is_feasible_args[0][1], True)
        self.assertEqual(network_is_feasible_args[0][2], 1e-3)
        self.assertEqual(network_is_feasible_args[0][3], 1e-5)
