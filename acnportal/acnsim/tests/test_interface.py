from unittest import TestCase
from unittest.mock import Mock, create_autospec, patch

from acnportal.acnsim import Simulator, Interface, InvalidScheduleError
from acnportal.acnsim.network import ChargingNetwork
from acnportal.acnsim.models import EV, EVSE

import numpy as np

class TestInterface(TestCase):
    def setUp(self):
        self.simulator = create_autospec(Simulator)
        self.network = ChargingNetwork()
        self.simulator.network = self.network
        self.interface = Interface(self.simulator)
        evse1 = EVSE('PS-001')
        self.network.register_evse(evse1, 120, -30)
        evse2 = EVSE('PS-002')
        evse3 = EVSE('PS-003')
        self.network.register_evse(evse3, 360, 150)
        self.network.register_evse(evse2, 240, 90)

    def test_init(self):
        self.assertIs(self.interface._simulator, self.simulator)

    def test_active_evs(self):
        _ = self.interface.active_evs()
        self.simulator.get_active_evs.assert_called_once()

    def test_last_applied_pilot_signals_low_iteration(self):
        self.simulator.iteration = 1
        self.assertEqual(self.interface.last_applied_pilot_signals, {})

    def test_evse_voltage(self):
        self.assertEqual(self.interface.evse_voltage('PS-002'), 240)

    def test_is_feasible_empty_schedule(self):
        self.assertTrue(self.interface.is_feasible({}))

    def test_is_feasible_unequal_schedules(self):
        with self.assertRaises(InvalidScheduleError):
            self.interface.is_feasible(
                {'PS-001' : [1, 2], 'PS-002' : [3, 4, 5], 'PS-003' : [4, 5]})

    def test_is_feasible(self):
        self.simulator.network = create_autospec(ChargingNetwork)
        self.simulator.network.station_ids = ['PS-002', 'PS-001', 'PS-003']
        _ = self.interface.is_feasible({'PS-001' : [1, 2], 'PS-002' : [4, 5]})
        network_is_feasible_args = self.interface._simulator.network.is_feasible.call_args
        # Check that the call to the network's is_feasible method has the correct arguments
        np.testing.assert_allclose(network_is_feasible_args[0][0], np.array([[4, 5], [1, 2], [0, 0]]))
        # Network's is_feasible method has its second argument (linear) defaulting to False. Check this is the case.
        self.assertEqual(network_is_feasible_args[0][1], False)