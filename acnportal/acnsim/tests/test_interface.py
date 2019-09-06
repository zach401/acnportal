from unittest import TestCase
from unittest.mock import Mock, create_autospec

from acnportal.acnsim import Simulator, Interface, InvalidScheduleError
from acnportal.acnsim.network import ChargingNetwork
from acnportal.acnsim.models import EV

class TestSimulator(TestCase):
    def setUp(self):
        self.simulator = create_autospec(Simulator)
        self.network = create_autospec(ChargingNetwork)
        self.simulator.network = self.network
        self.interface = Interface(self.simulator)

    def test_init(self):
        self.assertIs(self.interface._simulator, self.simulator)

    def test_active_evs(self):
        _ = self.interface.active_evs()
        self.simulator.get_active_evs.assert_called_once()

    def test_last_applied_pilot_signals(self):
        pass

    def test_last_applied_pilot_signals_low_iteration(self):
        self.simulator.iteration = 1
        self.assertEqual(self.interface.last_applied_pilot_signals, {})

    def test_evse_voltage(self):
        self.network.station_ids = ['PS-002', 'PS-001', 'PS-003']
        self.network.voltages = [120, 360, 240]
        self.assertEqual(self.interface.evse_voltage('PS-001'), 360)

    def test_is_feasible_empty_schedule(self):
        self.assertTrue(self.interface.is_feasible({}))

    def test_is_feasible_unequal_schedules(self):
        with self.assertRaises(InvalidScheduleError):
            self.interface.is_feasible(
                {'PS-001' : [1, 2], 'PS-002' : [3, 4, 5], 'PS-003' : [4, 5]})

    def test_is_feasible(self):
        pass
        # self.network.station_ids = ['PS-002', 'PS-001', 'PS-003']
        # all_calls = self.interface.is_feasible({'PS-001' : [1, 2], 'PS-003' : [4, 5]}).call_args
        # print(all_calls)