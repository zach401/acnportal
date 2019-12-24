from unittest import TestCase
from unittest.mock import Mock, create_autospec

import numpy as np
import pandas as pd

from acnportal.acnsim import Simulator
from acnportal.acnsim.network import ChargingNetwork
from acnportal.algorithms import BaseAlgorithm
from acnportal.acnsim.events import EventQueue, Event
from datetime import datetime
from acnportal.acnsim.models import EVSE


class TestSimulator(TestCase):
    def setUp(self):
        start = Mock(datetime)
        network = ChargingNetwork()
        evse1 = EVSE('PS-001', max_rate=32)
        network.register_evse(evse1, 240, 0)
        evse2 = EVSE('PS-002', max_rate=32)
        network.register_evse(evse2, 240, 0)
        evse3 = EVSE('PS-003', max_rate=32)
        network.register_evse(evse3, 240, 0)
        scheduler = create_autospec(BaseAlgorithm)
        scheduler.max_recompute = None
        events = EventQueue(events=[Event(1), Event(2)])
        self.simulator = Simulator(network, scheduler, events, start)

    def test_correct_on_init_pilot_signals(self):
        np.testing.assert_allclose(self.simulator.pilot_signals,
            np.zeros((len(self.simulator.network.station_ids), self.simulator.event_queue.get_last_timestamp() + 1)))

    def test_correct_on_init_charging_rates(self):
        np.testing.assert_allclose(self.simulator.charging_rates,
            np.zeros((len(self.simulator.network.station_ids), self.simulator.event_queue.get_last_timestamp() + 1)))

    def test_update_schedules_not_in_network(self):
        new_schedule = {'PS-001' : [24, 16], 'PS-004' : [16, 24]}
        with self.assertRaises(KeyError):
            self.simulator._update_schedules(new_schedule)

    def test_update_schedules_valid_schedule(self):
        new_schedule = {'PS-001' : [24, 16], 'PS-002' : [16, 24]}
        self.simulator._update_schedules(new_schedule)
        np.testing.assert_allclose(self.simulator.pilot_signals[:, :2], np.array([[24, 16], [16, 24], [0, 0]]))

    def test_index_of_evse_error(self):
        with self.assertRaises(KeyError):
            _ = self.simulator.index_of_evse('PS-004')

    def test_index_of_evse(self):
        idx = self.simulator.index_of_evse('PS-002')
        self.assertEqual(idx, 1)

    def test_pilot_signals_as_df(self):
        self.simulator.pilot_signals = np.array([[1, 2], [3, 4], [5, 6]])
        outframe = self.simulator.pilot_signals_as_df()
        pd.testing.assert_frame_equal(outframe,
            pd.DataFrame(np.array([[1, 3, 5], [2, 4, 6]]),
                columns=['PS-001', 'PS-002', 'PS-003']))

    def test_charging_rates_as_df(self):
        self.simulator.charging_rates = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])
        outframe = self.simulator.charging_rates_as_df()
        pd.testing.assert_frame_equal(outframe,
            pd.DataFrame(np.array([[1.1, 3.1, 5.1], [2.1, 4.1, 6.1]]),
                columns=['PS-001', 'PS-002', 'PS-003']))

# TODO: Complete this case. Runs a simulation which all schedules
# are empty to test the null edge case for simulator operations
# Can also greatly simplify this case
class TestEmptyScheduleSim(TestCase):
    @classmethod
    def setUpClass(self):
        # Make instance of each class in registry.
        # Battery
        self.battery1 = acnsim.Battery(100, 50, 20)
        self.battery1._current_charging_power = 10

        # Linear2StageBattery
        self.battery2 = acnsim.Linear2StageBattery(
            100, 50, 20)
        self.battery2._current_charging_power = 10
        self.battery2._noise_level = 0.1
        self.battery2._transition_soc = 0.85

        # EVs
        self.ev1 = acnsim.EV(
            10, 20, 30, 'PS-001', 'EV-001', deepcopy(self.battery1),
            estimated_departure=25
        )
        self.ev1._energy_delivered = 50
        self.ev1._current_charging_rate = 10

        self.ev2 = acnsim.EV(
            10, 20, 30, 'PS-002', 'EV-002',
            deepcopy(self.battery2), estimated_departure=25
        )
        self.ev2._energy_delivered = 50
        self.ev2._current_charging_rate = 10

        self.ev3 = acnsim.EV(
            10, 20, 30, 'PS-003', 'EV-003',
            deepcopy(self.battery2), estimated_departure=25
        )
        self.ev3._energy_delivered = 50
        self.ev3._current_charging_rate = 10

        # EVSEs
        self.evse0 = acnsim.EVSE('PS-000', max_rate=32,
            min_rate=0)

        self.evse1 = acnsim.EVSE('PS-001', max_rate=32,
            min_rate=0)
        self.evse1.plugin(self.ev1)
        self.evse1.set_pilot(30, 220, 1)

        self.evse2 = acnsim.DeadbandEVSE('PS-002', max_rate=32,
            min_rate=0, deadband_end=4)
        self.evse2.plugin(self.ev2)
        self.evse2.set_pilot(30, 220, 1)

        self.evse3 = acnsim.FiniteRatesEVSE('PS-003',
            allowable_rates=[0, 8, 16, 24, 32])
        self.evse3.plugin(self.ev3)
        self.evse3.set_pilot(24, 220, 1)

        # Events
        self.event = acnsim.Event(0)
        self.plugin_event1 = acnsim.PluginEvent(10, self.ev1)
        self.unplug_event = acnsim.UnplugEvent(20, 'PS-001',
            'EV-001')
        self.recompute_event = acnsim.RecomputeEvent(30)
        # Plugin with a previously-unseen ev.
        self.plugin_event2 = acnsim.PluginEvent(40, self.ev2)
        # Plugin with a previously-seen ev.
        self.plugin_event3 = acnsim.PluginEvent(50, self.ev1)


        # EventQueue
        self.event_queue = acnsim.EventQueue()
        self.event_queue.add_events([self.event, self.plugin_event1,
            self.recompute_event, self.plugin_event2,
            self.plugin_event3])

        # Network
        self.network = acnsim.ChargingNetwork()
        self.network.register_evse(self.evse1, 220, 30)
        self.network.register_evse(self.evse2, 220, 150)
        self.network.register_evse(self.evse3, 220, -90)
        self.network.constraint_matrix = np.array([[1, 0], [0, 1]])
        self.network.magnitudes = np.array([30, 30])
        self.network.constraint_index = ['C1', 'C2']

        # Simulator
        self.simulator = acnsim.Simulator(
            self.network, UncontrolledCharging(),
            self.event_queue, datetime(2019, 1, 1)
        )
