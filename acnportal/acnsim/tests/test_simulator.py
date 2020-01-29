from unittest import TestCase
from unittest.mock import Mock, create_autospec

import numpy as np
import pandas as pd

from acnportal.acnsim import Simulator
from acnportal.acnsim.network import ChargingNetwork, Current
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

class TestSimulatorWarnings(TestCase):
    def test_update_schedules_infeasible_schedule(self):
        network = ChargingNetwork()
        network.register_evse(EVSE('PS-001'), 240, 0)
        network.register_evse(EVSE('PS-004'), 240, 0)
        network.register_evse(EVSE('PS-003'), 240, 0)
        network.register_evse(EVSE('PS-002'), 240, 0)
        network.register_evse(EVSE('PS-006'), 240, 0)
        curr_dict1 = {'PS-001' : 0.25, 'PS-002' : 0.50, 'PS-003' : -0.25}
        current1 = Current(curr_dict1)
        curr_dict2 = {'PS-006' : 0.30, 'PS-004' : -0.60, 'PS-002' : 0.50}
        current2 = Current(curr_dict2)
        network.add_constraint(current1, 50, name='first_constraint')
        network.add_constraint(current2, 10)
        start = Mock(datetime)
        scheduler = create_autospec(BaseAlgorithm)
        scheduler.max_recompute = None
        events = EventQueue(events=[Event(1), Event(2)])
        simulator = Simulator(network, scheduler, events, start)

        bad_schedule = {'PS-001': [200, 0, 160, 0 ],
                        'PS-004': [0,   0, 0,   0 ],
                        'PS-003': [0,   0, 0,   0 ],
                        'PS-002': [0,   0, 26,  0 ],
                        'PS-006': [0,   0, 0,   40]}
        with self.assertWarnsRegex(
                UserWarning,
                r'Invalid schedule provided at iteration 0. '
                r'Max violation is 2.9999\d+? A on _const_1 at time index 2.'):
            simulator._update_schedules(bad_schedule)
