from unittest import TestCase
from unittest.mock import Mock, create_autospec

import pandas as pd
import numpy as np

from acnportal.acnsim import Simulator, InvalidScheduleError
from acnportal.acnsim.simulator import _overwrite_at_index
from acnportal.acnsim.network import ChargingNetwork
from acnportal.algorithms import BaseAlgorithm
from acnportal.acnsim.events import EventQueue
from datetime import datetime
from acnportal.acnsim.models import EVSE

class TestSimulator(TestCase):
	# TODO: Complete all test cases. Currently only has test cases
	# for methods involving pilot_signals and charging_rates
    # Namely, a test for the run function
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
        events = create_autospec(EventQueue)
        self.simulator = Simulator(network, scheduler, events, start)

    def test_correct_on_init_pilot_signals(self):
        EVSEs = self.simulator.network.space_ids
        ps_station_ids = self.simulator.pilot_signals.columns
        for station_id in EVSEs:
            self.assertTrue(not self.simulator.pilot_signals[station_id].to_numpy())
            self.assertTrue(station_id in ps_station_ids)
        self.assertEqual(len(self.simulator.pilot_signals.columns), len(EVSEs))
    
    def test_correct_on_init_charging_rates(self):
        EVSEs = self.simulator.network.space_ids
        cr_station_ids = self.simulator.charging_rates.columns
        for station_id in EVSEs:
            self.assertTrue(not self.simulator.charging_rates[station_id].to_numpy())
            self.assertTrue(station_id in cr_station_ids)
        self.assertEqual(len(self.simulator.charging_rates.columns), len(EVSEs))

    # TODO: Need a test for run(); likely an integration test.

    def test_expand_pilots(self):
        self.simulator._expand_pilots()
        for station_id in self.simulator.pilot_signals.columns:
            self.assertEqual(len(self.simulator.pilot_signals[station_id]), self.simulator._iteration + 1)
        # TODO: this subtest requires running the simulation to cover all cases
        # maybe integration testing?

    def test_update_schedules_unequal_lengths(self):
        new_schedule = {'PS-001' : [24], 'PS-002' : [16, 24]}
        with self.assertRaises(InvalidScheduleError):
            self.simulator._update_schedules(new_schedule)

    def test_update_schedules_not_in_network(self):
        new_schedule = {'PS-001' : [24, 16], 'PS-004' : [16, 24]}
        with self.assertRaises(KeyError):
            self.simulator._update_schedules(new_schedule)

    def test_update_schedules_valid_schedule(self):
        # TODO: figure out how to mock static helper function correctly
        new_schedule = {'PS-001' : [24, 16], 'PS-002' : [16, 24]}
        # mock_overwrite_at_index = Mock(_overwrite_at_index)
        self.simulator._update_schedules(new_schedule)
        # mock_overwrite_at_index.assert_any_call(0, [], [24, 16])
        # mock_overwrite_at_index.assert_any_call(0, [], [16, 24])
        # mock_overwrite_at_index.assert_any_call(0, [], [0, 0])
        self.assertTrue((self.simulator.pilot_signals['PS-001'] == np.array([24, 16])).all())
        self.assertTrue((self.simulator.pilot_signals['PS-002'] == np.array([16, 24])).all())
        self.assertTrue((self.simulator.pilot_signals['PS-003'] == np.array([0, 0])).all())

    # TODO: a test for _store_actual_charging_rates. Seems like it would be more of an
    # integration test.