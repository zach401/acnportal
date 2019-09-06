from unittest import TestCase
from unittest.mock import Mock, create_autospec

import pandas as pd
import numpy as np
import os

from acnportal.acnsim import Simulator
from acnportal.acnsim.network import ChargingNetwork
from acnportal.acnsim import acndata_events
from acnportal.acnsim import sites
from acnportal.algorithms import BaseAlgorithm
from acnportal.acnsim.events import EventQueue, Event
from datetime import datetime
from acnportal.acnsim.models import EVSE

import pickle
from acnportal.algorithms import BaseAlgorithm
import pytz
from copy import deepcopy

class EarliestDeadlineFirstAlgo(BaseAlgorithm):
    ''' See EarliestDeadlineFirstAlgo in tutorial 2. '''
    def __init__(self, increment=1):
        super().__init__()
        self._increment = increment

    def schedule(self, active_evs):
        schedule = {ev.station_id: [0] for ev in active_evs}

        sorted_evs = sorted(active_evs, key=lambda x: x.departure)

        for ev in sorted_evs:
            schedule[ev.station_id] = [self.interface.max_pilot_signal(ev.station_id)]

            while not self.interface.is_feasible(schedule):
                schedule[ev.station_id] = [schedule[ev.station_id][0] - self._increment]

                if schedule[ev.station_id][0] < 0:
                    schedule[ev.station_id] = [0]
                    break
        return schedule

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

    def test_tutorial_2(self):
        # Integration test. Tests that results of tutorial 2 are unchanged
        timezone = pytz.timezone('America/Los_Angeles')
        start = timezone.localize(datetime(2018, 9, 5))
        end = timezone.localize(datetime(2018, 9, 6))
        period = 5  # minute
        voltage = 220  # volts
        default_battery_power = 32 * voltage / 1000 # kW
        site = 'caltech'

        cn = sites.caltech_acn(basic_evse=True, voltage=voltage)

        API_KEY = 'DEMO_TOKEN'
        events = acndata_events.generate_events(API_KEY, site, start, end, period, voltage, default_battery_power)


        sch = EarliestDeadlineFirstAlgo(increment=1)

        self.sim = Simulator(deepcopy(cn), sch, deepcopy(events), start, period=period, max_recomp=1, verbose=False)
        self.sim.run()

        edf_algo_true_info_dict = {}
        edf_algo_new_info_dict = {}
        info_fields = ['pilot_signals', 'charging_rates', 'peak', 'ev_history', 'event_history']
        work_dir = os.path.join(os.path.dirname(__file__), "edf_algo_true_info_fields")
        for field in info_fields:
            with open(os.path.join(work_dir, "edf_algo_true_"+field+".p"), 'rb') as info_file:
                edf_algo_true_info_dict[field] = pickle.load(info_file)
                edf_algo_new_info_dict[field] = self.sim.__dict__[field]

        old_evse_keys = list(edf_algo_true_info_dict['pilot_signals'].keys())
        new_evse_keys = self.sim.network.station_ids
        self.assertEqual(sorted(new_evse_keys), sorted(old_evse_keys))


        edf_algo_new_info_dict['charging_rates'] = {new_evse_keys[i] : list(edf_algo_new_info_dict['charging_rates'][i]) for i in range(len(new_evse_keys))}
        edf_algo_new_info_dict['pilot_signals'] = {new_evse_keys[i] : list(edf_algo_new_info_dict['pilot_signals'][i]) for i in range(len(new_evse_keys))}

        for evse_key in new_evse_keys:
            np.testing.assert_allclose(np.array(edf_algo_true_info_dict['pilot_signals'][evse_key]),
                np.array(edf_algo_new_info_dict['pilot_signals'][evse_key])[:len(edf_algo_true_info_dict['pilot_signals'][evse_key])])
            np.testing.assert_allclose(np.array(edf_algo_true_info_dict['charging_rates'][evse_key]),
                np.array(edf_algo_new_info_dict['charging_rates'][evse_key])[:len(edf_algo_true_info_dict['charging_rates'][evse_key])])
        self.assertEqual(edf_algo_new_info_dict['peak'], edf_algo_true_info_dict['peak'])