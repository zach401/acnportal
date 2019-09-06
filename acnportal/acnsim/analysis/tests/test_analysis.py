from unittest import TestCase

from acnportal import acnsim
from acnportal.acnsim import Simulator
from acnportal.acnsim.network import ChargingNetwork
from acnportal.acnsim import acndata_events
from acnportal.acnsim import sites
from acnportal.algorithms import BaseAlgorithm
from acnportal.acnsim.events import EventQueue, Event
from datetime import datetime
from acnportal.acnsim.models import EVSE

import pytz
import numpy as np
import os
import json
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

def to_array_dict(list_dict):
    ''' Converts a dictionary of strings to lists to a dictionary of strings to numpy arrays. '''
    return {key : np.array(value) for key, value in list_dict.items()}

class TestAnalysis(TestCase):
    @classmethod
    def setUpClass(self):
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

        with open(os.path.join(os.path.dirname(__file__), 'edf_algo_true_analysis_fields.json'), 'r') as infile:
            self.edf_algo_true_analysis_dict = json.load(infile)

    def test_aggregate_current(self):
        np.testing.assert_allclose(acnsim.aggregate_current(self.sim),
            np.array(self.edf_algo_true_analysis_dict['aggregate_current']))

    def compare_array_dicts(self, array_dict1, array_dict2):
        self.assertEqual(sorted(list(array_dict1.keys())), sorted(list(array_dict2.keys())))
        for key in array_dict1.keys():
            np.testing.assert_allclose(array_dict1[key], array_dict2[key])

    def test_constraint_currents_all_linear(self):
        self.compare_array_dicts(
            acnsim.constraint_currents(self.sim),
            to_array_dict(
                self.edf_algo_true_analysis_dict['constraint_currents_all_linear']))

    def test_constraint_currents_some_linear(self):
        self.compare_array_dicts(
            acnsim.constraint_currents(self.sim, constraint_ids=['Primary A', 'Secondary C']),
            to_array_dict(
                self.edf_algo_true_analysis_dict['constraint_currents_some_linear']))

    def test_proportion_of_energy_delivered(self):
        self.assertEqual(
            acnsim.proportion_of_energy_delivered(self.sim),
            self.edf_algo_true_analysis_dict['proportion_of_energy_delivered'])

    def test_proportion_of_demands_met(self):
        self.assertEqual(
            acnsim.proportion_of_demands_met(self.sim),
            self.edf_algo_true_analysis_dict['proportion_of_demands_met'])

    def test_current_unbalance_nema(self):
        np.testing.assert_allclose(
            acnsim.current_unbalance(self.sim, ['Primary A', 'Primary B', 'Primary C']),
            np.array(self.edf_algo_true_analysis_dict['primary_current_unbalance_nema']))
        np.testing.assert_allclose(
            acnsim.current_unbalance(self.sim, ['Secondary A', 'Secondary B', 'Secondary C']),
            np.array(self.edf_algo_true_analysis_dict['secondary_current_unbalance_nema']))

    # def test_current_unbalance_sym_comp(self):
    #     print(
    #         acnsim.current_unbalance(self.sim, ['Primary A', 'Primary B', 'Primary C'], type='SYM_COMP'),
    #         np.array(self.edf_algo_true_analysis_dict['primary_current_unbalance_sym_comp'])
    #         )
    #     np.testing.assert_allclose(
    #         acnsim.current_unbalance(self.sim, ['Primary A', 'Primary B', 'Primary C'], type='SYM_COMP'),
    #         np.array(self.edf_algo_true_analysis_dict['primary_current_unbalance_sym_comp']))
    #     np.testing.assert_allclose(
    #         acnsim.current_unbalance(self.sim, ['Secondary A', 'Secondary B', 'Secondary C'], type='SYM_COMP'),
    #         np.array(self.edf_algo_true_analysis_dict['secondary_current_unbalance_sym_comp']))
