from unittest import TestCase

from acnportal import acnsim
from acnportal.acnsim import Simulator
from acnportal.acnsim import acndata_events
from acnportal.acnsim import sites
from acnportal.algorithms import BaseAlgorithm
from datetime import datetime

import pytz
import numpy as np
import os
import json
from copy import deepcopy


class EarliestDeadlineFirstAlgoStateful(BaseAlgorithm):
    """ See EarliestDeadlineFirstAlgo in tutorial 2. This is a stateful version that
    occasionally records charging rates and pilot signals to test the last_applied_pilot_signals
    and last_actual_charging_rate functions in Interface.
    """

    def __init__(self, increment=1):
        super().__init__()
        self._increment = increment
        self.polled_pilots = {}
        self.polled_charging_rates = {}
        self.max_recompute = 1

    def schedule(self, active_evs):
        schedule = {ev.station_id: [0] for ev in active_evs}

        sorted_evs = sorted(active_evs, key=lambda x: x.departure)

        for ev in sorted_evs:
            schedule[ev.station_id] = [self.interface.max_pilot_signal(ev.station_id)]

            while not self.interface.is_feasible(schedule):
                schedule[ev.station_id][0] -= self._increment

                if schedule[ev.station_id][0] < 0:
                    schedule[ev.station_id] = [0]
                    break
        if not self.interface.current_time % 100:
            self.polled_pilots[
                str(self.interface.current_time)
            ] = self.interface.last_applied_pilot_signals
            self.polled_charging_rates[
                str(self.interface.current_time)
            ] = self.interface.last_actual_charging_rate
        return schedule


def to_array_dict(list_dict):
    """ Converts a dictionary of strings to lists to a dictionary of strings to numpy arrays. """
    return {key: np.array(value) for key, value in list_dict.items()}


class TestIntegration(TestCase):
    @classmethod
    def setUpClass(self):
        timezone = pytz.timezone("America/Los_Angeles")
        start = timezone.localize(datetime(2018, 9, 5))
        end = timezone.localize(datetime(2018, 9, 6))
        period = 5  # minute
        voltage = 220  # volts
        default_battery_power = 32 * voltage / 1000  # kW
        site = "caltech"

        cn = sites.caltech_acn(basic_evse=True, voltage=voltage)

        API_KEY = "DEMO_TOKEN"
        events = acndata_events.generate_events(
            API_KEY, site, start, end, period, voltage, default_battery_power
        )

        self.sch = EarliestDeadlineFirstAlgoStateful(increment=1)

        self.sim = Simulator(
            deepcopy(cn),
            self.sch,
            deepcopy(events),
            start,
            period=period,
            verbose=False,
        )
        self.sim.run()

        with open(
            os.path.join(
                os.path.dirname(__file__), "edf_algo_true_analysis_fields.json"
            ),
            "r",
        ) as infile:
            self.edf_algo_true_analysis_dict = json.load(infile)

        with open(
            os.path.join(
                os.path.dirname(__file__), "edf_algo_true_datetimes_array.json"
            ),
            "r",
        ) as infile:
            self.edf_algo_true_datetimes_array = json.load(infile)

        with open(
            os.path.join(os.path.dirname(__file__), "edf_algo_true_info_fields.json"),
            "r",
        ) as infile:
            self.edf_algo_true_info_dict = json.load(infile)

    def test_aggregate_current(self):
        np.testing.assert_allclose(
            acnsim.aggregate_current(self.sim),
            np.array(self.edf_algo_true_analysis_dict["aggregate_current"]),
        )

    def compare_array_dicts(self, array_dict1, array_dict2):
        self.assertEqual(
            sorted(list(array_dict1.keys())), sorted(list(array_dict2.keys()))
        )
        for key in array_dict1.keys():
            np.testing.assert_allclose(array_dict1[key], array_dict2[key])

    def test_constraint_currents_all_magnitudes(self):
        self.compare_array_dicts(
            acnsim.constraint_currents(self.sim),
            to_array_dict(
                self.edf_algo_true_analysis_dict["constraint_currents_all_linear"]
            ),
        )

    def test_constraint_currents_some_magnitudes(self):
        self.compare_array_dicts(
            acnsim.constraint_currents(
                self.sim, constraint_ids=["Primary A", "Secondary C"]
            ),
            to_array_dict(
                self.edf_algo_true_analysis_dict["constraint_currents_some_linear"]
            ),
        )

    def test_proportion_of_energy_delivered(self):
        self.assertEqual(
            acnsim.proportion_of_energy_delivered(self.sim),
            self.edf_algo_true_analysis_dict["proportion_of_energy_delivered"],
        )

    def test_proportion_of_demands_met(self):
        self.assertEqual(
            acnsim.proportion_of_demands_met(self.sim),
            self.edf_algo_true_analysis_dict["proportion_of_demands_met"],
        )

    def test_current_unbalance_nema_error(self):
        with self.assertRaises(ValueError):
            acnsim.current_unbalance(
                self.sim, ["Primary A", "Primary B", "Primary C"], unbalance_type="ABC"
            )

    def test_current_unbalance_nema_warning(self):
        with self.assertWarns(DeprecationWarning):
            acnsim.current_unbalance(
                self.sim, ["Primary A", "Primary B", "Primary C"], type="NEMA"
            )

    def test_current_unbalance_nema(self):
        # A RuntimeWarning is expected to be raised in this test case as
        # of acnportal v.1.0.3. See Github issue #57 for a discussion of
        # why this occurs.
        with self.assertWarns(RuntimeWarning):
            np.testing.assert_allclose(
                acnsim.current_unbalance(
                    self.sim, ["Primary A", "Primary B", "Primary C"]
                ),
                np.array(
                    self.edf_algo_true_analysis_dict["primary_current_unbalance_nema"]
                ), atol=1e-6
            )
        with self.assertWarns(RuntimeWarning):
            np.testing.assert_allclose(
                acnsim.current_unbalance(
                    self.sim, ["Secondary A", "Secondary B", "Secondary C"]
                ),
                np.array(
                    self.edf_algo_true_analysis_dict["secondary_current_unbalance_nema"]
                ), atol=1e-6
            )

    def test_datetimes_array_tutorial_2(self):
        np.testing.assert_equal(
            acnsim.datetimes_array(self.sim),
            np.array(
                [
                    np.datetime64(date_time)
                    for date_time in self.edf_algo_true_datetimes_array
                ]
            ),
        )

    def test_tutorial_2(self):
        old_evse_keys = list(self.edf_algo_true_info_dict["pilot_signals"].keys())
        new_evse_keys = self.sim.network.station_ids
        self.assertEqual(sorted(new_evse_keys), sorted(old_evse_keys))

        edf_algo_new_info_dict = {
            field: self.sim.__dict__[field]
            for field in self.edf_algo_true_info_dict.keys()
        }
        edf_algo_new_info_dict["charging_rates"] = {
            self.sim.network.station_ids[i]: list(
                edf_algo_new_info_dict["charging_rates"][i]
            )
            for i in range(len(self.sim.network.station_ids))
        }
        edf_algo_new_info_dict["pilot_signals"] = {
            self.sim.network.station_ids[i]: list(
                edf_algo_new_info_dict["pilot_signals"][i]
            )
            for i in range(len(self.sim.network.station_ids))
        }

        for evse_key in new_evse_keys:
            np.testing.assert_allclose(
                np.array(self.edf_algo_true_info_dict["pilot_signals"][evse_key]),
                np.array(edf_algo_new_info_dict["pilot_signals"][evse_key])[
                    : len(self.edf_algo_true_info_dict["pilot_signals"][evse_key])
                ],
            )
            np.testing.assert_allclose(
                np.array(self.edf_algo_true_info_dict["charging_rates"][evse_key]),
                np.array(edf_algo_new_info_dict["charging_rates"][evse_key])[
                    : len(self.edf_algo_true_info_dict["charging_rates"][evse_key])
                ],
            )
        self.assertEqual(
            edf_algo_new_info_dict["peak"], self.edf_algo_true_info_dict["peak"]
        )

    def test_lap_interface_func(self):
        with open(
            os.path.join(os.path.dirname(__file__), "edf_algo_pilot_signals.json"), "r"
        ) as infile:
            self.edf_algo_true_lap = json.load(infile)

        self.assertDictEqual(self.edf_algo_true_lap, self.sch.polled_pilots)

    def test_cr_interface_func(self):
        with open(
            os.path.join(os.path.dirname(__file__), "edf_algo_charging_rates.json"), "r"
        ) as infile:
            self.edf_algo_true_cr = json.load(infile)

        self.assertDictEqual(self.edf_algo_true_cr, self.sch.polled_charging_rates)


class EmptyScheduler(BaseAlgorithm):
    """ Always submits an empty schedule (empty dict) as the output
    of its run function.
    """

    def __init__(self):
        super().__init__()

    def schedule(self, active_evs):
        return {}


class TestEmptyScheduleSim(TestCase):
    @classmethod
    def setUpClass(self):
        self.plugin_event1 = acnsim.PluginEvent(
            10, acnsim.EV(10, 20, 30, "PS-001", "EV-001", acnsim.Battery(100, 50, 20))
        )
        self.plugin_event2 = acnsim.PluginEvent(
            20, acnsim.EV(20, 30, 40, "PS-002", "EV-002", acnsim.Battery(100, 50, 20))
        )

        self.evse1 = acnsim.EVSE("PS-001", max_rate=32, min_rate=0)
        self.evse2 = acnsim.EVSE("PS-002", max_rate=32, min_rate=0)

        self.event_queue = acnsim.EventQueue()
        self.event_queue.add_events([self.plugin_event1, self.plugin_event2])

        self.network = acnsim.ChargingNetwork()
        self.network.register_evse(self.evse1, 220, 30)
        self.network.register_evse(self.evse2, 220, 30)

        # Simulator with scheduler that always returns an empty
        # schedule.
        self.simulator = acnsim.Simulator(
            self.network,
            EmptyScheduler(),
            self.event_queue,
            datetime(2019, 1, 1),
            verbose=False,
        )

        self.simulator.run()

    def test_pilot_signals_empty_schedule(self):
        np.testing.assert_allclose(self.simulator.pilot_signals, np.zeros((2, 31)))

    def test_charging_rates_empty_schedule(self):
        np.testing.assert_allclose(self.simulator.charging_rates, np.zeros((2, 31)))
