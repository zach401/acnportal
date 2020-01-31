from unittest import TestCase

from acnportal import acnsim
from acnportal.algorithms import BaseAlgorithm, UncontrolledCharging
from .serialization_extensions import *

import json
import sys
import os
import numpy as np
from copy import deepcopy
from datetime import datetime


class TestJSONIO(TestCase):
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
        staying_time = 10
        ev1_arrival = 10
        self.ev1 = acnsim.EV(
            ev1_arrival, ev1_arrival + staying_time, 30, 'PS-001', 'EV-001',
            deepcopy(self.battery1), estimated_departure=25
        )
        self.ev1._energy_delivered = 0.05
        self.ev1._current_charging_rate = 10

        ev2_arrival = 40
        self.ev2 = acnsim.EV(
            ev2_arrival, ev2_arrival + staying_time, 30, 'PS-002', 'EV-002',
            deepcopy(self.battery2), estimated_departure=25
        )
        self.ev2._energy_delivered = 0.05
        self.ev2._current_charging_rate = 10

        ev3_arrival = 50
        self.ev3 = acnsim.EV(
            ev3_arrival, ev3_arrival + staying_time, 30, 'PS-003', 'EV-003',
            deepcopy(self.battery2), estimated_departure=25
        )
        self.ev3._energy_delivered = 0.05
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
        self.plugin_event2 = acnsim.PluginEvent(40, self.ev2)
        self.plugin_event3 = acnsim.PluginEvent(50, self.ev3)

        # EventQueue
        self.event_queue = acnsim.EventQueue()
        self.event_queue.add_events(
            [self.event,
             self.plugin_event1,
             self.recompute_event,
             self.plugin_event2,
             self.plugin_event3]
        )

        # Network
        self.network = acnsim.ChargingNetwork(violation_tolerance=1e-3,
                                              relative_tolerance=1e-5)

        self.network.register_evse(self.evse1, 220, 30)
        self.network.register_evse(self.evse2, 220, 150)
        self.network.register_evse(self.evse3, 220, -90)
        self.network.constraint_matrix = np.array([[1, 0, 0],
                                                   [0, 1, 0],
                                                   [0, 0, 1]])
        self.network.magnitudes = np.array([32, 32, 32])
        self.network.constraint_index = ['C1', 'C2', 'C3']

        # Simulator
        self.simulator = acnsim.Simulator(
            self.network, UncontrolledCharging(),
            self.event_queue, datetime(2019, 1, 1),
            verbose=False,
            store_schedule_history=True
        )

        # Make a copy of the simulator to run
        self.simulator_run = deepcopy(self.simulator)
        # Do necessary unplugs.
        for station_id, evse in self.simulator_run.network._EVSEs.items():
            if evse.ev is not None:
                evse.unplug()
        # Run simulation
        self.simulator_run.run()

        # Make a copy of the simulator with signals
        self.simulator_signal = deepcopy(self.simulator)
        self.simulator_signal.signals = {'a': [0, 1, 2], 'b': [3, 4]}

        self.simulator_hard_signal = deepcopy(self.simulator)
        self.simulator_hard_signal.signals = {'a': BaseAlgorithm()}

        self.simulator_no_sch_hist = deepcopy(self.simulator)
        self.simulator_no_sch_hist.schedule_history = None

    def test_battery_json(self):
        battery_json = self.battery1.to_json()
        battery_loaded = acnsim.Battery.from_json(battery_json)
        self.assertIsInstance(
            battery_loaded, acnsim.Battery)
        self.assertEqual(
            battery_loaded.__dict__, self.battery1.__dict__)

    def test_linear_2_stage_battery_json(self):
        battery_json = self.battery2.to_json()
        self.assertEqual(len(json.loads(battery_json)['context_dict'].keys()),
                         1)
        battery_loaded = acnsim.Linear2StageBattery.from_json(battery_json)
        self.assertIsInstance(
            battery_loaded, acnsim.Linear2StageBattery)
        self.assertEqual(
            battery_loaded.__dict__, self.battery2.__dict__)

    def test_ev_json(self):
        ev_fields = ['_arrival', '_departure', '_requested_energy',
                     '_estimated_departure', '_session_id', '_station_id',
                     '_energy_delivered', '_current_charging_rate']

        def _load_dump_compare_helper(ev, bat_type):
            assert isinstance(ev._battery, bat_type)
            ev_json = ev.to_json()
            ev_loaded = acnsim.EV.from_json(ev_json)
            self.assertIsInstance(ev_loaded, acnsim.EV)
            for field in ev_fields:
                self.assertEqual(getattr(ev, field),
                                 getattr(ev_loaded, field))
            self.assertIsInstance(ev_loaded._battery, bat_type)
            self.assertEqual(ev_loaded._battery.__dict__,
                             ev._battery.__dict__)

        _load_dump_compare_helper(self.ev1, acnsim.Battery)
        _load_dump_compare_helper(self.ev2,
                                  acnsim.Linear2StageBattery)
        _load_dump_compare_helper(self.ev3,
                                  acnsim.Linear2StageBattery)

    def test_evse_no_ev_json(self):
        evse_json = self.evse0.to_json()
        evse_loaded = acnsim.EVSE.from_json(evse_json)
        self.assertIsInstance(evse_loaded, acnsim.EVSE)

        evse_fields = ['_station_id', '_max_rate',
                       '_min_rate', '_current_pilot', 'is_continuous']

        for field in evse_fields:
            self.assertEqual(getattr(self.evse0, field),
                             getattr(evse_loaded, field))
        self.assertEqual(getattr(evse_loaded, '_ev'), None)

    def test_evse_json(self):
        evse_json = self.evse1.to_json()
        evse_loaded = acnsim.EVSE.from_json(evse_json)
        self.assertIsInstance(evse_loaded, acnsim.EVSE)

        evse_fields = ['_station_id', '_max_rate',
                       '_min_rate', '_current_pilot', 'is_continuous']

        for field in evse_fields:
            self.assertEqual(getattr(self.evse1, field),
                             getattr(evse_loaded, field))
        self.assertEqual(
            getattr(getattr(evse_loaded, '_ev'), '_session_id'),
            'EV-001'
        )

    def test_deadband_evse_json(self):
        evse_json = self.evse2.to_json()
        evse_loaded = acnsim.DeadbandEVSE.from_json(evse_json)
        self.assertIsInstance(evse_loaded,
                              acnsim.DeadbandEVSE)

        evse_fields = ['_station_id', '_max_rate', '_min_rate',
                       '_current_pilot', '_deadband_end', 'is_continuous']

        for field in evse_fields:
            self.assertEqual(getattr(self.evse2, field),
                             getattr(evse_loaded, field))
        self.assertEqual(
            getattr(getattr(evse_loaded, '_ev'), '_session_id'),
            'EV-002'
        )

    def test_finite_rates_evse_json(self):
        evse_json = self.evse3.to_json()
        evse_loaded = acnsim.FiniteRatesEVSE.from_json(evse_json)
        self.assertIsInstance(evse_loaded,
                              acnsim.FiniteRatesEVSE)

        evse_fields = ['_station_id', '_max_rate', '_min_rate',
                       '_current_pilot', 'is_continuous', 'allowable_rates']

        for field in evse_fields:
            self.assertEqual(getattr(self.evse3, field),
                             getattr(evse_loaded, field))
        self.assertEqual(
            getattr(getattr(evse_loaded, '_ev'), '_session_id'),
            'EV-003'
        )

    def test_event_json(self):
        event_json = self.event.to_json()
        event_loaded = acnsim.Event.from_json(event_json)
        self.assertIsInstance(event_loaded, acnsim.Event)

        event_fields = ['timestamp', 'event_type', 'precedence']

        for field in event_fields:
            self.assertEqual(getattr(self.event, field),
                             getattr(event_loaded, field))

    def test_plugin_event_json(self):
        event_json = self.plugin_event1.to_json()
        event_loaded = acnsim.PluginEvent.from_json(event_json)
        self.assertIsInstance(event_loaded, acnsim.PluginEvent)

        event_fields = ['timestamp', 'event_type', 'precedence']

        for field in event_fields:
            self.assertEqual(getattr(self.plugin_event1, field),
                             getattr(event_loaded, field))
        self.assertEqual(
            getattr(getattr(event_loaded, 'ev'), '_session_id'),
            'EV-001'
        )

    def test_unplug_event_json(self):
        event_json = self.unplug_event.to_json()
        event_loaded = acnsim.UnplugEvent.from_json(event_json)
        self.assertIsInstance(event_loaded, acnsim.UnplugEvent)

        event_fields = ['timestamp', 'event_type', 'precedence',
                        'station_id', 'session_id']

        for field in event_fields:
            self.assertEqual(getattr(self.unplug_event, field),
                             getattr(event_loaded, field))

    def test_recompute_event_json(self):
        event_json = self.recompute_event.to_json()
        event_loaded = acnsim.RecomputeEvent.from_json(event_json)
        self.assertIsInstance(event_loaded, acnsim.RecomputeEvent)

        event_fields = ['timestamp', 'event_type', 'precedence']

        for field in event_fields:
            self.assertEqual(getattr(self.recompute_event, field),
                             getattr(event_loaded, field))

    def test_event_queue_json(self):
        event_queue_json = self.event_queue.to_json()
        event_queue_loaded = acnsim.EventQueue.from_json(event_queue_json)
        self.assertIsInstance(event_queue_loaded, acnsim.EventQueue)

        self.assertEqual(self.event_queue._timestep,
                         event_queue_loaded._timestep)

        for (ts, event), (tsl, event_loaded) in \
                zip(self.event_queue._queue, event_queue_loaded._queue):
            self.assertEqual(ts, tsl)
            self.assertIsInstance(event_loaded, acnsim.Event)
            if not isinstance(event, acnsim.PluginEvent):
                self.assertEqual(type(event), type(event_loaded))
                self.assertEqual(event.__dict__,
                                 event_loaded.__dict__)
            else:
                for field in ['timestamp', 'event_type', 'precedence']:
                    self.assertEqual(
                        getattr(event, field),
                        getattr(event_loaded, field)
                    )
                self.assertEqual(
                    getattr(getattr(event_loaded, 'ev'),
                            '_session_id'),
                    getattr(getattr(event, 'ev'),
                            '_session_id')
                )

    def test_charging_network_json(self):
        network_json = self.network.to_json()
        network_loaded = acnsim.ChargingNetwork.from_json(network_json)
        self.assertIsInstance(network_loaded, acnsim.ChargingNetwork)

        network_np_fields = ['constraint_matrix', 'magnitudes',
                             '_voltages', '_phase_angles',
                             'violation_tolerance', 'relative_tolerance']
        for field in network_np_fields:
            np.testing.assert_equal(getattr(self.network, field),
                                    getattr(network_loaded, field))

        self.assertEqual(self.network.constraint_index,
                         network_loaded.constraint_index)

        for (station_id, evse), (station_id_l, evse_l) in \
                zip(self.network._EVSEs.items(),
                    network_loaded._EVSEs.items()):
            self.assertEqual(station_id, station_id_l)
            self.assertEqual(evse.station_id, evse_l.station_id)

    def _sim_compare_helper(self, sim):
        simulator_json = sim.to_json()
        simulator_loaded = acnsim.Simulator.from_json(simulator_json)
        self.assertIsInstance(simulator_loaded, acnsim.Simulator)

        simulator_fields = ['period', 'max_recompute', 'verbose',
                            'peak', '_iteration', '_resolve',
                            '_last_schedule_update', 'schedule_history']
        for field in simulator_fields:
            self.assertEqual(getattr(sim, field),
                             getattr(simulator_loaded, field))

        if sys.version_info[1] < 7:
            self.assertEqual(sim.start.isoformat(), simulator_loaded.start)
        else:
            self.assertEqual(sim.start, simulator_loaded.start)

        self.assertIsInstance(simulator_loaded.scheduler, UncontrolledCharging)

        if simulator_loaded.signals is not None:
            self.assertEqual(sim.signals, simulator_loaded.signals)

        np.testing.assert_equal(sim.pilot_signals,
                                simulator_loaded.pilot_signals)
        np.testing.assert_equal(sim.charging_rates,
                                simulator_loaded.charging_rates)

        network_attrs = ['station_ids', 'active_station_ids',
                         'voltages', 'phase_angles']
        for attr in network_attrs:
            self.assertEqual(getattr(sim.network, attr),
                             getattr(simulator_loaded.network, attr))

        for (ts, event), (tsl, event_loaded) in \
                zip(sim.event_queue._queue,
                    simulator_loaded.event_queue._queue):
            self.assertEqual(ts, tsl)
            self.assertEqual(event.event_type, event_loaded.event_type)
            self.assertEqual(event.timestamp, event_loaded.timestamp)

        for event, event_loaded in \
                zip(sim.event_history,
                    simulator_loaded.event_history):
            self.assertEqual(event.event_type, event_loaded.event_type)
            self.assertEqual(event.timestamp, event_loaded.timestamp)

        for station_id, ev in sim.ev_history.items():
            self.assertEqual(ev.session_id,
                             simulator_loaded.ev_history[
                                 station_id].session_id)

    def test_init_simulator_json(self):
        self._sim_compare_helper(self.simulator)

    def test_run_simulator_json(self):
        self._sim_compare_helper(self.simulator_run)

    def test_object_equalities(self):
        # Tests that certain object equality invariants are preserved
        # after loading.
        simulator_json = self.simulator_run.to_json()
        simulator_loaded = acnsim.Simulator.from_json(simulator_json)

        plugins = filter(lambda x: isinstance(x, acnsim.PluginEvent),
                         simulator_loaded.event_history)
        evs = list(simulator_loaded.ev_history.values())
        for plugin, ev in zip(plugins, evs):
            self.assertIs(plugin.ev, ev)

    def test_sim_signal_json(self):
        self._sim_compare_helper(self.simulator_signal)

    def test_sim_signal_warning(self):
        with self.assertWarns(UserWarning):
            self._sim_compare_helper(self.simulator_hard_signal)

    def test_sim_no_sch_hist(self):
        self._sim_compare_helper(self.simulator_no_sch_hist)

    def test_acnportal_version_inequality(self):
        with open(os.path.join(os.path.dirname(__file__),
                               'old_version.json'),
                  'r') as infile:
            with self.assertWarns(UserWarning):
                _ = acnsim.Event.from_json(json.load(infile))

    def test_numpy_version_inequality(self):
        with open(os.path.join(os.path.dirname(__file__),
                               'old_dependencies.json'),
                  'r') as infile:
            with self.assertWarns(UserWarning):
                _ = acnsim.Event.from_json(json.load(infile))


class TestExtObjJSONIO(TestJSONIO):
    @classmethod
    def setUpClass(self):
        super().setUpClass()

        self.set_batt_event = SetAttrEvent(5)
        self.set_batt_event.set_extra_attr(self.battery1)

        self.event_queue.add_event(self.set_batt_event)

    def test_named_event_json(self):
        named_event = NamedEvent(0, "my_event")
        with self.assertWarns(UserWarning):
            named_event_json = named_event.to_json()
        with self.assertRaises(TypeError):
            _ = NamedEvent.from_json(named_event_json)

    def test_default_named_event_json(self):
        default_named_event = DefaultNamedEvent(5, "def_event")
        with self.assertWarns(UserWarning):
            default_named_event_json = default_named_event.to_json()
        default_named_event_loaded = \
            DefaultNamedEvent.from_json(default_named_event_json)
        self.assertIsInstance(default_named_event_loaded,
                              DefaultNamedEvent)

        event_fields = ['timestamp', 'event_type', 'precedence']

        for field in event_fields:
            self.assertEqual(getattr(default_named_event, field),
                             getattr(default_named_event_loaded, field))

        self.assertEqual(default_named_event_loaded.name, "my_event")

    def test_set_named_event_json(self):
        set_named_event = SetAttrEvent(5)
        set_named_event.set_extra_attr("set_event")
        with self.assertWarns(UserWarning):
            set_named_event_json = set_named_event.to_json()
        with self.assertWarns(UserWarning):
            set_named_event_loaded = \
                SetAttrEvent.from_json(set_named_event_json)
        self.assertIsInstance(set_named_event_loaded,
                              SetAttrEvent)

        event_fields = ['timestamp', 'event_type', 'precedence', 'extra_attr']

        for field in event_fields:
            self.assertEqual(getattr(set_named_event, field),
                             getattr(set_named_event_loaded, field))

    def test_set_list_event_json(self):
        set_list_event = SetAttrEvent(5)
        set_list_event.set_extra_attr(["set_event1", "set_event2"])
        with self.assertWarns(UserWarning):
            set_list_event_json = set_list_event.to_json()
        with self.assertWarns(UserWarning):
            set_list_event_loaded = \
                SetAttrEvent.from_json(set_list_event_json)
        self.assertIsInstance(set_list_event_loaded,
                              SetAttrEvent)

        event_fields = ['timestamp', 'event_type', 'precedence', 'extra_attr']

        for field in event_fields:
            self.assertEqual(getattr(set_list_event, field),
                             getattr(set_list_event_loaded, field))

    def test_set_batt_event_json(self):
        with self.assertWarns(UserWarning):
            set_batt_event_json = self.set_batt_event.to_json()
        with self.assertWarns(UserWarning):
            set_batt_event_loaded = \
                SetAttrEvent.from_json(set_batt_event_json)
        self.assertIsInstance(set_batt_event_loaded,
                              SetAttrEvent)

        event_fields = ['timestamp', 'event_type', 'precedence']

        for field in event_fields:
            self.assertEqual(getattr(self.set_batt_event, field),
                             getattr(set_batt_event_loaded, field))

        self.assertEqual(self.battery1.__dict__,
                         set_batt_event_loaded.extra_attr.__dict__)

    def test_set_np_event_json(self):
        set_np_event = SetAttrEvent(5)
        set_np_event.set_extra_attr(np.zeros((2, 2)))
        with self.assertWarns(UserWarning):
            set_np_event_json = set_np_event.to_json()
        with self.assertWarns(UserWarning):
            set_np_event_loaded = \
                SetAttrEvent.from_json(set_np_event_json)
        self.assertIsInstance(set_np_event_loaded,
                              SetAttrEvent)

        event_fields = ['timestamp', 'event_type', 'precedence']

        for field in event_fields:
            self.assertEqual(getattr(set_np_event, field),
                             getattr(set_np_event_loaded, field))

        self.assertEqual(set_np_event_loaded.extra_attr,
                         "array([[0., 0.],\n       [0., 0.]])")

    def test_batt_list_event_json(self):
        batt_list_event = BattListEvent(5, [self.battery1, self.battery2])
        batt_list_event_json = batt_list_event.to_json()
        batt_list_event_loaded = BattListEvent.from_json(batt_list_event_json)
        self.assertIsInstance(batt_list_event_loaded,
                              BattListEvent)

        event_fields = ['timestamp', 'event_type', 'precedence']

        for field in event_fields:
            self.assertEqual(getattr(batt_list_event, field),
                             getattr(batt_list_event_loaded, field))

        for batt, batt_loaded in zip(batt_list_event.batt_list,
                                     batt_list_event_loaded.batt_list):
            self.assertEqual(batt.__dict__, batt_loaded.__dict__)

    def test_event_queue_json(self):
        with self.assertWarns(UserWarning):
            event_queue_json = self.event_queue.to_json()
        with self.assertWarns(UserWarning):
            event_queue_loaded = acnsim.EventQueue.from_json(event_queue_json)
        self.assertIsInstance(event_queue_loaded, acnsim.EventQueue)

        self.assertEqual(self.event_queue._timestep,
                         event_queue_loaded._timestep)

        for (ts, event), (tsl, event_loaded) in \
                zip(self.event_queue._queue, event_queue_loaded._queue):
            self.assertEqual(ts, tsl)
            self.assertIsInstance(event_loaded, acnsim.Event)
            if isinstance(event_loaded, acnsim.PluginEvent):
                for field in ['timestamp', 'event_type', 'precedence']:
                    self.assertEqual(
                        getattr(event, field),
                        getattr(event_loaded, field)
                    )
                self.assertEqual(
                    getattr(getattr(event_loaded, 'ev'),
                            '_session_id'),
                    getattr(getattr(event, 'ev'),
                            '_session_id')
                )
            elif isinstance(event_loaded, SetAttrEvent):
                for field in ['timestamp', 'event_type', 'precedence']:
                    self.assertEqual(
                        getattr(event, field),
                        getattr(event_loaded, field)
                    )
                self.assertEqual(
                    getattr(event_loaded, 'extra_attr').__dict__,
                    getattr(event, 'extra_attr').__dict__
                )
            else:
                self.assertEqual(type(event), type(event_loaded))
                self.assertEqual(event.__dict__,
                                 event_loaded.__dict__)
