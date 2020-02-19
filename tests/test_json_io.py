from unittest import TestCase

from acnportal import acnsim
from acnportal.algorithms import BaseAlgorithm, UncontrolledCharging
from .serialization_extensions import NamedEvent, DefaultNamedEvent
from .serialization_extensions import SetAttrEvent, BatteryListEvent

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
        for evse in self.simulator_run.network._EVSEs.values():
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

        # Class data used for testing.
        self.simple_attributes = {
            'Battery': [
                '_max_power', '_current_charging_power', '_current_charge',
                '_capacity', '_init_charge'
            ],
            'Linear2StageBattery': [
                '_max_power', '_current_charging_power', '_current_charge',
                '_capacity', '_init_charge', '_noise_level', '_transition_soc'
            ],
            'EV': [
                '_arrival', '_departure', '_requested_energy',
                '_estimated_departure', '_session_id', '_station_id',
                '_energy_delivered', '_current_charging_rate'
            ],
            'EVSE': [
                '_station_id', '_max_rate', '_min_rate', '_current_pilot',
                'is_continuous'
            ],
            'DeadbandEVSE': [
                '_station_id', '_max_rate', '_min_rate', '_current_pilot',
                '_deadband_end', 'is_continuous'
            ],
            'FiniteRatesEVSE': [
                '_station_id', '_max_rate', '_min_rate','_current_pilot',
                'is_continuous', 'allowable_rates'
            ],
            'Event': [
                'timestamp', 'event_type', 'precedence'
            ],
            'PluginEvent': [
                'timestamp', 'event_type', 'precedence'
            ],
            'UnplugEvent': [
                'timestamp', 'event_type', 'precedence', 'station_id',
                'session_id'
            ],
            'RecomputeEvent': [
                'timestamp', 'event_type', 'precedence'
            ],
            'EventQueue': [
                '_timestep'
            ],
            'ChargingNetwork': [
                'constraint_index', 'violation_tolerance', 'relative_tolerance'
            ],
            'Simulator': [
                'period', 'max_recompute', 'verbose', 'peak', '_iteration',
                '_resolve', '_last_schedule_update', 'schedule_history'
            ]
        }

    def _obj_compare_helper(self, obj, attributes=None):
        obj_json = obj.to_json()
        obj_class = type(obj)
        obj_loaded = obj_class.from_json(obj_json)
        self.assertIsInstance(obj_loaded, obj_class)
        if attributes is None:
            attributes = self.simple_attributes[obj_class.__name__]
        for attribute in attributes:
            self.assertEqual(getattr(obj, attribute),
                             getattr(obj_loaded, attribute))
        return obj_loaded

    def test_battery_json(self):
        _ = self._obj_compare_helper(self.battery1)

    def test_linear_2_stage_battery_json(self):
        _ = self._obj_compare_helper(self.battery2)

    def test_ev_json(self):
        def _load_dump_compare_helper(ev, bat_type):
            self.assertIsInstance(ev._battery, bat_type)
            ev_loaded = self._obj_compare_helper(ev)
            self.assertIsInstance(ev_loaded._battery, bat_type)
            self.assertEqual(ev_loaded._battery.__dict__,
                             ev._battery.__dict__)
        _load_dump_compare_helper(self.ev1, acnsim.Battery)
        _load_dump_compare_helper(self.ev2,
                                  acnsim.Linear2StageBattery)
        _load_dump_compare_helper(self.ev3,
                                  acnsim.Linear2StageBattery)

    def test_evse_no_ev_json(self):
        evse_loaded = self._obj_compare_helper(self.evse0)
        self.assertEqual(getattr(evse_loaded, '_ev'), None)

    def test_evse_json(self):
        evse_loaded = self._obj_compare_helper(self.evse1)
        self.assertEqual(
            getattr(getattr(evse_loaded, '_ev'), '_session_id'), 'EV-001')

    def test_deadband_evse_json(self):
        evse_loaded = self._obj_compare_helper(self.evse2)
        self.assertEqual(
            getattr(getattr(evse_loaded, '_ev'), '_session_id'), 'EV-002')

    def test_finite_rates_evse_json(self):
        evse_loaded = self._obj_compare_helper(self.evse3)
        self.assertEqual(
            getattr(getattr(evse_loaded, '_ev'), '_session_id'), 'EV-003')

    def test_event_json(self):
        _ = self._obj_compare_helper(self.event)

    def test_plugin_event_json(self):
        event_loaded = self._obj_compare_helper(self.plugin_event1)
        self.assertEqual(
            getattr(getattr(event_loaded, 'ev'), '_session_id'), 'EV-001')

    def test_unplug_event_json(self):
        _ = self._obj_compare_helper(self.unplug_event)

    def test_recompute_event_json(self):
        _ = self._obj_compare_helper(self.recompute_event)

    def test_event_queue_json(self):
        event_queue_loaded = self._obj_compare_helper(self.event_queue)
        for (ts, event), (tsl, event_loaded) in zip(
                self.event_queue._queue, event_queue_loaded._queue):
            self.assertEqual(ts, tsl)
            self.assertIsInstance(event_loaded, acnsim.Event)
            if not isinstance(event, acnsim.PluginEvent):
                self.assertEqual(type(event), type(event_loaded))
                self.assertEqual(event.__dict__, event_loaded.__dict__)
            else:
                for field in ['timestamp', 'event_type', 'precedence']:
                    self.assertEqual(
                        getattr(event, field), getattr(event_loaded, field))
                self.assertEqual(
                    getattr(getattr(event_loaded, 'ev'), '_session_id'),
                    getattr(getattr(event, 'ev'), '_session_id')
                )

    def test_charging_network_json(self):
        network_loaded = self._obj_compare_helper(self.network)
        self.assertIsInstance(network_loaded, acnsim.ChargingNetwork)

        network_np_fields = ['constraint_matrix', 'magnitudes', '_voltages',
                             '_phase_angles']
        for field in network_np_fields:
            np.testing.assert_equal(getattr(self.network, field),
                                    getattr(network_loaded, field))

        for (station_id, evse), (station_id_l, evse_l) in zip(
                self.network._EVSEs.items(), network_loaded._EVSEs.items()):
            self.assertEqual(station_id, station_id_l)
            self.assertEqual(evse.station_id, evse_l.station_id)

    def _sim_compare_helper(self, sim):
        simulator_loaded = self._obj_compare_helper(sim)

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
                acnsim.Event.from_json(json.load(infile))

    def test_numpy_version_inequality(self):
        with open(os.path.join(os.path.dirname(__file__),
                               'old_dependencies.json'),
                  'r') as infile:
            with self.assertWarns(UserWarning):
                acnsim.Event.from_json(json.load(infile))


class TestExtObjJSONIO(TestJSONIO):
    @classmethod
    def setUpClass(self):
        super().setUpClass()

        self.set_battery_event = SetAttrEvent(5)
        self.set_battery_event.set_extra_attr(self.battery1)

        self.event_queue.add_event(self.set_battery_event)

        self.simple_attributes['DefaultNamedEvent'] = [
            'timestamp', 'event_type', 'precedence'
        ]
        self.simple_attributes['SetAttrEvent'] = [
            'timestamp', 'event_type', 'precedence', 'extra_attr'
        ]
        self.simple_attributes['BatteryListEvent'] = [
            'timestamp', 'event_type', 'precedence'
        ]

    def test_named_event_json(self):
        named_event = NamedEvent(0, "my_event")
        with self.assertWarns(UserWarning):
            named_event_json = named_event.to_json()
        with self.assertRaises(TypeError):
            NamedEvent.from_json(named_event_json)

    def test_default_named_event_json(self):
        default_named_event = DefaultNamedEvent(5, "def_event")
        with self.assertWarns(UserWarning):
            default_named_event_loaded = self._obj_compare_helper(
                default_named_event)
        self.assertEqual(default_named_event_loaded.name, "my_event")

    def _obj_compare_helper_warning(self, obj, attributes=None):
        with self.assertWarns(UserWarning):
            obj_json = obj.to_json()
        obj_class = type(obj)
        with self.assertWarns(UserWarning):
            obj_loaded = obj_class.from_json(obj_json)
        self.assertIsInstance(obj_loaded, obj_class)
        if attributes is None:
            attributes = self.simple_attributes[obj_class.__name__]
        for attribute in attributes:
            self.assertEqual(getattr(obj, attribute),
                             getattr(obj_loaded, attribute))
        return obj_loaded

    def test_set_named_event_json(self):
        set_named_event = SetAttrEvent(5)
        set_named_event.set_extra_attr("set_event")
        _ = self._obj_compare_helper_warning(set_named_event)

    def test_set_list_event_json(self):
        set_list_event = SetAttrEvent(5)
        set_list_event.set_extra_attr(["set_event1", "set_event2"])
        _ = self._obj_compare_helper_warning(set_list_event)

    def test_set_battery_event_json(self):
        set_battery_event_loaded = self._obj_compare_helper_warning(
            self.set_battery_event, self.simple_attributes['Event'])
        self.assertEqual(self.battery1.__dict__,
                         set_battery_event_loaded.extra_attr.__dict__)

    def test_set_np_event_json(self):
        set_np_event = SetAttrEvent(5)
        set_np_event.set_extra_attr(np.zeros((2, 2)))
        set_np_event_loaded = self._obj_compare_helper_warning(
            set_np_event, self.simple_attributes['Event'])
        self.assertEqual(set_np_event_loaded.extra_attr,
                         "array([[0., 0.],\n       [0., 0.]])")

    def test_battery_list_event_json(self):
        battery_list_event = BatteryListEvent(
            5, [self.battery1, self.battery2])
        battery_list_event_loaded = self._obj_compare_helper(
            battery_list_event, self.simple_attributes['Event'])
        for battery, battery_loaded in zip(
                battery_list_event.battery_list,
                battery_list_event_loaded.battery_list
        ):
            self.assertEqual(battery.__dict__, battery_loaded.__dict__)

    def test_event_queue_json(self):
        event_queue_loaded = self._obj_compare_helper_warning(self.event_queue)
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
