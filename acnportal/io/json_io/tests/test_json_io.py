from unittest import TestCase
from unittest.mock import create_autospec

from acnportal import acnsim
from acnportal.algorithms import BaseAlgorithm
from acnportal import io

import numpy as np
from copy import deepcopy
from datetime import datetime

# TODO: Integration test that completely compares simulator obj
# before and after loading.
# TODO: Call signature of from_json: Class.from_json()
# TODO: Equality of simulator objects.
# TODO: Repr for simulator objects.
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
        self.plugin_event = acnsim.PluginEvent(10, self.ev1)
        self.unplug_event = acnsim.UnplugEvent(20, 'PS-001', 
            'EV-001')
        self.recompute_event = acnsim.RecomputeEvent(30)

        # EventQueue
        self.event_queue = acnsim.EventQueue()
        self.event_queue.add_events([self.event, self.plugin_event, 
            self.unplug_event, self.recompute_event])
        
        # Network
        self.network = acnsim.ChargingNetwork()
        self.network.register_evse(self.evse1, 220, 30)
        self.network.register_evse(self.evse2, 220, 150)
        self.network.register_evse(self.evse3, 220, -90)
        self.network.constraint_matrix = np.array([[1, 0], [0, 1]])
        self.network.magnitude = np.array([30, 30])
        self.network.constraint_index = ['C1', 'C2']

        # Simulator
        self.simulator = acnsim.Simulator(
            self.network, BaseAlgorithm(), 
            self.event_queue, datetime(2019, 1, 1)
        )

    def test_battery_json(self):
        battery_json = io.to_json(self.battery1)
        battery_loaded = io.from_json(battery_json)
        self.assertIsInstance(
            battery_loaded, acnsim.Battery)
        self.assertEqual(
            battery_loaded.__dict__, self.battery1.__dict__)

    def test_linear_2_stage_battery_json(self):
        battery_json = io.to_json(self.battery2)
        battery_loaded = io.from_json(battery_json)
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
            ev_loaded = io.from_json(ev_json)
            self.assertIsInstance(ev_loaded, acnsim.EV)
            for field in ev_fields:
                self.assertEqual(getattr(ev, field), 
                    getattr(ev_loaded, field))
            self.assertIsInstance(ev_loaded._battery, bat_type)
            self.assertEqual(ev._battery.__dict__,
                ev_loaded._battery.__dict__)

        _load_dump_compare_helper(self.ev1, acnsim.Battery)
        _load_dump_compare_helper(self.ev2, 
            acnsim.Linear2StageBattery)
        _load_dump_compare_helper(self.ev3, 
            acnsim.Linear2StageBattery)

    def test_evse_json(self):
        evse_json = self.evse1.to_json()
        evse_loaded = io.from_json(evse_json)
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
        evse_loaded = io.from_json(evse_json)
        self.assertIsInstance(evse_loaded, 
            acnsim.DeadbandEVSE)

        evse_fields = ['_station_id', '_max_rate',  '_min_rate',
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
        evse_loaded = io.from_json(evse_json)
        self.assertIsInstance(evse_loaded, 
            acnsim.FiniteRatesEVSE)

        evse_fields = ['_station_id', '_max_rate',  '_min_rate',
            '_current_pilot', 'is_continuous','allowable_rates']

        for field in evse_fields:
            self.assertEqual(getattr(self.evse3, field), 
                getattr(evse_loaded, field))
        self.assertEqual(
            getattr(getattr(evse_loaded, '_ev'), '_session_id'), 
            'EV-003'
        )

    def test_event_json(self):
        event_json = self.event.to_json()
        event_loaded = io.from_json(event_json)
        self.assertIsInstance(event_loaded, acnsim.Event)

        event_fields = ['timestamp', 'type', 'precedence']

        for field in event_fields:
            self.assertEqual(getattr(self.event, field), 
                getattr(event_loaded, field))

    def test_plugin_event_json(self):
        event_json = self.plugin_event.to_json()
        event_loaded = io.from_json(event_json)
        self.assertIsInstance(event_loaded, acnsim.PluginEvent)

        event_fields = ['timestamp', 'type', 'precedence']

        for field in event_fields:
            self.assertEqual(getattr(self.plugin_event, field), 
                getattr(event_loaded, field))
        self.assertEqual(
            getattr(getattr(event_loaded, 'ev'), '_session_id'), 
            'EV-001'
        )

    def test_unplug_event_json(self):
        event_json = self.unplug_event.to_json()
        event_loaded = io.from_json(event_json)
        self.assertIsInstance(event_loaded, acnsim.UnplugEvent)

        event_fields = ['timestamp', 'type', 'precedence',
            'station_id', 'session_id']

        for field in event_fields:
            self.assertEqual(getattr(self.unplug_event, field), 
                getattr(event_loaded, field))

    def test_recompute_event_json(self):
        event_json = self.recompute_event.to_json()
        event_loaded = io.from_json(event_json)
        self.assertIsInstance(event_loaded, acnsim.RecomputeEvent)

        event_fields = ['timestamp', 'type', 'precedence']

        for field in event_fields:
            self.assertEqual(getattr(self.recompute_event, field), 
                getattr(event_loaded, field))

    def test_event_queue_json(self):
        event_queue_json = self.event_queue.to_json()
        event_queue_loaded = io.from_json(event_queue_json)
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
                for field in ['timestamp', 'type', 'precedence']:
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
        network_loaded = io.from_json(network_json)
        self.assertIsInstance(network_loaded, acnsim.ChargingNetwork)

        network_np_fields = ['constraint_matrix', 'magnitudes', 
            '_voltages', '_phase_angles']
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

    def test_simulator_json(self):
        simulator_json = self.simulator.to_json()
        simulator_loaded = io.from_json(simulator_json)
        self.assertIsInstance(simulator_loaded, acnsim.Simulator)

        simulator_fields = ['period', 'max_recompute', 'verbose',
            'peak', '_iteration', '_resolve', 'start',
            '_last_schedule_update']

        for field in simulator_fields:
            self.assertEqual(getattr(self.simulator, field), 
                getattr(simulator_loaded, field))

        # self.assertEqual(repr(self.simulator.scheduler),
        #     simulator_loaded.scheduler)

        np.testing.assert_equal(self.simulator.pilot_signals,
            simulator_loaded.pilot_signals)
        np.testing.assert_equal(self.simulator.charging_rates,
            simulator_loaded.charging_rates)

        # TODO: better proxy for network equality
        network_attrs = ['station_ids', 'active_station_ids', 
            'voltages', 'phase_angles']
        for attr in network_attrs:
            self.assertEqual(getattr(self.simulator.network, attr),
                getattr(simulator_loaded.network, attr))

        # TODO: better proxy for event queue equality
        # This only checks timestep and type
        for (ts, event), (tsl, event_loaded) in \
            zip(self.simulator.event_queue._queue, 
                simulator_loaded.event_queue._queue):
            self.assertEqual(ts, tsl)
            self.assertEqual(event.type, event_loaded.type)
            self.assertEqual(event.timestamp, event_loaded.timestamp)

class TestJSONIOObjExtension(TestCase):
    class HalfBattery(acnsim.Battery):
        pass


    def test_non_nested_obj_extension(self):
        pass