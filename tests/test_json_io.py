import io
import json
from unittest import TestCase

from acnportal import acnsim
from acnportal.algorithms import BaseAlgorithm, UncontrolledCharging
from acnportal.acnsim.base import ErrorAllWrapper
from tests.serialization_extensions import NamedEvent, DefaultNamedEvent
from tests.serialization_extensions import SetAttrEvent, BatteryListEvent

import os
import numpy as np
from copy import deepcopy
from datetime import datetime


class TestJSONIO(TestCase):
    @classmethod
    def setUpClass(cls):
        # Make instance of each class in registry.
        # Battery
        cls.battery1 = acnsim.Battery(100, 50, 20)
        cls.battery1._current_charging_power = 10

        # Linear2StageBattery
        cls.battery2 = acnsim.Linear2StageBattery(100, 50, 20)
        cls.battery2._current_charging_power = 10
        cls.battery2._noise_level = 0.1
        cls.battery2._transition_soc = 0.85

        # EVs
        staying_time = 10
        ev1_arrival = 10
        cls.ev1 = acnsim.EV(
            ev1_arrival,
            ev1_arrival + staying_time,
            30,
            "PS-001",
            "EV-001",
            deepcopy(cls.battery1),
            estimated_departure=35,
        )
        cls.ev1._energy_delivered = 0.05
        cls.ev1._current_charging_rate = 10

        ev2_arrival = 40
        cls.ev2 = acnsim.EV(
            ev2_arrival,
            ev2_arrival + staying_time,
            30,
            "PS-002",
            "EV-002",
            deepcopy(cls.battery2),
            estimated_departure=65,
        )
        cls.ev2._energy_delivered = 0.05
        cls.ev2._current_charging_rate = 10

        ev3_arrival = 50
        cls.ev3 = acnsim.EV(
            ev3_arrival,
            ev3_arrival + staying_time,
            30,
            "PS-003",
            "EV-003",
            deepcopy(cls.battery2),
            estimated_departure=75,
        )
        cls.ev3._energy_delivered = 0.05
        cls.ev3._current_charging_rate = 10

        # EVSEs
        cls.evse0 = acnsim.EVSE("PS-000", max_rate=32)

        cls.evse1 = acnsim.EVSE("PS-001", max_rate=32)
        cls.evse1.plugin(cls.ev1)
        cls.evse1.set_pilot(30, 220, 1)

        cls.evse2 = acnsim.DeadbandEVSE("PS-002", max_rate=32, deadband_end=4)
        cls.evse2.plugin(cls.ev2)
        cls.evse2.set_pilot(30, 220, 1)

        cls.evse3 = acnsim.FiniteRatesEVSE("PS-003", allowable_rates=[0, 8, 16, 24, 32])
        cls.evse3.plugin(cls.ev3)
        cls.evse3.set_pilot(24, 220, 1)

        # Events
        cls.event = acnsim.Event(0)
        cls.plugin_event1 = acnsim.PluginEvent(10, cls.ev1)
        cls.unplug_event = acnsim.UnplugEvent(20, cls.ev1)
        cls.recompute_event1 = acnsim.RecomputeEvent(30)
        cls.plugin_event2 = acnsim.PluginEvent(40, cls.ev2)
        cls.plugin_event3 = acnsim.PluginEvent(50, cls.ev3)
        # Modify a default attribute to check if it's loaded correctly.
        cls.recompute_event2 = acnsim.RecomputeEvent(10)
        cls.recompute_event2.event_type = "Recompute Modified"

        # EventQueue
        cls.event_queue = acnsim.EventQueue()
        cls.event_queue.add_events(
            [
                cls.event,
                cls.plugin_event1,
                cls.recompute_event1,
                cls.plugin_event2,
                cls.plugin_event3,
            ]
        )

        # Network
        cls.network = acnsim.ChargingNetwork(
            violation_tolerance=1e-3, relative_tolerance=1e-5
        )

        cls.network.register_evse(cls.evse1, 220, 30)
        cls.network.register_evse(cls.evse2, 220, 150)
        cls.network.register_evse(cls.evse3, 220, -90)
        cls.network.constraint_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        cls.network.magnitudes = np.array([32, 32, 32])
        cls.network.constraint_index = ["C1", "C2", "C3"]
        _ = cls.network._update_info_store()
        cls.empty_network = acnsim.ChargingNetwork()

        # Simulator
        cls.simulator = acnsim.Simulator(
            cls.network,
            UncontrolledCharging(),
            cls.event_queue,
            datetime(2019, 1, 1),
            verbose=False,
            store_schedule_history=True,
        )

        # Make a copy of the simulator to run
        cls.simulator_run = deepcopy(cls.simulator)
        # Do necessary unplugs.
        for evse in cls.simulator_run.network._EVSEs.values():
            if evse.ev is not None:
                evse.unplug()
        # Run simulation
        cls.simulator_run.run()

        # Make a copy of the simulator with signals
        cls.simulator_signal = deepcopy(cls.simulator)
        cls.simulator_signal.signals = {"a": [0, 1, 2], "b": [3, 4]}

        cls.simulator_hard_signal = deepcopy(cls.simulator)
        cls.simulator_hard_signal.signals = {"a": BaseAlgorithm()}

        cls.simulator_no_sch_hist = deepcopy(cls.simulator)
        cls.simulator_no_sch_hist.schedule_history = None

        # Class data used for testing.
        cls.simple_attributes = {
            "Battery": [
                "_max_power",
                "_current_charging_power",
                "_current_charge",
                "_capacity",
                "_init_charge",
            ],
            "Linear2StageBattery": [
                "_max_power",
                "_current_charging_power",
                "_current_charge",
                "_capacity",
                "_init_charge",
                "_noise_level",
                "_transition_soc",
            ],
            "EV": [
                "_arrival",
                "_departure",
                "_requested_energy",
                "_estimated_departure",
                "_session_id",
                "_station_id",
                "_energy_delivered",
                "_current_charging_rate",
            ],
            "EVSE": [
                "_station_id",
                "_max_rate",
                "_min_rate",
                "_current_pilot",
                "is_continuous",
            ],
            "DeadbandEVSE": [
                "_station_id",
                "_max_rate",
                "_current_pilot",
                "_deadband_end",
                "is_continuous",
            ],
            "FiniteRatesEVSE": [
                "_station_id",
                "_current_pilot",
                "is_continuous",
                "allowable_rates",
            ],
            "Event": ["timestamp", "event_type", "precedence"],
            "PluginEvent": ["timestamp", "event_type", "precedence"],
            "UnplugEvent": ["timestamp", "event_type", "precedence"],
            "RecomputeEvent": ["timestamp", "event_type", "precedence"],
            "EventQueue": ["_timestep"],
            "ChargingNetwork": [
                "constraint_index",
                "violation_tolerance",
                "relative_tolerance",
                "_station_ids_dict",
            ],
            "Simulator": [
                "period",
                "max_recompute",
                "verbose",
                "peak",
                "_iteration",
                "_resolve",
                "_last_schedule_update",
                "schedule_history",
            ],
        }

    def _obj_compare_helper(self, obj, attributes=None):
        obj_json = obj.to_json()
        obj_class = type(obj)
        obj_loaded = obj_class.from_json(obj_json)
        self.assertIsInstance(obj_loaded, obj_class)
        if attributes is None:
            attributes = self.simple_attributes[obj_class.__name__]
        for attribute in attributes:
            self.assertEqual(getattr(obj, attribute), getattr(obj_loaded, attribute))
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
            self.assertEqual(ev_loaded._battery.__dict__, ev._battery.__dict__)

        _load_dump_compare_helper(self.ev1, acnsim.Battery)
        _load_dump_compare_helper(self.ev2, acnsim.Linear2StageBattery)
        _load_dump_compare_helper(self.ev3, acnsim.Linear2StageBattery)

    def test_evse_no_ev_json(self):
        evse_loaded = self._obj_compare_helper(self.evse0)
        self.assertEqual(getattr(evse_loaded, "_ev"), None)

    def test_evse_json(self):
        evse_loaded = self._obj_compare_helper(self.evse1)
        self.assertEqual(getattr(getattr(evse_loaded, "_ev"), "_session_id"), "EV-001")

    def test_deadband_evse_json(self):
        evse_loaded = self._obj_compare_helper(self.evse2)
        self.assertEqual(getattr(getattr(evse_loaded, "_ev"), "_session_id"), "EV-002")

    def test_finite_rates_evse_json(self):
        evse_loaded = self._obj_compare_helper(self.evse3)
        self.assertEqual(getattr(getattr(evse_loaded, "_ev"), "_session_id"), "EV-003")

    def test_event_json(self):
        _ = self._obj_compare_helper(self.event)

    def test_plugin_event_json(self):
        event_loaded = self._obj_compare_helper(self.plugin_event1)
        self.assertEqual(getattr(getattr(event_loaded, "ev"), "_session_id"), "EV-001")

    def test_unplug_event_json(self):
        event_loaded = self._obj_compare_helper(self.unplug_event)
        self.assertEqual(getattr(getattr(event_loaded, "ev"), "_session_id"), "EV-001")

    def test_recompute_event_json(self):
        _ = self._obj_compare_helper(self.recompute_event1)

    def test_altered_recompute_event_json(self):
        _ = self._obj_compare_helper(self.recompute_event2)

    def test_event_queue_json(self):
        event_queue_loaded = self._obj_compare_helper(self.event_queue)
        for (ts, event), (tsl, event_loaded) in zip(
            self.event_queue._queue, event_queue_loaded._queue
        ):
            self.assertEqual(ts, tsl)
            self.assertIsInstance(event_loaded, acnsim.Event)
            if not isinstance(event, acnsim.PluginEvent):
                self.assertEqual(type(event), type(event_loaded))
                self.assertEqual(event.__dict__, event_loaded.__dict__)
            else:
                for field in ["timestamp", "event_type", "precedence"]:
                    self.assertEqual(
                        getattr(event, field), getattr(event_loaded, field)
                    )
                self.assertEqual(
                    getattr(getattr(event_loaded, "ev"), "_session_id"),
                    getattr(getattr(event, "ev"), "_session_id"),
                )

    def test_empty_charging_network_json(self):
        empty_network_loaded = self._obj_compare_helper(self.empty_network)
        self.assertIsInstance(empty_network_loaded, acnsim.ChargingNetwork)

        network_np_fields = [
            "magnitudes",
            "_voltages",
            "_phase_angles",
            "max_pilot_signals",
            "min_pilot_signals",
            "is_continuous",
        ]
        for field in network_np_fields:
            np.testing.assert_equal(
                getattr(self.empty_network, field), getattr(empty_network_loaded, field)
            )
        extra_simple_attributes = ["constraint_matrix", "_EVSEs", "allowable_rates"]
        for attribute in extra_simple_attributes:
            self.assertEqual(
                getattr(self.empty_network, attribute),
                getattr(empty_network_loaded, attribute),
            )

    def test_charging_network_json(self):
        network_loaded = self._obj_compare_helper(self.network)
        self.assertIsInstance(network_loaded, acnsim.ChargingNetwork)

        network_np_fields = [
            "constraint_matrix",
            "magnitudes",
            "_voltages",
            "_phase_angles",
            "max_pilot_signals",
            "min_pilot_signals",
            "is_continuous",
        ]
        for field in network_np_fields:
            np.testing.assert_equal(
                getattr(self.network, field), getattr(network_loaded, field)
            )

        for (station_id, evse), (station_id_l, evse_l) in zip(
            self.network._EVSEs.items(), network_loaded._EVSEs.items()
        ):
            self.assertEqual(station_id, station_id_l)
            self.assertEqual(evse.station_id, evse_l.station_id)

        for allowable_rates_array, allowable_rates_array_loaded in zip(
            self.network.allowable_rates, network_loaded.allowable_rates
        ):
            np.testing.assert_equal(allowable_rates_array, allowable_rates_array_loaded)

    def _sim_compare_helper(self, sim):
        simulator_loaded = self._obj_compare_helper(sim)

        self.assertEqual(sim.start, simulator_loaded.start)

        self.assertIsInstance(simulator_loaded.scheduler, UncontrolledCharging)

        if simulator_loaded.signals is not None:
            self.assertEqual(sim.signals, simulator_loaded.signals)

        np.testing.assert_equal(sim.pilot_signals, simulator_loaded.pilot_signals)
        np.testing.assert_equal(sim.charging_rates, simulator_loaded.charging_rates)

        network_attrs = [
            "station_ids",
            "active_station_ids",
            "voltages",
            "phase_angles",
        ]
        for attr in network_attrs:
            self.assertEqual(
                getattr(sim.network, attr), getattr(simulator_loaded.network, attr)
            )

        for (ts, event), (tsl, event_loaded) in zip(
            sim.event_queue._queue, simulator_loaded.event_queue._queue
        ):
            self.assertEqual(ts, tsl)
            self.assertEqual(event.event_type, event_loaded.event_type)
            self.assertEqual(event.timestamp, event_loaded.timestamp)

        for event, event_loaded in zip(
            sim.event_history, simulator_loaded.event_history
        ):
            self.assertEqual(event.event_type, event_loaded.event_type)
            self.assertEqual(event.timestamp, event_loaded.timestamp)

        for station_id, ev in sim.ev_history.items():
            self.assertEqual(
                ev.session_id, simulator_loaded.ev_history[station_id].session_id
            )

    def test_init_simulator_json(self):
        self._sim_compare_helper(self.simulator)

    def test_run_simulator_json(self):
        self._sim_compare_helper(self.simulator_run)

    def test_object_equalities(self):
        # Tests that certain object equality invariants are preserved
        # after loading.
        simulator_json = self.simulator_run.to_json()
        simulator_loaded = acnsim.Simulator.from_json(simulator_json)

        plugins = filter(
            lambda x: isinstance(x, acnsim.PluginEvent), simulator_loaded.event_history
        )
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
        with self.assertWarns(UserWarning):
            acnsim.Event.from_json(
                os.path.join(os.path.dirname(__file__), "old_version.json")
            )

    def test_numpy_version_inequality(self):
        with self.assertWarns(UserWarning):
            acnsim.Event.from_json(
                os.path.join(os.path.dirname(__file__), "old_dependencies.json")
            )


class TestExtObjJSONIO(TestJSONIO):
    battery1 = None
    event_queue = None
    simple_attributes = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.set_battery_event = SetAttrEvent(5)
        cls.set_battery_event.set_extra_attr(cls.battery1)

        cls.event_queue.add_event(cls.set_battery_event)

        cls.simple_attributes["DefaultNamedEvent"] = [
            "timestamp",
            "event_type",
            "precedence",
        ]
        cls.simple_attributes["SetAttrEvent"] = [
            "timestamp",
            "event_type",
            "precedence",
            "extra_attr",
        ]
        cls.simple_attributes["BatteryListEvent"] = [
            "timestamp",
            "event_type",
            "precedence",
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
            default_named_event_loaded = self._obj_compare_helper(default_named_event)
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
            self.assertEqual(getattr(obj, attribute), getattr(obj_loaded, attribute))
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
            self.set_battery_event, self.simple_attributes["Event"]
        )
        self.assertEqual(
            self.battery1.__dict__, set_battery_event_loaded.extra_attr.__dict__
        )

    def test_set_np_event_json(self):
        set_np_event = SetAttrEvent(5)
        set_np_event.set_extra_attr(np.zeros((2, 2)))
        set_np_event_loaded = self._obj_compare_helper_warning(
            set_np_event, self.simple_attributes["Event"]
        )
        self.assertIsInstance(set_np_event_loaded.extra_attr, ErrorAllWrapper)
        self.assertEqual(
            set_np_event_loaded.extra_attr.data, "array([[0., 0.],\n       [0., 0.]])"
        )

    def test_battery_list_event_json(self):
        battery_list_event = BatteryListEvent(5, [self.battery1, self.battery2])
        battery_list_event_loaded = self._obj_compare_helper(
            battery_list_event, self.simple_attributes["Event"]
        )
        for battery, battery_loaded in zip(
            battery_list_event.battery_list, battery_list_event_loaded.battery_list
        ):
            self.assertEqual(battery.__dict__, battery_loaded.__dict__)

    def test_event_queue_json(self):
        event_queue_loaded = self._obj_compare_helper_warning(self.event_queue)
        for (ts, event), (tsl, event_loaded) in zip(
            self.event_queue._queue, event_queue_loaded._queue
        ):
            self.assertEqual(ts, tsl)
            self.assertIsInstance(event_loaded, acnsim.Event)
            if isinstance(event_loaded, acnsim.PluginEvent):
                for field in ["timestamp", "event_type", "precedence"]:
                    self.assertEqual(
                        getattr(event, field), getattr(event_loaded, field)
                    )
                self.assertEqual(
                    getattr(getattr(event_loaded, "ev"), "_session_id"),
                    getattr(getattr(event, "ev"), "_session_id"),
                )
            elif isinstance(event_loaded, SetAttrEvent):
                for field in ["timestamp", "event_type", "precedence"]:
                    self.assertEqual(
                        getattr(event, field), getattr(event_loaded, field)
                    )
                self.assertEqual(
                    getattr(event_loaded, "extra_attr").__dict__,
                    getattr(event, "extra_attr").__dict__,
                )
            else:
                self.assertEqual(type(event), type(event_loaded))
                self.assertEqual(event.__dict__, event_loaded.__dict__)

    def test_init_simulator_json(self):
        with self.assertWarns(UserWarning):
            self._sim_compare_helper(self.simulator)


class TestJSONIOTypes(TestJSONIO):
    @classmethod
    def setUpClass(cls):
        cls.battery = acnsim.Battery(100, 50, 20)

    def test_to_json_string(self):
        battery_json_string = self.battery.to_json()
        self.assertIsInstance(battery_json_string, str)
        battery_loaded = acnsim.Battery.from_json(battery_json_string)
        self.assertEqual(self.battery.__dict__, battery_loaded.__dict__)

    def test_to_json_filepath(self):
        filepath = os.path.join(os.path.dirname(__file__), "battery_test_filepath.json")
        battery_json_filepath = self.battery.to_json(filepath)
        self.assertIsNone(battery_json_filepath)
        battery_loaded = acnsim.Battery.from_json(filepath)
        self.assertEqual(self.battery.__dict__, battery_loaded.__dict__)
        # Clear the file so that this test doesn't cause the package
        # state to change.
        with open(filepath, "w") as file_handle:
            json.dump({}, file_handle)
            file_handle.write("\n")

    def test_to_json_file_handle(self):
        filepath = os.path.join(
            os.path.dirname(__file__), "battery_test_file_handle.json"
        )
        file_handle = open(filepath, "w")
        battery_json_filepath = self.battery.to_json(file_handle)
        file_handle.close()
        self.assertIsNone(battery_json_filepath)
        file_handle = open(filepath)
        battery_loaded = acnsim.Battery.from_json(file_handle)
        file_handle.close()
        self.assertEqual(self.battery.__dict__, battery_loaded.__dict__)
        # Clear the file so that this test doesn't cause the package
        # state to change.
        with open(filepath, "w") as file_handle:
            json.dump({}, file_handle)
            file_handle.write("\n")

    def test_to_json_str_io(self):
        output = io.StringIO()
        battery_json_string = self.battery.to_json(output)
        self.assertIsNone(battery_json_string)
        self.assertIsInstance(output.getvalue(), str)
        output_str = output.getvalue()
        output.close()
        input_str_io = io.StringIO(output_str)
        battery_loaded = acnsim.Battery.from_json(input_str_io)
        self.assertEqual(self.battery.__dict__, battery_loaded.__dict__)


class TestLegacyObjJSONInput(TestCase):
    def test_legacy_unplug_event_json(self) -> None:
        """ Tests that UnplugEvents from <0.2.2 can be loaded.

        In acnportal v0.2.2, UnplugEvent had session_id and station_id attributes
        instead of an ev attribute.

        Returns:
            None
        """
        with self.assertWarns(UserWarning):
            unplug_loaded: acnsim.UnplugEvent = acnsim.UnplugEvent.from_json(
                os.path.join(os.path.dirname(__file__), "old_unplug.json")
            )
        self.assertIsInstance(unplug_loaded, acnsim.UnplugEvent)
        self.assertEqual(unplug_loaded.event_type, "Unplug")
        self.assertEqual(unplug_loaded.timestamp, 11)
        self.assertEqual(unplug_loaded.precedence, 0)

        # Check that UnplugEvent's EV is partially loaded
        self.assertEqual(getattr(getattr(unplug_loaded, "ev"), "_session_id"), "EV-001")
        self.assertEqual(getattr(getattr(unplug_loaded, "ev"), "_station_id"), "PS-001")

        for attribute in [
            "arrival",
            "departure",
            "requested_energy",
            "estimated_departure",
            "battery",
            "energy_delivered",
            "current_charging_rate",
        ]:
            with self.assertRaises(AttributeError):
                getattr(unplug_loaded.ev, attribute)
