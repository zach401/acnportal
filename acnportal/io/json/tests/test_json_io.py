from unittest import TestCase
from unittest.mock import create_autospec

from acnportal import acnsim
from acnportal.algorithms import BaseAlgorithm
from acnportal import io

import numpy as np

# class TestJSONIO(TestCase):
# 	@classmethod
# 	def setUpClass(self):
# 		self.battery = acnsim.models.Battery(100, 50, 20)
# 		self.battery._current_charging_power = 10
# 		self.ev = acnsim.models.EV(10, 20, 30, 'EV-001', 'SE-A', self.battery, estimated_departure=25)
# 		self.ev._energy_delivered = 50
# 		self.ev._current_charging_rate = 10
# 		self.evse = acnsim.models.EVSE('PS-001', max_rate=32, min_rate=0)
# 		self.evse.plugin(self.ev)
# 		self.evse.set_pilot(30, 220, 1)
# 		self.evse2 = acnsim.models.EVSE('PS-002')
# 		self.plugin_event = acnsim.events.PluginEvent(10, self.ev)
# 		self.unplug_event = acnsim.events.UnplugEvent(20, 'PS-001', 'SE-A')
# 		self.event_queue = acnsim.events.EventQueue()
# 		self.event_queue.add_events([self.plugin_event, self.unplug_event])
# 		self.network = acnsim.network.ChargingNetwork()
# 		self.network.register_evse(self.evse, 220, 30)
# 		self.network.register_evse(self.evse2, 220, 150)
# 		self.network.constraint_matrix = np.array([[1, 0], [0, 1]])
# 		self.network.magnitude = np.array([30, 30])
# 		self.network.constraint_index = ['C1', 'C2']
# 		self.simulator = acnsim.Simulator(self.network, create_autospec(BaseAlgorithm), self.event_queue, 20)

# 	def test_battery_json(self):
# 		battery_json = self.battery.to_json()
# 		self.battery_loaded = io.from_json(battery_json, typ='battery')
# 		self.assertIsInstance(self.battery_loaded, acnsim.models.Battery)
# 		self.assertEqual(self.battery_loaded.__dict__, self.battery.__dict__)

# 	def test_ev_json(self):
# 		ev_json = self.ev.to_json()
# 		self.ev_loaded = io.from_json(ev_json, typ='ev')
# 		self.assertIsInstance(ev_loaded, acnsim.models.EV)
# 		for field in ['arrival', 'departure', 'requested_energy', 'estimated_departure',
# 			'session_id', 'station_id', 'energy_delivered', 'current_charging_rate']:
# 			self.assertEqual(self.ev.getattr('_'+field), self.ev_loaded.getattr('_'+field))
# 		self.assertEqual(self.battery.__dict__, self.ev_loaded._battery.__dict__)

# 	def test_evse_json(self):
# 		evse_json = self.evse.to_json()
# 		self.evse_loaded = io.from_json(evse_json, typ='evse')
# 		evse_json2 = self.evse2.to_json()
# 		evse_loaded2 = io.from_json(evse_json2, typ='evse')
# 		self.assertIsInstance(self.evse_loaded, acnsim.models.EVSE)
# 		self.assertIsInstance(self.evse_loaded2, acnsim.models.EVSE)
# 		for field in ['station_id', 'max_rate', 'min_rate', 'current_pilot']:
# 			self.assertEqual(self.evse.getattr('_'+field), self.evse_loaded.getattr('_'+field))
# 			self.assertEqual(self.evse2.getattr('_'+field), self.evse_loaded2.getattr('_'+field))
# 		self.assertEqual(self.evse2.getattr('_ev'), None)