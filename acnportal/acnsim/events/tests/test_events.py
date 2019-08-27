from unittest import TestCase
from unittest.mock import create_autospec

from acnportal.acnsim.events import EventQueue, Event, Arrival, Departure, Recompute
from acnportal.acnsim import Simulator, ChargingNetwork
from acnportal.acnsim.models import EV

class TestEvent(TestCase):
    def test_less_than(self):
        e1 = Event(0)
        e1.precedence = 5
        e2 = Event(1)
        e2.precedence = 6
        self.assertLess(e1, e2)


class TestArrival(TestCase):
    def setUp(self):
        self.sim = create_autospec(Simulator)
        self.sim.network = create_autospec(ChargingNetwork)
        self.sim.event_queue = create_autospec(EventQueue)
        self.sim.ev_history = {}
        self.ev = create_autospec(EV)
        self.ev.arrival = 5
        self.a = Arrival(self.ev.arrival, self.ev)
        self.a.execute(self.sim)

    def test_execute_calls_plugin(self):
        self.sim.network.plugin.assert_called_once_with(self.ev, self.ev.station_id)

    def test_execute_adds_ev_to_history(self):
        self.assertIn(self.ev.session_id, self.sim.ev_history)
        self.assertEqual(self.sim.ev_history[self.ev.session_id], self.ev)

    def test_execute_adds_departure_event(self):
        event_added = self.sim.event_queue.add_event.call_args[0][0]
        self.assertIsInstance(event_added, Departure)
        self.assertEqual(event_added.timestamp, self.ev.departure)
        self.assertEqual(event_added.session_id, self.ev.session_id)
        self.assertEqual(event_added.station_id, self.ev.station_id)

    def test_recompute_flag_set(self):
        self.assertTrue(self.sim._resolve)


class TestDeparture(TestCase):
    def setUp(self):
        self.sim = create_autospec(Simulator)
        self.sim.network = create_autospec(ChargingNetwork)
        self.d = Departure(10, 'station_id', 'session_id')
        self.d.execute(self.sim)

    def test_execute_calls_unplug(self):
        self.sim.network.unplug.assert_called_once_with('station_id')

    def test_recompute_flag_set(self):
        self.assertTrue(self.sim._resolve)


class TestRecompute(TestCase):
    def setUp(self):
        self.sim = create_autospec(Simulator)
        self.sim.network = create_autospec(ChargingNetwork)
        self.r = Recompute(10)
        self.r.execute(self.sim)

    def test_recompute_flag_set(self):
        self.assertTrue(self.sim._resolve)
