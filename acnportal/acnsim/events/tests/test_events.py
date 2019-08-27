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

    def test_execute_calls_arrive(self):
        self.sim.network.arrive.assert_called_once_with(self.ev)

    def test_execute_adds_ev_to_history(self):
        self.assertIn(self.ev.session_id, self.sim.ev_history)
        self.assertEqual(self.sim.ev_history[self.ev.session_id], self.ev)

    def test_execute_adds_departure_event(self):
        event_added = self.sim.event_queue.add_event.call_args[0][0]
        self.assertIsInstance(event_added, Departure)
        self.assertEqual(event_added.timestamp, self.ev.departure)
        self.assertEqual(event_added.ev, self.ev)

    def test_recompute_flag_set(self):
        self.assertTrue(self.sim._resolve)


class TestDeparture(TestCase):
    def setUp(self):
        self.sim = create_autospec(Simulator)
        self.sim.network = create_autospec(ChargingNetwork)
        self.ev = create_autospec(EV)
        self.ev.departure = 10
        self.d = Departure(self.ev.departure, self.ev)
        self.d.execute(self.sim)

    def test_execute_calls_depart(self):
        self.sim.network.depart.assert_called_once_with(self.ev)

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
