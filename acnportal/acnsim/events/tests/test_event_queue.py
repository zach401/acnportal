from unittest import TestCase

from acnportal.acnsim.events import EventQueue, Event


class TestEventQueue(TestCase):
    def setUp(self):
        self.events = EventQueue()

    def test_empty_on_init(self):
        self.assertTrue(self.events.empty())

    def test_add_event(self):
        self.events.add_event(Event(5))
        self.assertFalse(self.events.empty())
        self.assertEqual(len(self.events._queue), 1)

    def test_add_events(self):
        events = [Event(i) for i in range(1, 6)]
        self.events.add_events(events)
        self.assertFalse(self.events.empty())
        self.assertEqual(5, len(self.events._queue))

    def test_len(self):
        events = [Event(i) for i in range(1, 6)]
        self.events.add_events(events)
        self.assertEqual(4, len(self.events))

    def test_get_event(self):
        self.events.add_event(Event(5))
        self.assertEqual(len(self.events._queue), 1)
        e = self.events.get_event()
        self.assertTrue(self.events.empty())
        self.assertEqual(5, e.timestamp)

    def test_get_current_events(self):
        events = [Event(i) for i in range(1, 6)]
        self.events.add_events(events)
        curr_events = self.events.get_current_events(3)
        self.assertEqual(len(curr_events), 3)

    def test_get_last_timestamp(self):
        events = [Event(i) for i in range(1, 6)]
        self.events.add_events(events)
        self.assertEqual(5, self.events.get_last_timestamp())
        self.events.add_event(Event(8))
        self.assertEqual(8, self.events.get_last_timestamp())
        curr_events = self.events.get_current_events(3)
        self.assertEqual(8, self.events.get_last_timestamp())
