from unittest import TestCase

from acnportal.acnsim.events import EventQueue, Event


class TestEventAccessors(TestCase):
    def setUp(self):
        self.event = Event(5)
        self.assertEqual(self.event.event_type, "")

    def test_type_deprecation_warning(self):
        self.event = Event(5)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(self.event.type, "")


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
        self.assertEqual(5, len(self.events))

    def test_get_event(self):
        self.events.add_event(Event(5))
        self.assertEqual(len(self.events._queue), 1)
        e = self.events.get_event()
        self.assertTrue(self.events.empty())
        self.assertEqual(5, e.timestamp)

    def test_get_current_events_empty(self):
        events = []
        self.events.add_events(events)
        curr_events = self.events.get_current_events(3)
        self.assertEqual(curr_events, [])

    def test_get_current_events(self):
        events = [Event(i) for i in range(1, 6)]
        self.events.add_events(events)
        curr_events = self.events.get_current_events(3)
        self.assertEqual(len(curr_events), 3)
        for event, timestamp in zip(curr_events, [1, 2, 3]):
            self.assertEqual(event.timestamp, timestamp)

    def test_get_last_timestamp(self):
        events = [Event(i) for i in range(1, 6)]
        self.events.add_events(events)
        self.assertEqual(5, self.events.get_last_timestamp())
        self.events.add_event(Event(8))
        self.assertEqual(8, self.events.get_last_timestamp())
        # Check that the last timestamp is unaltered from a call to
        # get_current_events at an earlier timestamp.
        _ = self.events.get_current_events(3)
        self.assertEqual(8, self.events.get_last_timestamp())

    def test_get_last_timestamp_no_events(self):
        events = []
        self.events.add_events(events)
        self.assertIsNone(self.events.get_last_timestamp())
