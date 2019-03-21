import heapq
from .event import Event


class EventQueue:
    """ Queue which stores simulation events.

    Args:
        events (List[Event]): A list of Event-like objects.
    """

    def __init__(self, events=None):
        self._queue = []
        if events is not None:
            self.add_events(events)
        self._timestep = 0

    def empty(self):
        """ Return if the queue is empty.

        Returns:
            bool: True if the queue is empty.
        """
        return len(self._queue) == 0

    def add_event(self, event):
        """ Add an event to the queue.

        Args:
            event (Event like): An Event-like object.

        Returns:
            None
        """
        heapq.heappush(self._queue, (event.timestamp, event))

    def add_events(self, events):
        """ Add multiple events at a time to the queue.

        Args:
            events (List[Event like]): A list of Event-like objects.

        Returns:
            None
        """
        for e in events:
            self.add_event(e)

    def get_event(self):
        """ Return the next event in the queue.

        Returns:
            Event like: The next event in the queue.
        """
        return heapq.heappop(self._queue)[1]

    def get_current_events(self, timestep):
        """ Return all events occurring before or during timestep.

        Args:
            timestep (int): Time index in periods.

        Returns:
            List[Event like]: List of all events occurring before or during timestep.
        """
        self._timestep = timestep
        current_events = []
        while len(self._queue) > 0 and self._queue[0][0] <= self._timestep:
            current_events.append(self.get_event())
        return current_events
