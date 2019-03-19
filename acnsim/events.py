import heapq


class EventQueue:
    """ Queue which stores simulation events."""

    def __init__(self):
        self._queue = []
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
            events (List[Event like]: A list of Event-like objects.

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


class Event:
    """ Base class for all events.

    Args:
        timestamp (int): Timestamp when an event occurs (periods)

    Attributes:
        timestamp (int): See args.
        type (str): Name of the event type.
        precedence (float): Used to order occurrence for events that happen in the same timestep. Higher precedence
            events occur before lower precedence events.

    """
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.type = ''
        self.precedence = float('inf')

    def __lt__(self, other):
        """ Return True if the precedence of self is less than that of other.

        Args:
            other (Event like): Another Event-like object.

        Returns:
            bool
        """
        return self.precedence < other.precedence


class PluginEvent(Event):
    """ Subclass of Event for EV plugins.

    Args:
        timestamp (int): See Event.
        ev (EV): The EV which will be plugged in.
    """
    def __init__(self, timestamp, ev):
        super().__init__(timestamp)
        self.type = 'Plugin'
        self.ev = ev
        self.precedence = 10


class UnplugEvent(Event):
    """ Subclass of Event for EV unplugs.

    Args:
        timestamp (int): See Event.
        station_id (str): ID of the EVSE where the EV is to be unplugged.
        session_id (str): ID of the session which should be ended.
    """
    def __init__(self, timestamp, station_id, session_id):
        super().__init__(timestamp)
        self.type = 'Unplug'
        self.station_id = station_id
        self.session_id = session_id
        self.precedence = 0


class RecomputeEvent(Event):
    """ Subclass of Event for when the algorithm should be recomputed."""
    def __init__(self, timestamp):
        super().__init__(timestamp)
        self.type = 'Recompute'
        self.precedence = 20
