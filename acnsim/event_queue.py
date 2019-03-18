import heapq


class EventQueue:
    """
    Queue used to store events for the simulation.

    :ivar list((int, Event)) _queue: heap used to store events. The heap invariant is based on the event timestamp.
    :ivar int _timestep: last timestep for which events have been popped.
    """

    def __init__(self):
        self._queue = []
        self._timestep = 0

    def empty(self):
        return len(self._queue) == 0

    def add_event(self, event):
        heapq.heappush(self._queue, (event.timestamp, event))

    def add_events(self, events):
        for e in events:
            self.add_event(e)

    def get_event(self):
        return heapq.heappop(self._queue)[1]

    def get_current_events(self, timestep):
        self._timestep = timestep
        current_events = []
        while len(self._queue) > 0 and self._queue[0][0] <= self._timestep:
            current_events.append(self.get_event())
        return current_events


class Event:
    """
    Base class for all events.

    :ivar int timestamp: timestamp (in simulation periods) when the event occurs.
    :ivar str type: name of the event type.
    :ivar float precedence: importance of the event. Used to order occurance for events that happen in the same timestep.

    """

    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.type = ''
        self.precedence = float('inf')

    def __lt__(self, other):
        return self.precedence < other.precedence


class PluginEvent(Event):
    def __init__(self, timestamp, ev):
        super().__init__(timestamp)
        self.type = 'Plugin'
        self.EV = ev
        self.precedence = 10


class UnplugEvent(Event):
    def __init__(self, timestamp, station_id, session_id):
        super().__init__(timestamp)
        self.type = 'Unplug'
        self.station_id = station_id
        self.session_id = session_id
        self.precedence = 0


class RecomputeEvent(Event):
    def __init__(self, timestamp):
        super().__init__(timestamp)
        self.type = 'Recompute'
        self.precedence = 20
