import heapq


class EventQueue:
    def __init__(self):
        self._queue = []
        self._timestep = 0

    def empty(self):
        return len(self._queue) == 0

    def add_event(self, event):
        # if event.timestamp > self._timestep:
        heapq.heappush(self._queue, (event.timestamp, event))
        # else:
        #     raise ValueError('Cannot add event in the past.\n'
        #                      'Current time: {0}\nEvent time: {1}'.format(self._timestep, event.timestamp))

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