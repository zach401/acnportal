import heapq

class EventQueue:
    def __init__(self):
        self.queue = []
        self.timestep = 0

    def empty(self):
        return len(self.queue) == 0

    def add_event(self, event):
        if event.timestamp > self.timestep:
            heapq.heappush(self.queue, (event.timestamp, event))
        else:
            pass #  TODO(zlee): raise error here

    def add_events(self, events):
        for e in events:
            self.add_event(e)

    def get_event(self):
        return heapq.heappop(self.queue)[1]

    def get_current_events(self, timestep):
        self.timestep = timestep
        current_events = []
        while len(self.queue) > 0 and self.queue[0][0] <= self.timestep:
            current_events.append(self.get_event())
        # TODO(zlee): sort current events (Plugin, Unplug, Recompute)
        return current_events


class Event:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.type = ''
        self.precedence = float('inf')

    def __lt__(self, other):
        return self.precedence < other.precedence


class PluginEvent(Event):
    def __init__(self, timestamp, ev_params):
        super().__init__(timestamp)
        self.type = 'Plugin'
        self.ev_params = ev_params
        self.precedence = 10


class UnplugEvent(Event):
    def __init__(self, timestamp, station_id, session_id):
        super().__init__(timestamp)
        self.type = 'Unplug'
        self.station_id = station_id
        self.session_id = session_id
        self.precedence = 0


class RecomputeEvent(Event):
    def __init__(self, timestamp, period=None):
        super().__init__(timestamp)
        self.type = 'Recompute'
        self.period = period
        self.precedence = 20