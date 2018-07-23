import copy

class SimulationOutput:

    def __init__(self, start_timestamp, period, max_rate, voltage):
        self.charging_data = {}
        self.events = []
        self.start_timestamp = start_timestamp
        self.period = period
        self.max_rate = max_rate
        self.voltage = voltage
        self.last_departure = 0
        self.EVs = []
        self.EVSEs = []

    def submit_charging_data(self, session_id, sample):
        if not session_id in self.charging_data:
            self.charging_data[session_id] = []
        self.charging_data[session_id].append(sample)

    def submit_event(self, event):
        self.events.append(event)

    def submit_all_EVs(self, EVs):
        self.EVs = copy.deepcopy(EVs)

    def submit_all_EVSEs(self, EVSEs):
        self.EVSEs = copy.deepcopy(EVSEs)


class Event:

    def __init__(self, type, iteration, description, session='GLOBAL'):
        self.type = type
        self.iteration = iteration
        self.description = description
        self.session = session

