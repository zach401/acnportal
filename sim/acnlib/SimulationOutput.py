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

    def get_charging_data(self):
        return self.charging_data

    def get_EV_charging_data(self, session_id):
        return self.charging_data[session_id]

    def get_all_events(self):
        return self.events

    def get_info_events(self):
        info = []
        for event in self.events:
            if event.type == 'INFO':
                info.append(event)
        return info

    def get_warning_events(self):
        warnings = []
        for event in self.events:
            if event.type == 'WARNING':
                warnings.append(event)
        return warnings

    def get_error_events(self):
        errors = []
        for event in self.events:
            if event.type == 'ERROR':
                errors.append(event)
        return errors

    def get_start_timestamp(self):
        return self.start_timestamp

    def get_period(self):
        return self.period

    def get_max_rate(self):
        return self.max_rate

    def get_voltage(self):
        return self.voltage

    def get_last_departure(self):
        return self.last_departure

    def get_all_EVs(self):
        return self.EVs

    def get_EV(self, session_id):
        return

    def get_all_EVSEs(self):
        return self.EVSEs

    def get_EVSE(self, station_id):
        return


class Event:

    def __init__(self, type, iteration, description, session='GLOBAL'):
        self.type = type
        self.iteration = iteration
        self.description = description
        self.session = session

