import copy

class SimulationOutput:

    def __init__(self, start_timestamp, period, max_rate, voltage):
        self.charging_data = {}
        self.network_data = []
        self.events = []
        self.start_timestamp = start_timestamp
        self.period = period
        self.max_rate = max_rate
        self.voltage = voltage
        self.last_departure = 0
        self.last_arrival = 0
        self.EVs = []
        self.EVSEs = []

    def submit_charging_data(self, session_id, sample):
        '''
        Submit a sample of charging data. The data will be stored in the simulation output as
        a dictionary of sessions described by the key and the sample stored in a list.

        The sample is a dict with the following keys:
        - time
        - charge_rate
        - pilot_signal
        - remaining_demand

        :param int session_id: The ID of the session to which the sample should be stored
        :param dict sample: A dict containing the sample
        :return: None
        '''
        if not session_id in self.charging_data:
            self.charging_data[session_id] = []
        self.charging_data[session_id].append(sample)

    def submit_network_data(self, sample):
        '''
        Submit a sample of charging data. The data will be stored in the simulation output as
        a list.

        The sample is a dict with the following keys
        - time
        - total_current
        - nbr_active_EVs

        :param dict sample: A dict containing the sample
        :return: None
        '''
        self.network_data.append(sample)

    def submit_event(self, event):
        '''
        Submit an event that will show information about the simulation.

        :param Event event: The event that will be saved
        :return: None
        '''
        self.events.append(event)

    def submit_all_EVs(self, EVs):
        '''
        Submit the EVs used in the simulation. This is done after the simulation is finished
        to be able to extract data of the EVs when analyzing the simulation output.

        :param list EVs: A list of the EVs used in the simulation.
        :return: None
        '''
        self.EVs = copy.deepcopy(EVs)

    def submit_all_EVSEs(self, EVSEs):
        '''
        Submit the EVSEs used in the simulation. This is done after the simulation is finished
        to be able to extract data of the EVSEs when analyzing the simulation output.

        :param list EVSEs: A list of the EVSEs used in the simulation.
        :return: None
        '''
        self.EVSEs = copy.deepcopy(EVSEs)

    def get_charging_data(self):
        '''
        Returns the charging data from the simulation. The data is stored as dict with
        key: ``session_id``, and value: ``list of samples (dicts)``

        :return: The charging data
        :rtype: dict
        '''
        return self.charging_data

    def get_EV_charging_data(self, session_id):
        '''
        Get the charging data of one EV described by its ``session_id``.

        :param string session_id: The ID of the session that the charging data should describe.
        :return: The charging data of the desired EV
        :rtype: list
        '''
        return self.charging_data[session_id]

    def get_events_all(self):
        '''
        Get all the submitted :class:`Events<sim.acnlib.SimulationOutput.Event>`

        :return: A list of all submitted events
        :rtype: list(Event)
        '''
        return self.events

    def get_events_info(self):
        '''
        Get all the submitted info :class:`Events<sim.acnlib.SimulationOutput.Event>`

        :return: A list of all submitted info events
        :rtype: list(Event)
        '''
        info = []
        for event in self.events:
            if event.type == 'INFO':
                info.append(event)
        return info

    def get_events_warnings(self):
        '''
        Get all the submitted warning :class:`Events<sim.acnlib.SimulationOutput.Event>`

        :return: A list of all submitted warning events
        :rtype: list(Event)
        '''
        warnings = []
        for event in self.events:
            if event.type == 'WARNING':
                warnings.append(event)
        return warnings

    def get_events_errors(self):
        '''
        Get all the submitted error :class:`Events<sim.acnlib.SimulationOutput.Event>`

        :return: A list of all submitted error events
        :rtype: list(Event)
        '''
        errors = []
        for event in self.events:
            if event.type == 'ERROR':
                errors.append(event)
        return errors

    def get_events_log(self):
        '''
        Get all the submitted log :class:`Events<sim.acnlib.SimulationOutput.Event>`

        :return: A list of all submitted log events
        :rtype: list(Event)
        '''
        logs = []
        for event in self.events:
            if event.type == 'LOG':
                logs.append(event)
        return logs

    def get_start_timestamp(self):
        '''
        Get the starting timestamp of the simulation.

        This timestamp is used to determine when the simulation is taking place as the simulation
        time (iteration) starts at 0.

        To get the real time (in UNIX time), use the following formula:
            UNIX time = start_timestamp + iteration * period * 60

        :return: The UNIX timestamp of the start of the simulation.
        :rtype: int
        '''
        return self.start_timestamp

    def get_period(self):
        '''
        Get the period time (in minutes) used in the simulation.

        :return: The simulation period
        :rtype: float
        '''
        return self.period

    def get_max_rate(self):
        '''
        Get the maximum charging rate allowed for an EV.

        :return: The maximum charging rate
        :rtype: float
        '''
        return self.max_rate

    def get_voltage(self):
        '''
        Get the grid voltage level

        :return: The voltage level
        :rtype: float
        '''
        return self.voltage

    def get_last_departure(self):
        '''
        Get the last departure time (in periods) of an EV charging at the ACN.

        :return: The last departure time
        :rtype: int
        '''
        return self.last_departure

    def get_last_arrival(self):
        '''
        Get the last arrival time (in periods) of an EV charging at the ACN.

        :return: The last arrival time
        :rtype: int
        '''
        return self.last_departure

    def get_all_EVs(self):
        '''
        Get all the :class:`EVs<sim.acnlib.EV.EV>` that have been using the ACN during this simulation.

        :return: List of EVs that have been using the ACN
        :rtype: list(EV)
        '''
        return self.EVs

    def get_EV(self, session_id):
        '''
        Get a specific EV that have been using the ACN during the simulation.

        :param string session_id: The session ID of the EV desired.
        :return: The requested EV, or if not found ``None``
        :rtype: EV or None
        '''
        requested_ev = None
        for ev in self.EVs:
            if ev.session_id == session_id:
                requested_ev = ev
        return requested_ev

    def get_all_EVSEs(self):
        return self.EVSEs

    def get_EVSE(self, station_id):
        return

    def get_occupancy(self, iteration):
        nbr_plugged_in_EVs = 0
        for ev in self.get_all_EVs():
            if ev.arrival <= iteration and iteration < ev.departure:
                nbr_plugged_in_EVs = nbr_plugged_in_EVs + 1
        return nbr_plugged_in_EVs

    def get_active_EVs(self, iteration):
        active_EVs = []
        for ev in self.get_all_EVs():
            if ev.arrival <= iteration and iteration < ev.departure and iteration <= ev.finishing_time:
                active_EVs.append(ev)
        return active_EVs


class Event:
    '''
    :ivar string type: The event type.
    :ivar int iteration: The time (iteration) this event was created.
    :ivar string description: The event description.
    :ivar string session: Which session that spawned this event.
    '''

    def __init__(self, type, iteration, description, session='GLOBAL'):
        self.type = type
        self.iteration = iteration
        self.description = description
        self.session = session

