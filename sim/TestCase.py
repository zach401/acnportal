import pickle
from datetime import datetime, timedelta
import math
from EV import EV
from SimulationOutput import SimulationOutput, Event


class TestCase:
    '''
    TestCase represents a garage of charging stations over a certain simulation time.
    Stores the data of the test case when the simulation is run.
    '''
    def __init__(self, EVs, start_timestamp,voltage=220, max_rate=32, period=1):
        self.VOLTAGE = voltage
        self.DEFAULT_MAX_RATE = max_rate
        self.period = period
        self.start_timestamp = start_timestamp
        self.EVs = EVs

        self.simulation_output = SimulationOutput(start_timestamp, period, max_rate, voltage)

        self.charging_data = {}
        self.clear_data()

    def step(self, pilot_signals, iteration):
        '''
        Updates the states of the EVs connected the system and stores the relevant data.

        :param pilot_signals: (dict) A dictionary where key is the EV id and the value is a number with the charging rate
        :param iteration: (int) The current time stamp of the simulation
        :return: None
        '''
        active_EVs = self.get_active_EVs(iteration)
        for ev in active_EVs:
            if ev.arrival == iteration:
                self.simulation_output.submit_event(Event('ERROR',
                                                          iteration,
                                                          'EV arrived at station {}'.format(ev.station_id),
                                                          ev.session_id))
            charge_rate = ev.charge(pilot_signals[ev.session_id], tail=True)
            if charge_rate > self.DEFAULT_MAX_RATE:
                # If charging rate was exceeded output an error
                self.simulation_output.submit_event(Event('ERROR',
                                                          iteration,
                                                          'Max charging rate exceeded: {}A'.format(charge_rate),
                                                          ev.session_id))
            if charge_rate == 0:
                # If charging rate is set to 0A before EV finished charging output a warning
                self.simulation_output.submit_event(Event('WARNING',
                                                          iteration,
                                                          'Charging rate is set to 0A before EV finished charging',
                                                          ev.session_id))
            sample = {'time': iteration,
                      'charge_rate': charge_rate,
                      'pilot_signal': pilot_signals[ev.session_id],
                      'remaining_demand': ev.remaining_demand}
            self.simulation_output.submit_charging_data(ev.session_id, sample)
            self.charging_data[ev.session_id].append(sample)
            if ev.fully_charged:
                ev.finishing_time = iteration
                self.simulation_output.submit_event(Event('INFO',
                                                          iteration,
                                                          'EV finished charging at station {}'.format(ev.station_id),
                                                          ev.session_id))


    def get_active_EVs(self, iteration):
        '''
        Returns the EVs that is currently attached to the charging stations and
        has not had their energy demand met.

        :param iteration: (int) The current time stamp of the simulation
        :return: (list) List of EVs currently plugged in and not finished charging
        '''
        active_EVs = []
        for ev in self.EVs:
            if not ev.fully_charged and ev.arrival <= iteration and ev.departure > iteration:
                active_EVs.append(ev)
        return active_EVs


    def get_charging_data(self):
        return self.charging_data


    def event_occured(self, iteration):
        for ev in self.EVs:
            if ev.arrival == iteration or ev.departure == iteration:
                return True
        return False

    def clear_data(self):
        for ev in self.EVs:
            self.charging_data[ev.session_id] = []
            ev.reset()

    @property
    def last_departure(self):
        last_departure = 0
        for ev in self.EVs:
            if ev.departure > last_departure:
                last_departure = ev.departure
        return last_departure


def generate_test_case_local(file_name, start, end, voltage=220, max_rate=32, period=1, max_duration=3600):
    sessions = pickle.load(open(file_name, 'rb'))
    EVs = []
    uid = 0
    min_arrival = None
    for s in sessions:
        if start <= s[0]-timedelta(hours=7) and s[1]-timedelta(hours=7) <= end and s[2] >= 0.5:
            arrival = s[0]-timedelta(hours=7)
            departure = s[1]-timedelta(hours=7)
            ev = EV(arrival.timestamp() // 60 // period,
                    (math.ceil(departure.timestamp() / 60 / period)),
                    ((s[2] * (60/period) * 1e3) / voltage),
                    max_rate,
                    s[3],
                    uid)
            if ev.departure - ev.arrival < ev.requested_energy / ev.max_rate:
                ev.departure = math.ceil(ev.requested_energy / ev.max_rate) + ev.arrival
            uid += 1
            if not min_arrival:
                min_arrival = ev.arrival
            elif min_arrival > ev.arrival:
                min_arrival = ev.arrival
            EVs.append(ev)

    for ev in EVs:
        ev.arrival -= min_arrival
        ev.departure -= min_arrival
        if ev.departure - ev.arrival > max_duration:
            ev.departure = ev.arrival + max_duration
    EVs.sort(key=lambda x: x.station_id)
    return TestCase(EVs, (min_arrival*60*period),voltage, max_rate, period)