import pickle
from datetime import datetime, timedelta
import math
from EV import EV


class TestCase:
    '''
    TestCase represents a garage of charging stations over a certain simulation time.
    Stores the data of the test case when the simulation is run.
    '''
    def __init__(self, EVs, start_timestamp,voltage=220, max_rate=32, period=1, allowable_rates=[0, 8, 16, 24, 32]):
        self.VOLTAGE = voltage
        self.DEFAULT_MAX_RATE = max_rate
        self.ALLOWABLE_RATES = allowable_rates
        self.period = period
        self.start_timestamp = start_timestamp
        self.EVs = EVs

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
            charge_rate = ev.charge(pilot_signals[ev.session_id], tail=True)
            self.charging_data[ev.session_id].append({'time': iteration,
                                                      'charge_rate': charge_rate,
                                                      'pilot_signal': pilot_signals[ev.session_id],
                                                      'remaining_demand': ev.remaining_demand})
            ev.finishing_time = iteration


    def get_active_EVs(self, iteration):
        '''
        Returns the EVs that is currently attached to the charging stations and
        has not had their energy demand met.

        :param iteration: (int) The current time stamp of the simulation
        :return: (list) List of EVs currently plugged in and not finished charging
        '''
        active_EVs = []
        for ev in self.EVs:
            if ev.remaining_demand > 0 and ev.arrival <= iteration and ev.departure > iteration:
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
            ev = EV(s[0].timestamp() // 60 // period,
                    (math.ceil(s[1].timestamp() / 60 / period)),
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
    return TestCase(EVs, start.timestamp(),voltage, max_rate, period)