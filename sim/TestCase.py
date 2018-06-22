import pickle
from datetime import datetime, timedelta
import math
from EV import EV


class TestCase:
    def __init__(self, EVs, voltage=220, max_rate=32, period=1):
        self.VOLTAGE = voltage
        self.DEFAULT_MAX_RATE = max_rate
        self.period = period
        self.EVs = EVs

    def step(self):
        pass




def generate_test_case_local(file_name, start, end, voltage=220, max_rate=32, period=1, max_duration=720):
    sessions = pickle.load(open(file_name, 'rb'))
    EVs = []
    uid = 0
    min_arrival = None
    for s in sessions:
        if start <= s[0]-timedelta(hours=7) and s[0]-timedelta(hours=7) <= end and s[2] >= 0.5:
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
    return TestCase(EVs, voltage, max_rate, period)