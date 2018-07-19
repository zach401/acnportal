from EVSE import EVSE
from EV import EV
from TestCase import TestCase
from StatModel import StatModel
import math
import pickle
from datetime import datetime, timedelta
import random
from scipy.stats import norm
import numpy as np

class Garage:

    def __init__(self):
        self.EVSEs = []
        self.test_case = None
        self.stat_model = StatModel()

        self.define_garage()
        pass

    def define_garage(self):
        '''
        Creates the EVSEs of the garage.
        :return: None
        '''
        self.EVSEs.append(EVSE('CA-148', 'AeroVironment'))
        self.EVSEs.append(EVSE('CA-149', 'AeroVironment'))
        self.EVSEs.append(EVSE('CA-212', 'AeroVironment'))
        self.EVSEs.append(EVSE('CA-213', 'AeroVironment'))
        for i in range(303, 328):
            # CA-303 to CA-327
            if i >= 320 and i <= 323:
                self.EVSEs.append(EVSE('CA-' + str(i), 'ClipperCreek'))
            else:
                self.EVSEs.append(EVSE('CA-' + str(i), 'AeroVironment'))
        for i in range(489, 513):
            # CA-489 to CA-513
            if i >= 493 and i <= 496:
                self.EVSEs.append(EVSE('CA-' + str(i), 'ClipperCreek'))
            else:
                self.EVSEs.append(EVSE('CA-' + str(i), 'AeroVironment'))

    def set_test_case(self, test_case):
        self.test_case = test_case

    def generate_test_case(self, start_dt, end_dt, period=1):
        # define start and end time
        start = (start_dt + timedelta(hours=7)).timestamp()
        end = (end_dt + timedelta(hours=7)).timestamp()
        # get the arrival rates
        last_arrival = start
        # specifications
        max_rate = 32
        voltage = 220
        EVs = []
        uid = 0
        min_arrival = None
        # loop until end time
        while last_arrival < end:
            weekday = datetime.fromtimestamp(last_arrival).weekday()
            hour = datetime.fromtimestamp(last_arrival).hour
            # get the rate used in the poisson process
            rate = self.stat_model.get_arrival_rate(weekday, hour)
            rand = 1 - random.random() # a number in range (0, 1]
            next_full_hour = (datetime.fromtimestamp(last_arrival).replace(microsecond=0,second=0,minute=0) + timedelta(hours=1)).timestamp()
            next_arrival = last_arrival + 3601
            if rate != 0:
                next_arrival = last_arrival + (-math.log(rand)/rate)
            if next_arrival > next_full_hour:
                # no EV arrived by time within this hour
                last_arrival = next_full_hour
            else:
                # an EV arrived
                new_hour = datetime.fromtimestamp(next_arrival).hour
                stay_duration = self.stat_model.get_stay_duration(new_hour)
                #while stay_duration <= 0:
                #    stay_duration = np.abs(np.random.normal(stay_hourly_mean[hour], math.sqrt(stay_hourly_var[hour])))
                energy = 20
                free_charging_station_id = self.find_free_EVSE(EVs, next_arrival // 60 // period)
                if free_charging_station_id != None:
                    ev = EV(next_arrival // 60 // period,
                            math.ceil((next_arrival + stay_duration * 3600) / 60 / period),
                            ((energy * (60 / period) * 1e3) / voltage),
                            max_rate,
                            free_charging_station_id,
                            uid)
                    #if ev.departure - ev.arrival < ev.requested_energy / ev.max_rate:
                    #    ev.departure = math.ceil(ev.requested_energy / ev.max_rate) + ev.arrival
                    uid += 1
                    if not min_arrival:
                        min_arrival = ev.arrival
                    elif min_arrival > ev.arrival:
                        min_arrival = ev.arrival
                    EVs.append(ev)
                    last_arrival = next_arrival
        for ev in EVs:
            ev.arrival -= min_arrival
            ev.departure -= min_arrival
        self.test_case = TestCase(EVs, (min_arrival*60*period), voltage, max_rate, period)


    def find_free_EVSE(self, EVs, current_time):
        evse_collection = []
        for evse in self.EVSEs:
            evse_collection.append(evse.station_id)

        for ev in EVs:
            if ev.arrival <= current_time and current_time <= ev.departure:
                evse_collection.remove(ev.station_id)

        coll_length = len(evse_collection)
        if coll_length > 0:
            return evse_collection[random.randint(0, coll_length - 1)]
        else:
            return None

    def update_state(self, pilot_signals, iteration):
        self.test_case.step(pilot_signals, iteration)

    def event_occured(self, iteration):
        return self.test_case.event_occured(iteration)

    def get_charging_data(self):
        return self.test_case.get_charging_data()

    def get_active_EVs(self, iteration):
        return self.test_case.get_active_EVs(iteration)

    def get_allowable_rates(self, station_id):
        EVSE = next((x for x in self.EVSEs if x.station_id == station_id), None)
        if EVSE == None:
            return [0]
        return EVSE.allowable_pilot_signals

    @property
    def last_departure(self):
        return self.test_case.last_departure

    @property
    def max_rate(self):
        return self.test_case.DEFAULT_MAX_RATE

    @property
    def allowable_rates(self):
        return self.test_case.ALLOWABLE_RATES