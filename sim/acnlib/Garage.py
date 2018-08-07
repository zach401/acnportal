from acnlib.EVSE import EVSE
from acnlib.EV import EV
from acnlib.TestCase import TestCase
from acnlib.StatModel import StatModel
from acnlib.SimulationOutput import Event
import math
from datetime import datetime, timedelta
import random
import config


class Garage:

    def __init__(self):
        self.EVSEs = []
        self.test_case = None
        self.stat_model = StatModel()
        self.active_EVs = []

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
        '''
        Manually set test case for the simulation.
        This function is used if the test case is generated from real data.

        :param test_case: (TestCase) The manually generated test case
        :return: None
        '''
        self.test_case = test_case

    def generate_test_case(self, start_dt, end_dt, model='empirical', period=1, voltage = 220, max_rate = 32):
        '''
        Function for auto-generating a test case. The test case is generated from a statistical model based
        on real data from Caltech ACN.

        :param datetime start_dt: When the simulation should start.
        :param datetime end_dt: When the simulation should end.
        :param int period: How many minutes one period should be.
        :param float voltage: Grid voltage level.
        :param float max_rate: Maximum charging rate.
        :param string model: Defining which model will be used to generate the test case. Available: {'empirical'}
        :return: The test case generated. It is stored in the garage but is also returned if it should be analyzed
        :rtype: TestCase
        '''

        # NOTE:
        # To add another model to generate a test case, create a new function and
        # add it to the if-else statements below.

        if model == 'empirical':
            return self.emprical_model(start_dt,end_dt,period,voltage,max_rate)
        else:
            return None



    def emprical_model(self, start_dt, end_dt, period, voltage, max_rate):
        '''
        This model is based on the real data and the arrivals, stay durations and energy demand
        are modelled as:

        - A Poisson process with variable rates depending of the time of day
        is used to model the arrivals of the EVs to the garage.
        - To model the stay durations, cumulative distribution functions are empirically derived from
        the distributions for every hour of the day.
        - To model the energy demand, cumulative distribution functions are also derived from the real data

        :param datetime start_dt: When the simulation should start.
        :param datetime end_dt: When the simulation should end.
        :param int period: How many minutes one period should be.
        :param float voltage: Grid voltage level.
        :param float max_rate: Maximum charging rate.
        :return: The test case generated. It is stored in the garage but is also returned if it should be analyzed
        :rtype: TestCase
        '''
        # define start and end time
        start = (start_dt).timestamp()
        end = (end_dt + timedelta(hours=config.time_zone_diff_hour)).timestamp()
        # get the arrival rates
        last_arrival = start
        # specifications
        EVs = []
        uid = 0
        min_arrival = None
        # loop until end time. This loop is the base of the Poisson process.
        while last_arrival < end:
            weekday = datetime.fromtimestamp(last_arrival).weekday()
            hour = datetime.fromtimestamp(last_arrival).hour
            # get the rate used in the poisson process
            rate = self.stat_model.get_arrival_rate(weekday, hour)
            rand = 1 - random.random()  # a number in range (0, 1]
            next_full_hour = (
                        datetime.fromtimestamp(last_arrival).replace(microsecond=0, second=0, minute=0) + timedelta(
                    hours=1)).timestamp()

            next_arrival = last_arrival + 3601
            if rate != 0:
                next_arrival = last_arrival + (-math.log(rand) / rate)
            if next_arrival > next_full_hour:
                # no EV arrived by time within this hour
                last_arrival = next_full_hour
            else:
                # an EV arrived
                new_hour = datetime.fromtimestamp(next_arrival).hour
                new_weekday = datetime.fromtimestamp(next_arrival).weekday()
                stay_duration = self.stat_model.get_stay_duration(new_weekday, new_hour)
                # while stay_duration <= 0:
                #    stay_duration = np.abs(np.random.normal(stay_hourly_mean[hour], math.sqrt(stay_hourly_var[hour])))
                energy = self.stat_model.get_energy_demand(new_weekday)
                free_charging_station_id = self.find_free_EVSE(EVs, next_arrival // 60 // period)
                arrival_dt = datetime.fromtimestamp(next_arrival)
                departure_dt = datetime.fromtimestamp(next_arrival + stay_duration * 3600)
                if free_charging_station_id != None \
                        and departure_dt < end_dt - timedelta(hours=config.time_zone_diff_hour):
                    ev = EV(next_arrival // 60 // period,
                            math.ceil((next_arrival + stay_duration * 3600) / 60 / period),
                            ((energy * (60 / period) * 1e3) / voltage),
                            max_rate,
                            free_charging_station_id,
                            uid)
                    if ev.departure - ev.arrival < ev.requested_energy / ev.max_rate:
                        new_departure = math.ceil(ev.requested_energy / ev.max_rate) + ev.arrival
                        if new_departure < math.ceil(end / 60 / period):
                            ev.departure = math.ceil(ev.requested_energy / ev.max_rate) + ev.arrival
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
        EVs.sort(key=lambda x: x.station_id)
        self.test_case = TestCase(EVs, (min_arrival * 60 * period), voltage, max_rate, period)
        return self.test_case

    def find_free_EVSE(self, EVs, current_time):
        '''
        Function to determine which charging stations are empty.
        Used in the function self.generate_test_case.

        :param EVs: (list) List of the current EVs that has been added to the simulation
        :param current_time: (int) The current time. Represented by the period number.
        :return: (string) Station ID for an empty randomly selected EVSE
        '''
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
        evse_pilot_signals = {}
        for ev in self.active_EVs:
            new_pilot_signal = pilot_signals[ev.session_id]
            evse_pilot_signals[ev.station_id] = (ev.session_id ,new_pilot_signal)
        for evse in self.EVSEs:
            if evse.station_id in evse_pilot_signals:
                session_id = evse_pilot_signals[evse.station_id][0]
                pilot_signal = evse_pilot_signals[evse.station_id][1]
                change_ok = evse.change_pilot_signal(pilot_signal, session_id)
                if not change_ok and session_id == evse.last_session_id:
                    self.submit_event(Event('WARNING',
                                            iteration,
                                            'Wrong increase/decrease of pilot signal for station {}'.format(evse.station_id),
                                            session_id))
            else:
                # if no EV is using this station
                evse.last_applied_pilot_signal = 0

        self.test_case.step(pilot_signals, iteration)

    def submit_event(self, event):
        self.test_case.simulation_output.submit_event(event)

    def event_occurred(self, iteration):
        return self.test_case.event_occurred(iteration)

    def get_simulation_output(self):
        simulation_output = self.test_case.get_simulation_output()
        simulation_output.submit_all_EVSEs(self.EVSEs)
        return simulation_output

    def get_charging_data(self):
        return self.test_case.get_charging_data()

    def get_network_data(self):
        return self.test_case.get_network_data()

    def get_actual_charging_rates(self):
        return self.test_case.get_actual_charging_rates()

    def get_active_EVs(self, iteration):
        self.active_EVs = self.test_case.get_active_EVs(iteration)
        return self.active_EVs

    def get_allowable_rates(self, station_id):
        '''
        Returns the allowable pilot level signals for the selected EVSE.
        If no EVSE with the station_id presented is found, it will be created

        :param station_id: (string) The station ID for the EVSE
        :return: (list) List of allowable pilot signal levels for the EVSE
        '''
        evse = next((x for x in self.EVSEs if x.station_id == station_id), None)
        if evse == None:
            # If the EVSE was not found. Create it and add it to available stations
            # default manufacturer is AeroVironment.
            evse = EVSE(station_id, 'AeroVironment')
            self.EVSEs.append(evse)
        return evse.allowable_pilot_signals

    @property
    def last_departure(self):
        return self.test_case.last_departure

    @property
    def max_rate(self):
        return self.test_case.max_rate

    @property
    def allowable_rates(self):
        return self.test_case.ALLOWABLE_RATES

    def __get_intermediate_hours(self, current_time, next_time):
        hours_timestamps = []
        t = (datetime.fromtimestamp(current_time).replace(microsecond=0, second=0, minute=0) + timedelta(
                    hours=1)).timestamp()
        while t < next_time:
            hours_timestamps.append(t)
            t = (datetime.fromtimestamp(t).replace(microsecond=0, second=0, minute=0) + timedelta(
                    hours=1)).timestamp()
        return hours_timestamps


