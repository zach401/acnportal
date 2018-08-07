import pickle
from datetime import datetime, timedelta
import numpy as np
import random
import config

class StatModel:
    '''
    This class describes the statistical model of EVs **arriving times**, **stay durations** and **energy demand**.

    - The arriving times of the EVs are described with a poisson process with variable rate depending on the hour of
        the day and if it is a weekday or a weekend.
    - The stay durations are modeled by analyzing the hourly distributions of the real charging sessions and creating
        empirical density functions. After that a CDF is calculated for every hour of the day from which the stay duration
        can be caluclated when using random variables.
    - The energy demand for the EVs are calculated by analyzing the real charging sessions and the distribution of how much
        energy the EVs need. After that a CDF is calculated from which the energy demand can be calculated using random variables.

    Upon creation, this class reads the file with the real charging sessions with the file name defined in ``sim/config.py``.
    '''

    def __init__(self):
        self.sessions = pickle.load(open(config.stat_model_data_source, 'rb'))
        self.arrival_rates_week, self.arrival_rates_weekend = self.__determine_arrival_rates()
        self.stay_density_arrays, self.stay_density_edges, \
            self.stay_density_arrays_weekend, self.stay_density_edges_weekend = self.__determine_stay_density_arrays()
        self.energy_demand_density_arrays, self.energy_demand_density_edges_arrays = self.__determine_energy_demand_array()

        pass

    def __determine_arrival_rates(self):
        '''
        Determines the rates of the poisson process that models the arrival rates
        of the EVs to the parking garage. As the arrival rates are very different during the
        week and the weekend, two different arrival rates arrays are calculated.

        :return: Returns a tuple of two lists containing the hourly rates in weeks and the weekends.
        :rtype: tuple(list(float), list(float))
        '''
        days_weekend = set()
        days_week = set()
        arrival_rates_weekend = [0] * 24
        arrival_rates_week = [0] * 24
        for s in self.sessions:
            arrival = s[0] + timedelta(hours=config.time_zone_diff_hour)
            hour_of_day = arrival.hour
            if arrival.weekday() < 5:
                days_week.add(arrival.strftime('%y-%m-%d'))
                arrival_rates_week[hour_of_day] = arrival_rates_week[hour_of_day] + 1
            else:
                days_weekend.add(arrival.strftime('%y-%m-%d'))
                arrival_rates_weekend[hour_of_day] = arrival_rates_weekend[hour_of_day] + 1
        nbr_weekdays = len(days_week)
        nbr_weekenddays = len(days_weekend)
        arrival_rates_week[:] = [x / nbr_weekdays for x in arrival_rates_week]
        arrival_rates_weekend[:] = [x / nbr_weekenddays for x in arrival_rates_weekend]
        return arrival_rates_week, arrival_rates_weekend

    def __determine_stay_density_arrays(self):
        '''
        Calculates the cumulative distribution functions (CDF) modeling the stay durations of the EVs per hour.
        Along with the cumulative distribution functions the edges of the distribution are calculated as well.

        :return: The first list contains 24 arrays containing the CDFs for the stay durations for every hour
                The second list also contains 24 arrays but these describes the values each step in the CDF corresponds to.
        :rtype: tuple(list(float), list(float))
        '''
        stay_duration_hours = {}
        stay_duration_hours_weekend = {}
        for i in range(24):
            stay_duration_hours[i] = []
            stay_duration_hours_weekend[i] = []
        # stay duration
        for s in self.sessions:
            arrival = s[0] - timedelta(hours=config.time_zone_diff_hour)
            departure = s[1] - timedelta(hours=config.time_zone_diff_hour)
            stay_duration = (departure - arrival).total_seconds() / 3600
            # if stay_duration >= 0 and stay_duration <= 1000:
            if arrival.weekday() < 5:
                stay_duration_hours[arrival.hour].append(stay_duration)
            else:
                stay_duration_hours_weekend[arrival.hour].append(stay_duration)
        stay_density_arrays = []
        stay_density_edges = []
        for key, data in stay_duration_hours.items():
            hist, edges = np.histogram(data, bins=60, density=True)
            density_array = []
            i = 0
            for h in np.nditer(hist):
                new_value = h * (edges[i+1]-edges[i])
                if i == 0:
                    density_array.append(new_value)
                else:
                    density_array.append(density_array[i - 1] + new_value)
                i = i + 1
            stay_density_arrays.append(density_array)
            stay_density_edges.append(edges)
        stay_density_arrays_weekend = []
        stay_density_edges_weekend = []
        for key, data in stay_duration_hours_weekend.items():
            hist, edges = np.histogram(data, bins=60, density=True)
            density_array = []
            i = 0
            for h in np.nditer(hist):
                new_value = h * (edges[i+1]-edges[i])
                if i == 0:
                    density_array.append(new_value)
                else:
                    density_array.append(density_array[i - 1] + new_value)
                i = i + 1
            stay_density_arrays_weekend.append(density_array)
            stay_density_edges_weekend.append(edges)
        return stay_density_arrays, stay_density_edges, stay_density_arrays_weekend, stay_density_edges_weekend

    def __determine_energy_demand_array(self):
        '''
        Calculates the Cumulative Distribution function (CDF) modeling how much energy each EV needs.
        Also calculates the edge value each value in the CDF corresponds to.

        :return: The first list contains the CDFs for the stay durations for every hour.
                The second list also contains the edge values each step in the CDF corresponds to.
        :rtype: tuple(tuple(list(float), list(float)), tuple(list(float), list(float)))
        '''
        energy_demands = []
        energy_demands_weekend = []
        energy_demand_density = []
        energy_demand_density_weekend =[]
        for s in self.sessions:
            arrival = s[0] + timedelta(hours=config.time_zone_diff_hour)
            if arrival.weekday() < 5:
                energy_demands.append(s[2])
            else:
                energy_demands_weekend.append(s[2])
        hist_week, density_edges_week = np.histogram(energy_demands, bins=60, density=True)
        hist_weekend, density_edges_weekend = np.histogram(energy_demands_weekend, bins=60, density=True)
        i = 0
        # Build the CDFs
        for h in np.nditer(hist_week):
            new_value = h * (density_edges_week[i + 1] - density_edges_week[i])
            if i == 0:
                energy_demand_density.append(new_value)
            else:
                energy_demand_density.append(energy_demand_density[i - 1] + new_value)
            i = i + 1
        i=0
        for h in np.nditer(hist_weekend):
            new_value = h * (density_edges_weekend[i + 1] - density_edges_weekend[i])
            if i == 0:
                energy_demand_density_weekend.append(new_value)
            else:
                energy_demand_density_weekend.append(energy_demand_density_weekend[i - 1] + new_value)
            i = i + 1
        return (energy_demand_density, energy_demand_density_weekend), (density_edges_week, density_edges_weekend)


    def get_arrival_rate(self, weekday, hour):
        '''
        Returns the poisson process rate describing the EV arrival rate.
        :param int weekday: The weekday. Monday=0, Sunday=6
        :param int hour: The hour of the day
        :return: The arrival rate [arrivals/hour]
        :rtype: float
        '''
        if weekday < 5:
            return self.arrival_rates_week[hour] / 3600
        else:
            return self.arrival_rates_weekend[hour] / 3600

    def get_stay_duration(self, weekday, hour):
        '''
        Returns the stay duration according to the stay duration distributions
        extracted from the real ACN data.

        :param int hour: The hour of the day
        :param int weekday: The weekday
        :return: The stay duration [hours]
        :rtype: float
        '''
        density_array = []
        edges = []
        if weekday < 5:
            density_array = self.stay_density_arrays[hour]
            edges = self.stay_density_edges[hour]
        else:
            density_array = self.stay_density_arrays_weekend[hour]
            edges = self.stay_density_edges_weekend[hour]
        rand = random.uniform(0, 1)
        i = 0
        while density_array[i] < rand:
            i = i + 1
        stay_duration = random.uniform(edges[i], edges[i+1])
        if stay_duration > 30:
            a = 0
        return stay_duration

    def get_energy_demand(self, weekday=0):
        '''
        Returns the energy demand for an EV according to the energy demand distributions
        extracted from the real ACN data.

        :param int weekday: Weekday of arriving EV
        :return: The energy demand for an EV [kWh]
        :rtype: float
        '''
        density_array = []
        edges = []
        if weekday < 5:
            density_array = self.energy_demand_density_arrays[0]
            edges = self.energy_demand_density_edges_arrays[0]
        else:
            density_array = self.energy_demand_density_arrays[1]
            edges = self.energy_demand_density_edges_arrays[1]
        rand = random.uniform(0, 1)
        i = 0
        while density_array[i] < rand:
            i = i + 1
        energy_demand = random.uniform(edges[i], edges[i + 1])
        if energy_demand > 30:
            a = 0
        return energy_demand

def __get_beahavioral_stats(test_case):
    arrival_hours = []
    departure_hours = []
    requested_energy = []
    stay_durations = []
    number_of_EVs = len(test_case.EVs)
    for ev in test_case.EVs:
        # - Gather data for arrivals and departures
        arrival_time = datetime.fromtimestamp(ev.arrival * 60 * test_case.period +
                                              test_case.start_timestamp)
        departure_time = datetime.fromtimestamp(ev.departure * 60 * test_case.period +
                                                test_case.start_timestamp)
        arrival_time = arrival_time - timedelta(hours=config.time_zone_diff_hour)
        departure_time = departure_time - timedelta(hours=config.time_zone_diff_hour)
        arrival_hours.append(arrival_time.hour)
        departure_hours.append(departure_time.hour)
        # - Gather data for requested energy
        requested_energy.append(
            (ev.requested_energy / (60 / test_case.period)) * test_case.voltage / 1000)
        # - Gather data for stay times
        stay_durations.append(((ev.departure - ev.arrival) * test_case.period) / 60)
    return (arrival_hours, departure_hours, requested_energy, stay_durations)

def __get_distribution_probabilities(data, percentage=True, bins=20, range=None, align='center'):
    hist, bins = np.histogram(data, bins=bins, range=range)
    distribution = hist.astype(np.float32) / ((hist.sum()) if percentage else 1)
    return distribution.tolist()

def __calc_statistical_distance(P, Q):
    '''
    Impelents the Hellinger distance. Returns the upper bound.

    :param list(float) P: real distribution
    :param list(float) Q: model distribution
    :return: The Hellinger statistical distance
    :rtype: float
    '''
    pow_sqrt_sum = 0
    lenght = len(P)
    for i in range(0,lenght):
        pow_sqrt_sum = pow_sqrt_sum + np.power((np.sqrt(P[i]) - np.sqrt(Q[i])), 2)
    hellinger_dist = np.sqrt(pow_sqrt_sum) / np.sqrt(2)
    return hellinger_dist * np.sqrt(2)

def compare_model_to_real(real_test_case, model_test_case):
    '''
    Compares a real test case to a test case generated by a model. Calculates the
    statistical distance (Hellinger distance) of the arrival, stay duration and energy demand distributions.

    Prints the results

    :param real_test_case: A test case generated from real session data
    :param model_test_case: A test case generated from a statistical model
    :return: None
    '''
    arrival_hours_real, departure_hours_real, requested_energy_real, stay_durations_real = __get_beahavioral_stats(
        real_test_case)
    arrival_hours_model, departure_hours_model, requested_energy_model, stay_durations_model = __get_beahavioral_stats(
        model_test_case)

    arrival_dist_real, arrival_dist_model = __get_distribution_probabilities(arrival_hours_real,
                                                                             range=(0,24),
                                                                             bins=24),\
                                            __get_distribution_probabilities(arrival_hours_model,
                                                                             range=(0,24),
                                                                             bins=24)
    departure_dist_real, departure_dist_model = __get_distribution_probabilities(departure_hours_real,
                                                                                 range=(0,24),
                                                                                 bins=24), \
                                                __get_distribution_probabilities(departure_hours_model,
                                                                                 range=(0,24),
                                                                                 bins=24)
    requested_energy_dist_real, requested_energy_dist_model = __get_distribution_probabilities(requested_energy_real,
                                                                                               range=(0,50),
                                                                                               bins=15), \
                                                              __get_distribution_probabilities(requested_energy_model,
                                                                                               range=(0,50),
                                                                                               bins=15)
    stay_duration_dist_real, stay_duration_dist_model = __get_distribution_probabilities(stay_durations_real,
                                                                                         range=(0,50),
                                                                                         bins=15), \
                                                        __get_distribution_probabilities(stay_durations_model,
                                                                                         range=(0,50),
                                                                                         bins=15)

    arrival_distance = __calc_statistical_distance(arrival_dist_real, arrival_dist_model)
    departure_distance = __calc_statistical_distance(departure_dist_real, departure_dist_model)
    energy_distance = __calc_statistical_distance(requested_energy_dist_real, requested_energy_dist_model)
    stay_distance = __calc_statistical_distance(stay_duration_dist_real, stay_duration_dist_model)

    print('Arrival stat dist: {}'.format(arrival_distance))
    print('Departure stat dist: {}'.format(departure_distance))
    print('Energy stat dist: {}'.format(energy_distance))
    print('Stay stat dist: {}'.format(stay_distance))


