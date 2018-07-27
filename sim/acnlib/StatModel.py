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
        self.stay_density_arrays, self.stay_density_edges = self.__determine_stay_density_arrays()
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
        for i in range(24):
            stay_duration_hours[i] = []
        # stay duration
        for s in self.sessions:
            arrival = s[0] - timedelta(hours=config.time_zone_diff_hour)
            departure = s[1] - timedelta(hours=config.time_zone_diff_hour)
            stay_duration = (departure - arrival).total_seconds() / 3600
            # if stay_duration >= 0 and stay_duration <= 1000:
            stay_duration_hours[arrival.hour].append(stay_duration)
        stay_density_arrays = []
        stay_density_edges = []
        for key, data in stay_duration_hours.items():
            hist, edges = np.histogram(data, bins=20, density=True)
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
        return stay_density_arrays, stay_density_edges

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
        hist_week, density_edges_week = np.histogram(energy_demands, bins=20, density=True)
        hist_weekend, density_edges_weekend = np.histogram(energy_demands_weekend, bins=20, density=True)
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

    def get_stay_duration(self, hour):
        '''
        Returns the stay duration according to the stay duration distributions
        extracted from the real ACN data.

        :param int hour: The hour of the day
        :return: The stay duration [hours]
        :rtype: float
        '''
        density_array = self.stay_density_arrays[hour]
        edges = self.stay_density_edges[hour]
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