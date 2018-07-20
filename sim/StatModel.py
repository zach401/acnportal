import pickle
from datetime import datetime, timedelta
import numpy as np
import random

class StatModel:

    def __init__(self):
        self.sessions = pickle.load(open('April_2018_Sessions.pkl', 'rb'))
        self.arrival_rates_week, self.arrival_rates_weekend = self.determine_arrival_rates()
        self.stay_density_arrays, self.stay_density_edges = self.determine_stay_density_arrays()
        self.energy_demand_density, self.energy_demand_density_edges = self.determine_energy_demand_array()

        pass

    def determine_arrival_rates(self):
        '''
        Determines the rates of the poisson process that models the arrival rates
        of the EVs to the parking garage. As the arrival rates are very different during the
        week and the weekend, two different arrival rates arrays are calculated.

        :return: (list, list) Returns two lists containing the hourly rates in weeks and the weekends.
        '''
        days_weekend = set()
        days_week = set()
        arrival_rates_weekend = [0] * 24
        arrival_rates_week = [0] * 24
        for s in self.sessions:
            arrival = s[0] - timedelta(hours=7)
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

    def determine_stay_density_arrays(self):
        '''
        Calculates the cumulative distribution functions (CDF) modeling the stay durations of the EVs per hour.
        Along with the cumulative distribution functions the edges of the distribution are calculated as well.

        :return: (list, list) The first list contains 24 arrays containing the CDFs for the stay durations for every hour
                The second list also contains 24 arrays but these describes the values each step in the CDF corresponds to.
        '''
        stay_duration_hours = {}
        for i in range(24):
            stay_duration_hours[i] = []
        # stay duration
        for s in self.sessions:
            arrival = s[0] - timedelta(hours=7)
            departure = s[1] - timedelta(hours=7)
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

    def determine_energy_demand_array(self):
        '''
        Calculates the Cumulative Distribution function (CDF) modeling how much energy each EV needs.
        Also calculates the edge value each value in the CDF corresponds to.

        :return: (list, list) The first list contains the CDFs for the stay durations for every hour.
                The second list also contains the edge values each step in the CDF corresponds to.
        '''
        energy_demands = []
        energy_demand_density = []
        for s in self.sessions:
            energy_demands.append(s[2])
        hist, density_edges = np.histogram(energy_demands, bins=20, density=True)
        i = 0
        for h in np.nditer(hist):
            new_value = h * (density_edges[i + 1] - density_edges[i])
            if i == 0:
                energy_demand_density.append(new_value)
            else:
                energy_demand_density.append(energy_demand_density[i - 1] + new_value)
            i = i + 1
        return energy_demand_density, density_edges


    def get_arrival_rate(self, weekday, hour):
        '''
        Returns the poisson process rate describing the EV arrival rate.
        :param weekday: (int) The weekday. Monday=0, Sunday=6
        :param hour: (int) The hour of the day
        :return: (float) The arrival rate [arrivals/hour]
        '''
        if weekday < 5:
            return self.arrival_rates_week[hour] / 3600
        else:
            return self.arrival_rates_weekend[hour] / 3600

    def get_stay_duration(self, hour):
        '''
        Returns the stay duration according to the stay duration distributions
        extracted from the real ACN data.

        :param hour: (int) The hour of the day
        :return: (float) The stay duration [hours]
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

    def get_energy_demand(self):
        '''
        Returns the energy demand for an EV according to the energy demand distributions
        extracted from the real ACN data.

        :return: (float) The energy demand for an EV [kWh]
        '''
        density_array = self.energy_demand_density
        edges = self.energy_demand_density_edges
        rand = random.uniform(0, 1)
        i = 0
        while density_array[i] < rand:
            i = i + 1
        energy_demand = random.uniform(edges[i], edges[i + 1])
        if energy_demand > 30:
            a = 0
        return energy_demand