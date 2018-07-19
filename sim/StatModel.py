import pickle
from datetime import datetime, timedelta
import numpy as np
import random

class StatModel:

    def __init__(self):
        self.sessions = pickle.load(open('April_2018_Sessions.pkl', 'rb'))
        self.arrival_rates_week, self.arrival_rates_weekend = self.determine_arrival_rates()
        self.hourly_mean, self.hourly_var = self.determine_stay_duration_distribution_parametres()
        self.stay_density_arrays, self.stay_density_edges = self.determine_stay_density_arrays()

        pass

    def determine_arrival_rates(self):
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

    def determine_stay_duration_distribution_parametres(self):
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
        hourly_mean = []
        hourly_var = []
        for key, data in stay_duration_hours.items():
            mu = np.mean(data)
            sigma = np.var(data)
            if sigma == 0:
                sigma = 1
            if sigma > 50:
                sigma = 50
            hourly_mean.append(mu)
            hourly_var.append(sigma)
        return hourly_mean, hourly_var

    def determine_stay_density_arrays(self):
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


    def get_arrival_rate(self, weekday, hour):
        if weekday < 5:
            return self.arrival_rates_week[hour] / 3600
        else:
            return self.arrival_rates_weekend[hour] / 3600

    def get_stay_duration(self, hour):
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
