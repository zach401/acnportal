import pickle
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import numpy as np

sessions = pickle.load(open('April_2018_Sessions.pkl', 'rb'))


stay_duration_hours = {}
for i in range(24):
    stay_duration_hours[i] = []
stay_duration_list = []
energy_demand = []

days_weekend = set()
days_week = set()
arrival_rates_weekend = [0]*24
arrival_rates_week = [0]*24
for s in sessions:
    arrival = s[0]-timedelta(hours=7)
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

# stay duration
for s in sessions:
    arrival = s[0]-timedelta(hours=7)
    departure = s[1]-timedelta(hours=7)
    stay_duration = (departure - arrival).total_seconds() / 3600
    #if stay_duration >= 0 and stay_duration <= 1000:
    stay_duration_hours[arrival.hour].append(stay_duration)
    stay_duration_list.append(stay_duration)
    energy_demand.append(s[2])

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


def normal_curve(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


plt.figure(1)
hourly_mean = []
hourly_var = []
for key, data in stay_duration_hours.items():
    mu = np.mean(data)
    sigma = np.var(data)
    hourly_mean.append(mu)
    hourly_var.append(sigma)
    if sigma == 0:
        sigma = 1
    x = key//6
    y = key%6
    plt.subplot(4, 6, key + 1)
    plt.hist(data, bins=20, range=(0, 40), normed=True)
    xx = np.linspace(0,40)
    yy = normal_curve(xx, mu, np.sqrt(sigma))
    plt.plot(xx, yy)

plt.figure(2)
for i in range(24):
    plt.subplot(4, 6, i + 1)
    density_array = stay_density_arrays[i]
    density_edges = stay_density_edges[i].tolist()
    density_array.insert(0,0)
    density_edges.insert(0,0)
    density_edges.pop()
    plt.step(density_edges, density_array)
    plt.xlim((0,40))

plt.figure(3)
plt.scatter(stay_duration_list, energy_demand)
