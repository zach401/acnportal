from acnlib import TestCase
import pickle
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import numpy as np
from acnlib.StatModel import *

from acnlib.Garage import *
import config



sessions = pickle.load(open('July_25_Sessions.pkl', 'rb'))


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
    arrival = s[0]-timedelta(hours=config.time_zone_diff_hour)
    departure = s[1]-timedelta(hours=config.time_zone_diff_hour)
    stay_duration = (departure - arrival).total_seconds() / 3600
    #if stay_duration >= 0 and stay_duration <= 1000:
    stay_duration_hours[arrival.hour].append(stay_duration)
    stay_duration_list.append(stay_duration)
    energy_demand.append(s[2])

stay_density_arrays = []
stay_density_edges = []
for key, data in stay_duration_hours.items():
    hist, edges = np.histogram(data, bins=20, density=True,  range=(0, 40))
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


hourly_mean = []
hourly_var = []
fig1 = plt.figure(1)
plt.suptitle('Distribution of EVs stay durations every hour of the day')
ax = fig1.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
ax.set_xlabel('Stay durations [h]')
ax.set_ylabel('Probability denisty')
for key, data in stay_duration_hours.items():
    mu = np.mean(data)
    sigma = np.var(data)
    hourly_mean.append(mu)
    hourly_var.append(sigma)
    if sigma == 0:
        sigma = 1
    x = key//6
    y = key%6
    ax = fig1.add_subplot(4, 6, key + 1)
    ax.hist(data, bins=20, range=(0, 40), normed=True)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(28, ax.get_ylim()[1] - 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 'Hour {}'.format(key), bbox=props)
    xx = np.linspace(0,40)
    yy = normal_curve(xx, mu, np.sqrt(sigma))
    #plt.plot(xx, yy)

fig = plt.figure(2)
plt.suptitle('Cumulative distribution functions of EVs stay durations every hour of the day')
ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
ax.set_xlabel('Stay durations [h]')
ax.set_ylabel('Probability')
for i in range(24):
    ax = fig.add_subplot(4, 6, i + 1)
    density_array = stay_density_arrays[i]
    density_edges = stay_density_edges[i].tolist()
    density_array.insert(0,0)
    ax.step(density_edges, density_array)
    ax.set_xlim(-1,40)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(27, ax.get_ylim()[1] - 0.9 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 'Hour {}'.format(i), bbox=props)

# --------------------------------------------------
energy_demands = []
energy_demands_weekend = []
energy_demand_density = []
energy_demand_density_weekend =[]
for s in sessions:
    arrival = s[0] + timedelta(hours=config.time_zone_diff_hour)
    if arrival.weekday() < 5:
        energy_demands.append(s[2])
    else:
        energy_demands_weekend.append(s[2])
hist_week, density_edges_week = np.histogram(energy_demands, bins=20, density=True, range=(0,40))
hist_weekend, density_edges_weekend = np.histogram(energy_demands_weekend, bins=20, density=True, range=(0,40))
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

fig = plt.figure(3)
ax1 = fig.add_subplot(221)
ax1.hist(energy_demands, range=(0,40), bins=20, density=True)
ax1.set_ylabel('Probability density')
ax1.set_title('PDF of EV energy demand during weekdays.')

ax2 = fig.add_subplot(222)
ax2.hist(energy_demands_weekend, range=(0,40), bins=20, density=True)
ax2.set_title('PDF of EV energy demand during weekends.')

ax3 = fig.add_subplot(223)
energy_demand_density.insert(0,0)
ax3.step(density_edges_week, energy_demand_density)
ax3.set_xlim(-1,40)
ax3.set_xlabel('Energy demand [kWh]')
ax3.set_ylabel('Probability')
ax3.set_title('CDF of EV energy demand during weekdays.')

ax4 = fig.add_subplot(224)
energy_demand_density_weekend.insert(0,0)
ax4.step(density_edges_weekend, energy_demand_density_weekend)
ax4.set_xlim(-1,40)
ax4.set_xlabel('Energy demand [kWh]')
ax4.set_title('CDF of EV energy demand during weekends.')

stat_model = StatModel()

# ----------

fig = plt.figure(4)
ax1 = fig.add_subplot(121)
ax1.bar(range(0,24), stat_model.arrival_rates_week)
ax1.set_ylim(0,18)
ax1.set_ylabel('Arrivals / hour')
ax1.set_xlabel('Hour of day')
ax1.set_title('Arrival rates for every hour of a weekday')
ax2 = fig.add_subplot(122)
ax2.bar(range(0,24), stat_model.arrival_rates_weekend)
ax2.set_ylim(0,18)
ax2.set_xlabel('Hour of day')
ax2.set_title('Arrival rates for every hour of a \nday during the weekend')


fig = plt.figure(5)
plt.suptitle('Cumulative distribution functions of EVs stay durations every hour of the day')
ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
ax.set_xlabel('Stay durations [h]')
ax.set_ylabel('Probability')
for i in range(24):
    ax = fig.add_subplot(4, 6, i + 1)
    density_array_weekday = stat_model.stay_density_arrays[i]
    density_edges_weekday = stat_model.stay_density_edges[i].tolist()
    density_array_weekday.insert(0,0)
    ax.step(density_edges_weekday, density_array_weekday)
    density_array_weekend = stat_model.stay_density_arrays_weekend[i]
    density_edges_weekend = stat_model.stay_density_edges_weekend[i].tolist()
    density_array_weekend.insert(0, 0)
    ax.step(density_edges_weekend, density_array_weekend)
    ax.set_xlim(-1,40)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(27, ax.get_ylim()[1] - 0.9 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 'Hour {}'.format(i), bbox=props)
