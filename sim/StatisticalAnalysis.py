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

for s in sessions:
    arrival = s[0]-timedelta(hours=7)
    departure = s[1]-timedelta(hours=7)
    stay_duration = (departure - arrival).total_seconds() / 60
    #if stay_duration >= 0 and stay_duration <= 1000:
    stay_duration_hours[arrival.hour].append(stay_duration)
    stay_duration_list.append(stay_duration)
    energy_demand.append(s[2])


def normal_curve(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


plt.figure(1)
for key, data in stay_duration_hours.items():
    mu = np.mean(data)
    sigma = np.var(data)
    if sigma == 0:
        sigma = 1
    x = key//6
    y = key%6
    plt.subplot(4, 6, key + 1)
    plt.hist(data, bins=20, range=(0, 2000), normed=True)
    xx = np.linspace(0,2000)
    yy = normal_curve(xx, mu, np.sqrt(sigma))
    plt.plot(xx, yy)

plt.figure(2)
plt.scatter(stay_duration_list, energy_demand)
