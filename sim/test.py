'''
This is a script for testing different parts of the simulation module and
draw some useful plots.
'''

from datetime import datetime

import matplotlib.pyplot as plt
from BaseAlgorithm import *
from acnlib import StatModel
from acnlib import TestCase
from acnlib.ACNsim import ACNsim
from acnlib.Garage import Garage


# tc = TestCase.generate_test_case_local()
def __datetime_format_func(self, value, tick_number):
    return self.__datetime_string_from_iteration(value)


def __datetime_string_from_iteration(self, timestamp):
    st = self.simulation_output.start_timestamp
    period = self.simulation_output.period
    dt = datetime.fromtimestamp(st + timestamp * period * 60)
    return self.__get_datetime_string(dt)


def __get_datetime_string(self, dt):
    return dt.strftime('%H:%M %m/%d/%y')


def filter_total(series):
    b = 200
    h = 200
    filtered = []
    for i in range(0, len(series)):
        sum = 0
        below = i - b
        if below < 0:
            below = 0
        top = i + h
        if top >= len(series):
            top = len(series)
        for j in range(below, top):
            sum = sum + series[j] / (h + b + 1)
        filtered.append(sum)
    return filtered


if __name__ == '__main__':

    garage = Garage()
    test_case_model = garage.generate_test_case(datetime.strptime("02/06/18", "%d/%m/%y"),
                                                datetime.strptime("09/06/18", "%d/%m/%y"),
                                                period=1)
    test_case_real = TestCase.generate_test_case_local('July_25_Sessions.pkl',
                                                       datetime.strptime("02/06/18", "%d/%m/%y"),
                                                       datetime.strptime("09/06/18", "%d/%m/%y"),
                                                       period=1)
    test_case_real2 = TestCase.generate_test_case_local('July_25_Sessions.pkl',
                                                        datetime.strptime("02/06/18", "%d/%m/%y"),
                                                        datetime.strptime("09/06/18", "%d/%m/%y"),
                                                        period=1)
    StatModel.compare_model_to_real(test_case_real, test_case_model)
    print(len(test_case_model.EVs))
    print(len(test_case_real.EVs))

    scheduler = MaxRateAlgorithm()
    scheduler2 = MLLF(queue_length=7)
    acnsim = ACNsim()
    simout1 = acnsim.simulate_real(scheduler, test_case_real)
    simout2 = acnsim.simulate_real(scheduler2, test_case_real2)

    network_data_1 = simout1.get_network_data()
    network_data_2 = simout2.get_network_data()

    total_current1 = []
    for sample in network_data_1:
        total_current1.append(sample['total_current'] * simout1.voltage / 1000)

    total_current2 = []
    for sample in network_data_2:
        total_current2.append(sample['total_current'] * simout2.voltage / 1000)

    fig = plt.figure(0)
    ax = fig.add_subplot('111')
    ax.plot(range(0, len(total_current1)), total_current1, range(0, len(total_current2)), total_current2)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('Total Power [kWh]', fontsize=20)
    ax.set_title('Total Power usage of the simulated network', fontsize=20)
    ax.legend(('Regular charging', 'Adaptive Charging'), fontsize=16)

    filt = filter_total(total_current1)
    fig = plt.figure(1)
    ax = fig.add_subplot('111')
    ax.plot(range(0, len(total_current1)), total_current1, range(0, len(filt)), filt)
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('Total Power [kWh]', fontsize=20)
    ax.set_title('Total Power usage of the simulated network', fontsize=20)
    ax.legend(('Regular charging', 'Desired charging'), fontsize=16)
