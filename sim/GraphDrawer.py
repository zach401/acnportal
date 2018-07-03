import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math

class GraphDrawer:

    def __init__(self, simulator):
        self.figures = 0
        plt.close('all')

    def new_figure(self):
        plt.figure(self.figures)
        self.figures = self.figures + 1

    def draw_charge_rates(self, test_case):
        '''
        Simple graph to draw the charging rates for every EV against time
        :return: None
        '''
        plt.figure(self.figures)
        test_case_data = test_case.charging_data
        for ev_id, data in test_case_data.items():
            t = []
            value = []
            for sample in data:
                t.append(sample['time'])
                value.append(sample['charge_rate'])
            plt.plot(t, value)
            plt.xlabel('Period')
            plt.ylabel('Charging rate [A]')
        plt.show()
        self.figures = self.figures + 1

    def plot_station_activity(self, test_case):
        '''
        Plots an activity plot of the test case. It shows the session activities at every charging station
        in terms of present EVs and charge rates.
        :return: None
        '''
        plt.figure(self.figures)
        EVs = test_case.EVs
        charging_data = test_case.charging_data
        max_rate = test_case.DEFAULT_MAX_RATE
        for ev in EVs:
            x, y = [ev.arrival, ev.departure - 1], [ev.station_id, ev.station_id]
            plt.plot(x, y, color='y', linewidth=7.0)
            if ev.session_id in charging_data:
                start = ev.arrival
                end = ev.departure
                charge_rate = 0
                time_series = charging_data[ev.session_id]
                counter = 0
                for sample in time_series:
                    if sample['charge_rate'] != charge_rate or counter == len(time_series) - 1:
                        end = sample['time']
                        xc, yc = [start, end], [ev.station_id, ev.station_id]
                        if sample['charge_rate'] != 0:
                            color = (charge_rate/max_rate)
                            plt.plot(xc, yc, color=([1.0, 0.0, 0.0, color]), linewidth=7.0)
                        start = end
                        charge_rate = sample['charge_rate']
                    counter = counter + 1
        plt.xlabel('Time')
        plt.ylabel('Station')
        plt.title('Charging station activity')
        plt.show()
        self.figures = self.figures + 1

    def plot_EV_behavioral_stats(self, test_case):
        '''
        Plot the bahavior of the EVs during a test case.
            Figure 1:
                - A histogram showing the total number of EVs arriving and departing every hour of the day
                - A histogram showing the distribution of energy demand for all the EVs in the test case.
                - A histogram showing the distribution of stay duration of all the EVs in the test case.
        :return: None
        '''
        plt.figure(self.figures)
        arrival_hours = []
        departure_hours = []
        requested_energy = []
        stay_durations = []
        for ev in test_case.EVs:
            # - Gather data for arrivals and departures
            arrival_time = datetime.fromtimestamp(ev.arrival * 60 * test_case.period +
                                                  test_case.start_timestamp)
            departure_time = datetime.fromtimestamp(ev.departure * 60 * test_case.period +
                                                  test_case.start_timestamp)
            arrival_time = arrival_time - timedelta(hours=-7)
            departure_time = departure_time - timedelta(hours=-7)
            arrival_hours.append(arrival_time.hour)
            departure_hours.append(departure_time.hour)
            # - Gather data for requested energy
            requested_energy.append(ev.requested_energy / (60 / test_case.period))
            # - Gather data for stay times
            stay_durations.append(((ev.departure - ev.arrival) * test_case.period) / 60)

        plt.subplot(1,3,1)
        plt.hist([arrival_hours, departure_hours], bins=24)
        plt.xlabel('Hour of day')
        plt.ylabel('Number of EVs')
        plt.title('Arrival and Departure hours of the EVs using the ACN')
        plt.legend(['Arrivals', 'Departures'])
        plt.subplot(1, 3, 2)
        plt.hist(requested_energy, bins=15, edgecolor='black')
        plt.xlabel('Requested energy [Ah]')
        plt.ylabel('Number of EVs')
        plt.title('Requested energy by the EVs using the ACN')
        plt.subplot(1, 3, 3)
        plt.hist(stay_durations, bins=15, edgecolor='black')
        plt.xlabel('Parking duration [hours]')
        plt.ylabel('Number of EVs')
        plt.title('Parking duration of the EVs using the ACN')
        self.figures = self.figures + 1

    def plot_algorithm_result_stats(self, test_case):
        '''
        Plots the results after the simulation has been run.
            Figure 1:
                - A histogram showing the distribution of how many percent the EV requested energy has been met.
                - A histogram showing the the stay time for the EVs that did not finsh charging.
            Figure 2:
                - A line graph of the total current draw of the test case.
        :return:
        '''
        plt.figure(self.figures)
        energy_percentage = []
        stay_duration_not_finished_EVs = []
        total_current = [0] * math.ceil(test_case.last_departure)
        for ev in test_case.EVs:
            # - Calculate the percentage of requested energy met
            percentage = (ev.energy_delivered / ev.requested_energy) * 100
            if percentage > 100:
                percentage = 100
            energy_percentage.append(percentage)
            # - Calculate the stay time of EVs not fully charged
            if ev.remaining_demand > 0:
                stay_duration_not_finished_EVs.append(((ev.departure - ev.arrival) * test_case.period) / 60)
            # - Accumulate the total current used by all sessions
            for sample in test_case.charging_data[ev.session_id]:
                total_current[sample['time']] = total_current[sample['time']] + sample['charge_rate']


        plt.subplot(1, 2, 1)
        plt.hist(energy_percentage, bins=50, edgecolor='black', range=(0,100))
        plt.xlabel('Percentage of requested energy received')
        plt.ylabel('Number of EVs')
        plt.title('How much of the EVs energy demand that was met')
        plt.subplot(1, 2, 2)
        plt.hist(stay_duration_not_finished_EVs, bins=20, edgecolor='black')
        plt.xlabel('Stay duration [hours]')
        plt.ylabel('Number of EVs not fully charged')
        plt.title('Stay duration of EVs that did not get their requested energy fulfilled')
        self.figures = self.figures + 1
        self.new_figure()
        plt.plot(total_current)
        plt.xlabel('time')
        plt.ylabel('Current draw [A]')
        plt.title('Total current draw of the test case')
