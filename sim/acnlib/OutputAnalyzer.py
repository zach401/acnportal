import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
import numpy as np

class OutputAnalyzer:

    def __init__(self, simulation_output):
        self.simulation_output = simulation_output
        self.figures = 0
        plt.close('all')

    def set_simulation_output(self, simulation_output):
        self.simulation_output = simulation_output

    def new_figure(self):
        plt.figure(self.figures)
        self.figures = self.figures + 1

    def plot_station_activity(self):
        '''
        Plots an activity plot of the test case. It shows the session activities at every charging station
        in terms of present EVs and charge rates.

        :return: None
        '''
        plt.figure(self.figures)
        EVs = self.simulation_output.EVs
        charging_data = self.simulation_output.charging_data
        max_rate = self.simulation_output.max_rate
        for ev in EVs:
            if ev.fully_charged:
                x, y = [ev.arrival, ev.finishing_time], [ev.station_id, ev.station_id]
                x2, y2 = [ev.finishing_time, ev.departure - 1], [ev.station_id, ev.station_id]
                plt.plot(x, y, color='y', linewidth=7.0)
                plt.plot(x2, y2, color='g', linewidth=7.0)
            else:
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
                        start = end + 0.01
                        charge_rate = sample['charge_rate']
                    counter = counter + 1
            #if ev.finishing_time > 0:
                #plt.plot(ev.finishing_time, ev.station_id,'ko')
        plt.xlabel('Time')
        plt.ylabel('Station')
        plt.title('Charging station activity')
        plt.show()
        self.figures = self.figures + 1

    def plot_EV_behavioral_stats(self):
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
        for ev in self.simulation_output.EVs:
            # - Gather data for arrivals and departures
            arrival_time = datetime.fromtimestamp(ev.arrival * 60 * self.simulation_output.period +
                                                  self.simulation_output.start_timestamp)
            departure_time = datetime.fromtimestamp(ev.departure * 60 * self.simulation_output.period +
                                                    self.simulation_output.start_timestamp)
            arrival_time = arrival_time #- timedelta(hours=7)
            departure_time = departure_time #- timedelta(hours=7)
            arrival_hours.append(arrival_time.hour)
            departure_hours.append(departure_time.hour)
            # - Gather data for requested energy
            requested_energy.append((ev.requested_energy / (60 / self.simulation_output.period))*self.simulation_output.voltage/1000)
            # - Gather data for stay times
            stay_durations.append(((ev.departure - ev.arrival) * self.simulation_output.period) / 60)

        mean_stay_duration = np.mean(stay_durations)
        mean_energy_requested = np.mean(requested_energy)

        plt.subplot(1,3,1)
        plt.hist([arrival_hours, departure_hours], bins=24)
        plt.xlabel('Hour of day')
        plt.ylabel('Number of EVs')
        plt.title('Distribution of Arrival and Departure hours of the EVs using\n the ACN during the period April 9 - April 22 2018')
        plt.legend(['Arrivals', 'Departures'])
        plt.subplot(1, 3, 2)
        plt.hist(stay_durations, bins=15,edgecolor='black', range=(0, 40))
        plt.axvline(mean_stay_duration, color='r', linestyle='dashed', linewidth=1)
        plt.xlabel('Parking duration [hours]')
        plt.ylabel('Number of EVs')
        plt.title('Distribution of parking duration of the EVs using\n the ACN during the period April 9 - April 22 2018')
        plt.subplot(1, 3, 3)
        plt.hist(requested_energy, bins=15, edgecolor='black')
        plt.axvline(mean_energy_requested, color='r', linestyle='dashed', linewidth=1)
        plt.xlabel('Requested energy [kWh]')
        plt.ylabel('Number of EVs')
        plt.title('Distribution of requested energy by the EVs using\n the ACN during the period April 9 - April 22 2018')
        self.figures = self.figures + 1

    def plot_algorithm_result_stats(self):
        '''
        Plots the results of the simulation.

        Figure 1:
            - A histogram showing the distribution of how many percent the EV requested energy has been met.
            - A histogram showing the stay time for the EVs that did not finish charging.
        Figure 2:
            - A line graph of the total power usage of the test case.

        :return: None
        '''
        plt.figure(self.figures)
        energy_percentage = []
        stay_duration_not_finished_EVs = []
        total_current = [0] * math.ceil(self.simulation_output.last_departure)
        for ev in self.simulation_output.EVs:
            # - Calculate the percentage of requested energy met
            percentage = (ev.energy_delivered / ev.requested_energy) * 100
            if percentage > 100:
                percentage = 100
            energy_percentage.append(percentage)
            # - Calculate the stay time of EVs not fully charged
            if not ev.fully_charged:
                stay_duration_not_finished_EVs.append(((ev.departure - ev.arrival) * self.simulation_output.period) / 60)
            # - Accumulate the total current used by all sessions
            for sample in self.simulation_output.charging_data[ev.session_id]:
                total_current[sample['time']] = total_current[sample['time']] + sample['charge_rate']*self.simulation_output.voltage/1000


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
        plt.ylabel('Power [kW]')
        plt.title('Power usage of the test case')

    def plot_EV_stats(self, session_id):
        '''
        Plot the charging rate and applied pilot signal of an EV charging session.

        :param int session_id: The ID of the session evaluated
        :return: None
        '''
        data = self.simulation_output.charging_data[session_id]
        time = []
        pilot_signal = []
        charge_rate = []
        remaining_demand = []
        for sample in data:
            time.append(sample['time'])
            pilot_signal.append(sample['pilot_signal'])
            charge_rate.append(sample['charge_rate'])
            remaining_demand.append(sample['remaining_demand'])
        self.new_figure()
        plt.subplot(2, 1, 1)
        plt.plot(time, pilot_signal)
        plt.plot(time, charge_rate)
        plt.subplot(2, 1, 2)
        plt.plot(time, remaining_demand)

    def print_station_sessions(self):
        '''
        Print all the EVSEs used in the simulation and the charging sessions belonging to each EVSE.

        :return: None
        '''
        stations = {}
        for ev in self.simulation_output.EVs:
            if ev.station_id in stations:
                stations[ev.station_id].append(ev.session_id)
            else:
                stations[ev.station_id] = [ev.session_id]

        sorted_stations = sorted(stations)
        for station_id, sessions in sorted(stations.items()):
            line = str(station_id) + ': '
            for element in sessions:
                line = line + str(element) + ', '
            line = line + '\n'
            print(line)

    def print_events(self, type='all'):
        '''
        Print the simulation events in the terminal window.

        :param string type: What type of events that should be printed.
            Available values: ('all', 'warning', 'error', 'log')
        :return: None
        '''
        events = []
        if type == 'all':
            events = self.simulation_output.get_events_all()
            print('Printing all simulation events. Total number of events: {}'.format(len(events)))
        elif type == 'info':
            events = self.simulation_output.get_events_info()
            print('Printing INFO events. Number of INFO events: {}'.format(len(events)))
        elif type == 'warning':
            events = self.simulation_output.get_events_warnings()
            print('Printing WARNING events. Number of WARNING events: {}'.format(len(events)))
        elif type == 'error':
            events = self.simulation_output.get_events_errors()
            print('Printing ERROR events. Number of ERROR events: {}'.format(len(events)))
        elif type == 'log':
            events = self.simulation_output.get_events_errors()
            print('Printing LOG events. Number of LOG events: {}'.format(len(events)))

        print('-'*60)
        for e in events:
            print('{0:5d} | {1:6s} | {2:8s} | {3}'.format(e.iteration, e.type, str(e.session), e.description))

        if len(events) == 0:
            print('No events')

        print('-' * 60)

        if type == 'all':
            print('Total number of events: {}'.format(len(events)))
        elif type == 'info':
            print('Number of INFO events: {}'.format(len(events)))
        elif type == 'warning':
            print('Number of WARNING events: {}'.format(len(events)))
        elif type == 'error':
            print('Number of ERROR events: {}'.format(len(events)))
        elif type == 'log':
            print('Number of LOG events: {}'.format(len(events)))

        print('\n')


    def create_time_axis(self, start_timestamp, nbr_of_periods, period):
        time_array = []
        for i in range(int(nbr_of_periods)):
            time = datetime.fromtimestamp(start_timestamp + i * period)
            time_array.append(time)
        return time_array