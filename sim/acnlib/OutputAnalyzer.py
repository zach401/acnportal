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

    def plot_EV_behavioral_stats(self, percentage=True):
        '''
        Plot the bahavior of the EVs during a test case.

        Figure 1:
            - A histogram showing the total number of EVs arriving and departing every hour of the day
            - A histogram showing the distribution of energy demand for all the EVs in the test case.
            - A histogram showing the distribution of stay duration of all the EVs in the test case.

        :param boolean percentage: Specifies if the histograms should be in percentage or in absolute value.
        :return: None
        '''
        plt.figure(self.figures)
        arrival_hours = []
        departure_hours = []
        requested_energy = []
        stay_durations = []
        number_of_EVs = len(self.simulation_output.EVs)
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

        mean_stay_duration = round(np.mean(stay_durations), 2)
        mean_energy_requested = round(np.mean(requested_energy), 2)
        std_stay_duration = round(np.sqrt(np.var(stay_durations)), 2)
        std_energy_requested = round(np.sqrt(np.var(requested_energy)), 2)

        arrival_hours_percentage = [x / number_of_EVs * 100 for x in arrival_hours]


        ax1 = plt.subplot(1,3,1)
        #plt.hist([arrival_hours, departure_hours], bins=24)
        hist_arrival, bins_arrival = np.histogram(arrival_hours, bins=24, range=(0,24))
        hist_departure, bins_departure = np.histogram(departure_hours, bins=24, range=(0,24))
        b1 = ax1.bar(bins_arrival[:-1],
                hist_arrival.astype(np.float32) / ((hist_arrival.sum() / 100) if percentage else 1),
                width=-(bins_arrival[1] - bins_arrival[0])/2,
                edgecolor=['black'] * len(hist_arrival),
                align='edge')
        b2 = ax1.bar(bins_departure[:-1],
                hist_departure.astype(np.float32) / ((hist_departure.sum() / 100) if percentage else 1),
                width=(bins_departure[1] - bins_departure[0]) / 2,
                edgecolor=['black'] * len(bins_departure),
                align='edge')
        handles, labels = ax1.get_legend_handles_labels()
        ax1.set_ylim(0, 25)
        ax1.legend([b1, b2], ['Arrivals', 'Departures'])
        plt.xlabel('Hour of day')
        plt.ylabel('Percentage of arriving and departing EVs [%]' if percentage else 'Number of EVs')
        plt.title('Distribution of Arrival and Departure hours of the EVs using\n the ACN during the period ' +
                  self.__get_simulation_duration_date_string())
        #plt.legend(['Arrivals', 'Departures'])
        plt.subplot(1, 3, 2)
        #plt.hist(stay_durations, bins=15,edgecolor='black', range=(0, 40))
        hist, bins = np.histogram(stay_durations, bins=20, range=(0, 40))
        plt.bar(bins[:-1],
                hist.astype(np.float32) / ((hist.sum() / 100) if percentage else 1),
                width=(bins[1] - bins[0]),
                edgecolor=['black']*len(hist),
                align='edge')
        plt.axvline(mean_stay_duration,
                    color='r',
                    linestyle='dashed',
                    linewidth=1,
                    label='Average stay duration: {} h,\nSTD: {} h'.format(mean_stay_duration, std_stay_duration))
        plt.xlabel('Parking duration [hours]')
        plt.ylabel('Percentage of EVs [%]' if percentage else 'Number of EVs')
        plt.title('Distribution of parking duration of the EVs using\n the ACN during the period ' +
                  self.__get_simulation_duration_date_string())
        plt.legend()
        plt.subplot(1, 3, 3)
        #plt.hist(requested_energy, bins=15, edgecolor='black')
        hist, bins = np.histogram(requested_energy, bins=20, range=(0, 40))
        plt.bar(bins[:-1],
                hist.astype(np.float32) / ((hist.sum() / 100) if percentage else 1),
                width=(bins[1] - bins[0]),
                edgecolor=['black'] * len(hist),
                align='edge')
        plt.axvline(mean_energy_requested,
                    color='r',
                    linestyle='dashed',
                    linewidth=1,
                    label='Average requested energy: {} kWh,\nSTD: {} kWh'.format(mean_energy_requested, std_energy_requested))
        plt.xlabel('Requested energy [kWh]')
        plt.ylabel('Percentage of EVs [%]' if percentage else 'Number of EVs')
        plt.title('Distribution of requested energy by the EVs using\n the ACN during the period ' +
                  self.__get_simulation_duration_date_string())
        plt.legend()
        self.figures = self.figures + 1

    def plot_EV_daily_arrivals(self, percentage=True):
        '''
        Plots the average daily arrivals of a day during the week and the weekend.
        Figure 1:
            - The distribution of the number of arrivals per day
        Figure 2:
            - The average number of arrivals during the week and the weekend
            and the standard deviation

        :param boolean percentage: If the histograms should be presented in percentage or absolute values.
        :return: None
        '''
        weekdays = {}
        weekends = {}

        period = self.simulation_output.get_period()
        start_timestamp = self.simulation_output.get_start_timestamp()

        for ev in self.simulation_output.get_all_EVs():
            d = datetime.fromtimestamp(start_timestamp + ev.arrival * period * 60).date()
            d_str = d.strftime('%y-%m-%d')
            if d.weekday() < 5:
                if d_str in weekdays:
                    weekdays[d_str] = weekdays[d_str] + 1
                else:
                    weekdays[d_str] = 1
            else:
                if d_str in weekends:
                    weekends[d_str] = weekends[d_str] + 1
                else:
                    weekends[d_str] = 1

        weekdays_arrivals = []
        weekends_arrivals = []
        for key, value in weekdays.items():
            weekdays_arrivals.append(value)
        for key, value in weekends.items():
            weekends_arrivals.append(value)

        weekdays_mean = round(np.mean(weekdays_arrivals), 0)
        weekdays_std = round(np.sqrt(np.var(weekdays_arrivals)), 2)
        weekends_mean = round(np.mean(weekends_arrivals), 0)
        weekends_std = round(np.sqrt(np.var(weekends_arrivals)), 2)

        self.new_figure()
        ax1 = plt.subplot(1,2,1)
        #ax1.hist(weekdays_arrivals, range=(0,70), bins=15, edgecolor='black')
        self.__plot_histogram(ax1, weekdays_arrivals, percentage=percentage, bins=20, range=(0,100), align='center')
        ax1.axvline(weekdays_mean,
                    color='r',
                    linestyle='dashed',
                    linewidth=1,
                    label='Average value: {}\nSTD: {}'.format(weekdays_mean, weekdays_std))
        ax1.set_xlabel('Numbers of EVs arriving per day')
        ax1.set_ylabel('Percentage of days [%]' if percentage else 'Number of days')
        ax1.set_title('Distribution of the daily number of EVs arriving during a weekday,\n(' +
                     self.__get_simulation_duration_date_string()+')')
        ax1.legend()
        ax2 = plt.subplot(1,2,2)
        #ax2.hist(weekends_arrivals, range=(0,70), bins=15, edgecolor='black')
        self.__plot_histogram(ax2, weekends_arrivals, percentage=percentage, bins=20, range=(0, 100), align='center')
        ax2.axvline(weekends_mean,
                    color='r',
                    linestyle='dashed',
                    linewidth=1,
                    label='Average value: {}\nSTD: {}'.format(weekends_mean, weekends_std))
        ax2.set_xlabel('Number of EVs arriving per day')
        ax2.set_ylabel('Percentage of days [%]' if percentage else 'Number of days')
        ax2.set_title('Distribution of the daily number of EVs arriving during a day of the weekend,\n(' +
                     self.__get_simulation_duration_date_string()+')')
        ax2.legend()

        fig, ax = plt.subplots()
        bar1 = ax.bar(['Weekdays', 'Weekends'], [weekdays_mean, 0], width=0.95)
        bar2 = ax.bar(['Weekdays', 'Weekends'], [0, weekends_mean], width=0.95, color='grey')
        plt.errorbar(0, weekdays_mean, yerr=weekdays_std, fmt='ro', ecolor='r')
        plt.errorbar(1, weekends_mean, yerr=weekends_std, fmt='ro', ecolor='r')
        ax.legend((bar1, bar2), ('Average value: {}, STD: {}'.format(weekdays_mean, weekdays_std),
                                 'Average value: {}, STD: {}'.format(weekends_mean, weekends_std)))
        ax.set_ylabel('Average number of EVs arriving per day')
        ax.set_title('Average number of EVs arriving per day of a\n weekday or a day during the weekend,\n(' +
                     self.__get_simulation_duration_date_string()+')')
        #ax.set_xticks(2)
        #ax.set_xticklabels(('Weekdays', 'Weekends'))



    def plot_algorithm_result_stats(self, percentage=True):
        '''
        Plots the results of the simulation.

        Figure 1:
            - A histogram showing the distribution of how many percent the EV requested energy has been met.
            - A histogram showing the stay time for the EVs that did not finish charging.
        Figure 2:
            - A line graph of the total power usage of the test case.

        :param boolean percentage: Specifies if the histograms should be in percentage or in absolute value.
        :return: None
        '''
        plt.figure(self.figures)
        energy_percentage = []
        stay_duration_not_finished_EVs = []
        total_power = [0] * math.ceil(self.simulation_output.last_departure)
        for ev in self.simulation_output.EVs:
            # - Calculate the percentage of requested energy met
            p = (ev.energy_delivered / ev.requested_energy) * 100
            if p > 100:
                p = 100
            energy_percentage.append(p)
            # - Calculate the stay time of EVs not fully charged
            if not ev.fully_charged:
                stay_duration_not_finished_EVs.append(((ev.departure - ev.arrival) * self.simulation_output.period) / 60)
            # - Accumulate the total current used by all sessions
            if ev.session_id in self.simulation_output.charging_data:
                for sample in self.simulation_output.charging_data[ev.session_id]:
                    total_power[sample['time']] = total_power[sample['time']] + sample['charge_rate']*self.simulation_output.voltage/1000

        plt.subplot(1, 2, 1)
        #plt.hist(energy_percentage, bins=50, edgecolor='black', range=(0,100))
        self.__plot_histogram(plt, energy_percentage, percentage=True, bins=50, range=(0,100))
        plt.xlabel('Percentage of requested energy received')
        plt.ylabel('Percentage of EVs [%]' if percentage else 'Number of EVs')
        plt.title('How much of the EVs energy demand that was met')
        plt.subplot(1, 2, 2)
        #plt.hist(stay_duration_not_finished_EVs, bins=20, edgecolor='black')
        self.__plot_histogram(plt, stay_duration_not_finished_EVs, percentage=True, bins=20, range=(0, 40))
        plt.xlabel('Stay duration [hours]')
        plt.ylabel('Percentage of EVs not fully charged [%]' if percentage else 'Number of EVs not fully charged')
        plt.title('Stay duration of EVs that did not get their requested energy fulfilled')
        self.figures = self.figures + 1
        self.new_figure()
        plt.plot(total_power)
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

    def print_total_number_of_sessions(self):
        n = len(self.simulation_output.get_all_EVs())
        print('Total numbers of EV charging sessions: {}'.format(n))
        print('\n')

    def print_energy_delivery(self):
        '''
        Prints the energy delivered per day and in total

        :return: None
        '''
        daily_energy = {}
        start_timestamp = self.simulation_output.start_timestamp
        period = self.simulation_output.period
        total_power_series = self.__get_total_power_series()
        total_energy = 0
        for iteration, power in enumerate(total_power_series):
            d = datetime.fromtimestamp(start_timestamp + iteration * period * 60).date()
            energy = power * (period / 60)
            if d in daily_energy:
                daily_energy[d] = daily_energy[d] + energy
            else:
                daily_energy[d] = energy
            total_energy = total_energy + energy

        print('ENERGY DELIVERED')
        print('-' * 22)
        for key, value in daily_energy.items():
            print('{0:9s}: {1:7.1f} kWh'.format(key.strftime('%m/%d/%y'), value))
        print('-' * 22)
        print('{0:9s}: {1:7.1f} kWh'.format('TOTAL', total_energy))
        print('\n')

    def print_energy_fulfillment(self):
        nbr_EVs_fully_charged = 0
        not_fully_charged = []
        for ev in self.simulation_output.EVs:
            if ev.fully_charged:
                nbr_EVs_fully_charged = nbr_EVs_fully_charged + 1
            else:
                not_fully_charged.append(ev.energy_delivered / ev.requested_energy)
        hist, bins = np.histogram(not_fully_charged, bins=10, range=(0,1))
        total_nbr_EVs = hist.sum() + nbr_EVs_fully_charged
        hist_norm = hist.astype(np.float32) / (total_nbr_EVs / 100)
        i = 0
        print('ENERGY FULFILLMENT')
        print('-'*40)
        print('{0:15s} | {1:10s}'.format('Percentage of demand', 'Percentage of EVs'))
        for v in hist_norm.tolist():
            print('{0:12.1f} -{1:5.1f}% | {2:5.2f}%'.format(bins[i]*100, bins[i+1]*100-0.1, v))
            i = i+1
        print('{0:19.1f}% | {1:5.2f}%'.format(100, (nbr_EVs_fully_charged / total_nbr_EVs)*100))
        print('-' * 40)
        print('\n')

    def print_algorithm_result_report(self):
        '''
        Prints a compilation of simulation results. The information included in the report are:
        - Total number of sessions
        - Energy delivery
        - Error events
        - Warning events
        :return:
        '''
        print('\n')
        self.print_total_number_of_sessions()
        self.print_energy_delivery()
        self.print_energy_fulfillment()
        self.print_events(type='error')
        self.print_events(type='warning')

    def __get_total_power_series(self):
        total_power = [0] * math.ceil(self.simulation_output.last_departure)
        for ev in self.simulation_output.EVs:
            if ev.session_id in self.simulation_output.charging_data:
                for sample in self.simulation_output.charging_data[ev.session_id]:
                    total_power[sample['time']] = total_power[sample['time']] + sample['charge_rate']*self.simulation_output.voltage/1000
        return total_power

    def __create_time_axis(self, start_timestamp, nbr_of_periods, period):
        time_array = []
        for i in range(int(nbr_of_periods)):
            time = datetime.fromtimestamp(start_timestamp + i * period)
            time_array.append(time)
        return time_array

    def __get_simulation_duration_date_string(self):
        '''
        Get a string representation of the simulation duration.

        Used in the graphs to write titles.

        :return: A string with the simulation duration
        :rtype: string
        '''
        st = self.simulation_output.get_start_timestamp()
        p = self.simulation_output.get_period()
        ld = self.simulation_output.get_last_arrival()
        start = datetime.fromtimestamp(st)
        end = datetime.fromtimestamp(st + p * ld * 60)
        text = start.strftime('%m/%d/%y') + ' to ' + end.strftime('%m/%d/%y')
        return text

    def __plot_histogram(self, ax, data, percentage=True, bins=20, range=None, align='edge'):
        hist, bins = np.histogram(data, bins=bins, range=range)
        bar = ax.bar(bins[:-1],
                hist.astype(np.float32) / ((hist.sum() / 100) if percentage else 1),
                width=(bins[1] - bins[0]),
                edgecolor=['black'] * len(hist),
                align=align)
        return bar