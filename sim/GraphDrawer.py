import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class GraphDrawer:

    def __init__(self, simulator):
        self.simulator = simulator
        self.figures = 0
        plt.close('all')

    def draw_charge_rates(self):
        plt.figure(self.figures)
        test_case_data = self.simulator.get_simulation_data()
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

    def draw_station_activity(self):
        plt.figure(self.figures)
        EVs = self.simulator.test_case.EVs
        charging_data = self.simulator.test_case.charging_data
        max_rate = self.simulator.test_case.DEFAULT_MAX_RATE
        for ev in EVs:
            x, y = [ev.arrival, ev.departure], [ev.station_id, ev.station_id]
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
        plt.show()
        self.figures = self.figures + 1

    def draw_EV_behavioral_stats(self):
        plt.figure(self.figures)

        arrival_hours = []
        departure_hours = []
        requested_energy = []
        stay_durations = []
        for ev in self.simulator.test_case.EVs:
            arrival_time = datetime.fromtimestamp(ev.arrival * 60 * self.simulator.test_case.period +
                                                  self.simulator.test_case.start_timestamp)
            departure_time = datetime.fromtimestamp(ev.departure * 60 * self.simulator.test_case.period +
                                                  self.simulator.test_case.start_timestamp)
            arrival_time = arrival_time - timedelta(hours=-7)
            departure_time = departure_time - timedelta(hours=-7)
            arrival_hours.append(arrival_time.hour)
            departure_hours.append(departure_time.hour)
            requested_energy.append(ev.requested_energy / (60 / self.simulator.test_case.period))
            stay_durations.append(((ev.departure - ev.arrival) * self.simulator.test_case.period) / 60)

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