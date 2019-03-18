import copy

from event_queue import UnplugEvent
from interface import Interface


class InvalidScheduleError(Exception):
    pass


class Simulator:
    """
    The Simulator class is the central class of the ACN research portal simulator.
    """

    def __init__(self, network, scheduler, events, start, period=1, max_recomp=None, prices=None,
                 store_schedule_history=False):
        self.network = network
        self.scheduler = scheduler
        self.scheduler.register_interface(Interface(self))
        self.event_queue = events
        self.period = period
        self.max_recompute = max_recomp
        self.prices = prices
        self.start = start

        # Local Variables
        self.iteration = 0
        self.resolve = False
        self.last_schedule_update = 0
        self.pilot_signals = {station_id: [] for station_id in self.network.get_space_ids()}
        self.charging_rates = {station_id: [] for station_id in self.network.get_space_ids()}
        self.peak = 0
        self.ev_history = {}
        self.event_history = []
        if store_schedule_history:
            self.schedule_history = {}
        else:
            self.schedule_history = None

    def run(self):
        """
        The most essential function of the Simulator class. Runs the main function of
        the simulation from start to finish and calls for the scheduler when needed.

        :return: The simulation output
        :rtype: SimulationOutput
        """

        while not self.event_queue.empty():
            current_events = self.event_queue.get_current_events(self.iteration)
            for e in current_events:
                self.event_history.append(e)
                self.process_event(e)
            if self.resolve or \
                    self.max_recompute is not None and self.iteration - self.last_schedule_update >= self.max_recompute:
                self.update_schedules(self.scheduler.run())
                self.last_schedule_update = self.iteration
                self.resolve = False
            self._expand_pilots()
            self.network.update_pilots(self.pilot_signals, self.iteration)
            self.update_charging_rates()
            self.iteration = self.iteration + 1

    def process_event(self, event):
        if event.type == 'Plugin':
            print('Plugin Event...')
            self.network.plugin(event.EV, event.EV.station_id)
            self.ev_history[event.EV.session_id] = event.EV
            self.event_queue.add_event(UnplugEvent(event.EV.departure, event.EV.station_id, event.EV.session_id))
            self.resolve = True
            self.last_schedule_update = event.timestamp
        elif event.type == 'Unplug':
            print('Unplug Event...')
            self.network.unplug(event.station_id)
            self.resolve = True
            self.last_schedule_update = event.timestamp
        elif event.type == 'Recompute':
            print('Recompute Event...')
            self.resolve = True

    def update_schedules(self, new_schedule):
        if len(new_schedule) == 0:
            return
        if self.schedule_history is not None:
            self.schedule_history[self.iteration] = new_schedule

        schedule_lengths = set(len(x) for x in new_schedule.values())
        if len(schedule_lengths) > 1:
            raise InvalidScheduleError('All schedules should have the same length.')
        schedule_length = schedule_lengths.pop()

        for station_id in new_schedule:
            if station_id not in self.network.get_space_ids():
                raise KeyError('Station {0} in schedule but not found in network.'.format(station_id))

        for station_id in self.network.get_space_ids():
            if station_id in new_schedule:
                self.pilot_signals[station_id] = _overwrite_at_index(self.iteration, self.pilot_signals[station_id],
                                                                     new_schedule[station_id])
            else:
                # If a station is not in the new schedule, it shouldn't be charging.
                # Extends the pilot signal for all station_schedule so that they all have the same length.
                self.pilot_signals[station_id] = _overwrite_at_index(self.iteration, self.pilot_signals[station_id],
                                                                     [0] * schedule_length)

    def _expand_pilots(self):
        for signal in self.pilot_signals.values():
            if len(signal) < self.iteration + 1:
                signal.append(0)

    def update_charging_rates(self):
        current_rates = self.network.get_current_charging_rates()
        agg = 0
        for station_id, rate in current_rates.items():
            self.charging_rates[station_id].append(rate)
            agg += rate
        self.peak = max(self.peak, agg)

    def get_active_evs(self):
        """
        Returns the current active EVs connected to the system.

        :return:  List of EVs currently plugged in and not finished charging
        :rtype: list
        """
        evs = copy.deepcopy(self.network.active_evs())
        return evs


def _overwrite_at_index(i, prev_list, new_list):
    if len(prev_list) < i:
        return prev_list + [0] * (i - len(prev_list)) + list(new_list)
    if len(prev_list) == i:
        return prev_list + list(new_list)
    else:
        return prev_list[:i] + list(new_list)
