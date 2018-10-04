import copy

from acnlib.SimulationOutput import SimulationOutput, Event
from acnlib.event_queue import UnplugEvent
from acnlib.EV import EV


class Simulator:
    '''
    The Simulator class is the central class of the ACN research portal simulator.
    '''

    def __init__(self, garage, scheduler, max_iterations=3000):
        self.garage = garage
        self.scheduler = scheduler
        self.schedules = {}
        self.last_applied_pilot_signals = {}
        self.max_iterations = max_iterations
        self.iteration = 0
        self.last_schedule_update = -1

    def run(self):
        '''
        The most essential function of the Simulator class. Runs the main function of
        the simulation from start to finish and calls for the scheduler when needed.

        :return: The simulation output
        :rtype: SimulationOutput
        '''
        self.submit_event(Event('INFO', self.iteration, 'Simulation started'))
        schedule_horizon = 0
        while self.iteration < self.garage.last_departure:  # TODO(zlee): last_departure should be a member of test_case
            if self.iteration >= self.last_schedule_update + schedule_horizon or self.garage.event_occurred(  # TODO(zlee): schedule_horizon should be minimum_resolve_period
                    self.iteration):  # TODO(zlee): event_occured should come from test_case
                # call the scheduling algorithm
                self.scheduler.run()  # TODO(zlee): this should probably return something rather than using the interface behind the scenes
                self.last_schedule_update = self.iteration
                schedule_horizon = self.get_schedule_horizon()  # TODO(zlee): this about this, should horizon be set by algorithm in this way, or configured beforehand
                self.submit_event(Event('INFO', self.iteration, 'New charging schedule calculated'))
            pilot_signals = self.get_current_pilot_signals()
            self.garage.update_state(pilot_signals, self.iteration)
            self.last_applied_pilot_signals = pilot_signals
            self.iteration = self.iteration + 1
        self.submit_event(Event('INFO', self.iteration, 'Simulation finished'))
        simulation_output = self.garage.get_simulation_output()
        # charging_data = self.test_case.get_charging_data()
        return simulation_output

    def submit_event(self, event):  # TODO(zlee): move simulation output to simulation, perhaps each level gets a reference though
        self.garage.test_case.simulation_output.submit_event(event)

    def submit_log_event(self, text):
        event = Event('LOG', self.iteration, text, 'ALGORITHM')
        self.garage.test_case.simulation_output.submit_event(event)

    def get_current_pilot_signals(self):
        '''
        Function for extracting the pilot signals at the current time for the active EVs

        :return: A dictionary where key is the EV id and the value is a number with the charging rate.
        :rtype: dict
        '''
        pilot_signals = {}
        for ev_id, sch_list in self.schedules.items():
            iterations_since_last_update = self.iteration - self.last_schedule_update
            if iterations_since_last_update > len(sch_list):
                pilot = sch_list[-1]
            else:
                pilot = sch_list[iterations_since_last_update]
            pilot_signals[ev_id] = pilot
        return pilot_signals

    def get_last_applied_pilot_signals(self):
        return self.last_applied_pilot_signals

    def get_last_actual_charging_rate(self):
        self.garage.get_actual_charging_rates()

    def get_schedule_horizon(self):
        min_horizon = 0
        for ev_id, sch_list in self.schedules.items():
            if min_horizon > len(sch_list):
                min_horizon = len(sch_list)
        return min_horizon

    def update_schedules(self, new_schedule):                                                                           # TODO(zlee): this should not be nessesary when we update the interface
        '''
        Update the schedules used in the simulation.
        This function is called by the interface to the scheduling algorithm.

        :param dict new_schedule: Dictionary where key is the id of the EV and value is a list of scheduled charging rates.
        :return: None
        '''
        self.schedules = new_schedule

    def get_active_EVs(self):
        '''
        Returns the current active EVs connected to the system.

        :return:  List of EVs currently plugged in and not finished charging
        :rtype: list
        '''
        EVs = copy.deepcopy(self.garage.get_active_EVs(self.iteration))
        return EVs

    def get_simulation_data(self):
        '''
        Returns the data from the simulation.
        :return: Dictionary where key is the id of the EV and value is a list of dicts representing every sample.
        :rtype: dict
        '''
        return self.garage.get_charging_data()
