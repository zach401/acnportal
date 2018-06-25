
class Simulator:

    def __init__(self, tc, max_iterations=100):
        self.iteration = 0
        self.last_schedule_update = -1
        self.test_case = tc
        self.scheduler = None
        self.schedules = {}
        self.max_iterations = max_iterations

    def define_scheduler(self, scheduler):
        self.schedules = scheduler

    def run(self):
        while self.iteration < self.max_iterations:
            if self.iteration >= self.last_schedule_update + 1:
                self.scheduler.run()
                self.last_schedule_update = self.iteration
            pilot_signals = self.get_current_pilot_signals()
            self.test_case.step(pilot_signals)

    def get_current_pilot_signals(self):
        '''
        Function for extracting the pilot signals at the current time for the active EVs

        :return: A dictionary where key is the EV id and the value is a number with the charging rate
        '''
        pilot_signals = {}
        for ev_id, sch_list in self.schedules:
            iterations_since_last_update = self.iteration - self.last_schedule_update
            if iterations_since_last_update > len(sch_list):
                pilot = sch_list[-1]
            else:
                pilot = sch_list[iterations_since_last_update]
            pilot_signals[ev_id] = pilot
        return pilot_signals

    def update_schedule(self, new_schedule):
        self.schedules = new_schedule

    def get_active_EVs(self):
        return self.test_case.get_active_EVs()



