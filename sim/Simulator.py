import TestCase
from datetime import datetime
import BaseAlgorithm

class Simulator:

    scheduler: BaseAlgorithm

    def __init__(self, test_case, scheduler, max_iterations=100):
        self.iteration = 0
        self.last_schedule_update = -1
        self.test_case = test_case
        self.scheduler = scheduler
        self.schedule = {}
        self.max_iterations = max_iterations
        pass

    def run(self):
        while self.iteration < self.max_iterations:
            if self.iteration >= self.last_schedule_update + 10:
                self.scheduler.run()
                self.last_schedule_update = self.iteration
            pilot_signals = self.get_current_pilots()
            self.test_case.step(pilot_signals)

        pass

    def update_schedule(self, new_schedule):
        self.schedule = new_schedule
        pass

    def get_current_pilots(self):
        pilot_signals = {}
        for ev_id, sch_list in self.schedule:
            iterations_since_last_update = self.iteration - self.last_schedule_update
            if iterations_since_last_update > len(sch_list):
                pilot = sch_list[-1]
            else:
                pilot = sch_list[iterations_since_last_update]
            pilot_signals[ev_id] = pilot
        return pilot_signals


if __name__ == '__main__':
    test_case = TestCase.generate_test_case_local('test_session.p',
                                                  datetime.strptime("21/11/06", "%d/%m/%y"),
                                                  datetime.strptime("21/11/22", "%d/%m/%y"))
    scheduler = BaseAlgorithm()
    sim = Simulator(test_case, scheduler)