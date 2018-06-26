import TestCase
from datetime import datetime
from BaseAlgorithm import *
from Interface import Interface
from Simulator import Simulator

if __name__ == '__main__':
    test_case = TestCase.generate_test_case_local('test_session.p',
                                                  datetime.strptime("21/11/06", "%d/%m/%y"),
                                                  datetime.strptime("21/11/22", "%d/%m/%y"))
    sim = Simulator(test_case)
    interface = Interface(sim)
    scheduler = EarliestDeadlineFirstAlgorithm(interface)
    sim.define_scheduler(scheduler)

    sim.run()

    print(sim.get_simulation_data())