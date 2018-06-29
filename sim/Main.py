'''
This is currently the main starting point of the simulator of the
ACN research portal.
'''

import TestCase
from datetime import datetime
from BaseAlgorithm import *
from Interface import Interface
from Simulator import Simulator
from GraphDrawer import GraphDrawer

if __name__ == '__main__':
    '''
    test_case = TestCase.generate_test_case_local('test_session.p',
                                                  datetime.strptime("21/11/06", "%d/%m/%y"),
                                                  datetime.strptime("21/11/22", "%d/%m/%y"))
    '''
    test_case = TestCase.generate_test_case_local('April_2018_Sessions.pkl',
                                                  datetime.strptime("12/04/18", "%d/%m/%y"),
                                                  datetime.strptime("18/04/18", "%d/%m/%y"))


    sim = Simulator(test_case)
    interface = Interface(sim)
    scheduler = EarliestDeadlineFirstAlgorithm(interface)
    sim.define_scheduler(scheduler)

    sim.run()

    #print(sim.get_simulation_data())
    gd = GraphDrawer(sim)
    #gd.draw_charge_rates()
    gd.draw_station_activity()
    gd.plot_EV_behavioral_stats()
    gd.plot_algorithm_result_stats()