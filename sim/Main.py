'''
This is currently the main starting point of the simulator of the
ACN research portal.
'''

import TestCase
from datetime import datetime
from BaseAlgorithm import *
from GraphDrawer import GraphDrawer
from ACNsim import ACNsim

if __name__ == '__main__':
    '''
    test_case = TestCase.generate_test_case_local('test_session.p',
                                                  datetime.strptime("21/11/06", "%d/%m/%y"),
                                                  datetime.strptime("21/11/22", "%d/%m/%y"))
    '''
    test_case = TestCase.generate_test_case_local('April_2018_Sessions.pkl',
                                                  datetime.strptime("18/04/18", "%d/%m/%y"),
                                                  datetime.strptime("20/04/18", "%d/%m/%y"),
                                                  period=5)
    scheduler = MLLF()
    acnsim = ACNsim()

    acnsim.simulate(test_case, scheduler)

    gd = GraphDrawer(test_case)
    gd.plot_station_activity(test_case)
    gd.plot_EV_behavioral_stats(test_case)
    gd.plot_algorithm_result_stats(test_case)
    gd.print_station_sessions(test_case)
    gd.plot_EV_stats(test_case, 88)