'''
This is currently the main starting point of the simulator of the
ACN research portal.
'''

from datetime import datetime
from acnlib import TestCase
from BaseAlgorithm import *
from acnlib.GraphDrawer import GraphDrawer
from acnlib.ACNsim import ACNsim

if __name__ == '__main__':
    test_case = TestCase.generate_test_case_local('April_2018_Sessions.pkl',
                                                  datetime.strptime("09/04/18", "%d/%m/%y"),
                                                  datetime.strptime("22/04/18", "%d/%m/%y"),
                                                  period=5)
    scheduler = MLLF()
    acnsim = ACNsim()

    simulation_output = acnsim.simulate_real(scheduler, test_case)
    #simulation_output = acnsim.simulate_model(scheduler, period=1)

    gd = GraphDrawer(simulation_output)
    gd.plot_station_activity()
    gd.plot_EV_behavioral_stats()
    gd.plot_algorithm_result_stats()
    #gd.print_station_sessions(test_case)
    #gd.plot_EV_stats(test_case, 102)