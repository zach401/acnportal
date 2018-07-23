'''
This is currently the main starting point of the simulator of the
ACN research portal.
'''

from datetime import datetime
import TestCase
from BaseAlgorithm import *
from GraphDrawer import GraphDrawer
from ACNsim import ACNsim

if __name__ == '__main__':
    test_case = TestCase.generate_test_case_local('April_2018_Sessions.pkl',
                                                  datetime.strptime("20/04/18", "%d/%m/%y"),
                                                  datetime.strptime("25/04/18", "%d/%m/%y"),
                                                  period=5)
    scheduler = MLLF()
    acnsim = ACNsim()

    #test_case = acnsim.simulate_real(scheduler, test_case)
    simulation_output = acnsim.simulate_model(scheduler, period=1)

    gd = GraphDrawer()
    gd.plot_station_activity(simulation_output)
    gd.plot_EV_behavioral_stats(simulation_output)
    gd.plot_algorithm_result_stats(simulation_output)
    #gd.print_station_sessions(test_case)
    #gd.plot_EV_stats(test_case, 102)