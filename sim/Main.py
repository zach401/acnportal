'''
This is currently the main starting point of the simulator of the
ACN research portal.
'''

from datetime import datetime, timedelta
from acnlib import TestCase
from BaseAlgorithm import *
from acnlib.OutputAnalyzer import OutputAnalyzer
from acnlib.ACNsim import ACNsim

if __name__ == '__main__':
    test_case = TestCase.generate_test_case_local('July_25_Sessions.pkl',
                                                  datetime.strptime("01/06/18", "%d/%m/%y"),
                                                  datetime.strptime("30/06/18", "%d/%m/%y"),
                                                  period=5)
    #scheduler = MLLF(queue_length=10)
    scheduler = MLLF()
    acnsim = ACNsim()

    #simulation_output = acnsim.simulate_real(scheduler, test_case)
    simulation_output = acnsim.simulate_model(scheduler, period=5, start=datetime.now(), end=(datetime.now() + timedelta(days=30)))

    oa = OutputAnalyzer(simulation_output)
    #oa.plot_station_activity()
    oa.plot_EV_behavioral_stats()
    #oa.plot_EV_behavioral_stats(percentage=False)
    #oa.plot_algorithm_result_stats()
    oa.print_algorithm_result_report()
    #oa.plot_EV_daily_arrivals()
    #oa.print_events(type='info')
    #gd.print_station_sessions(test_case)
    #gd.plot_EV_stats(test_case, 102)