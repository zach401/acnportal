"""
This module contains methods for directly interacting with the simulator. 
"""

class Interface:

    def __init__(self, simulator):
        self.simulator = simulator
        pass

    def get_active_EVs(self):
        """ Returns a list of active EVs for use by the algorithm.

        :return: (list) List of EVs currently plugged in and not finished charging
        """
        active_EVs = self.simulator.get_active_EVs()
        return active_EVs

    def get_max_charging_rate(self):
        return self.simulator.test_case.DEFAULT_MAX_RATE

    '''
    TODO: 
    - add a function to receive the pilot signals in the last iteration
    - add a function to receive the available charging rates of the system
    '''


    def submit_schedules(self, schedules):
        """ Sends scheduled charging rates the the appropiate next step (simulator or influxDB).

        :param schedules: (dict) Dictionary where key is the id of the EV and value is a list of scheduled charging rates.
        :return: None
        """
        self.simulator.update_schedules(schedules)
        pass