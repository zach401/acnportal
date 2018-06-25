"""
This module contains methods for directly interacting with the simulator. 
"""
from Simulator import Simulator

class Interface:

    def __init__(self, simulator):
        self.simulator = simulator
        pass

    def get_active_EVs(self):
        """ Returns a list of active EVs for use by the algorithm.

        :return: list of EVs currently plugged in and not finished charging
        """
        active_EVs = self.simulator.get_active_EVs()
        return active_EVs


    def submit_schedules(self, schedules):
        """ Sends scheduled charging rates the the appropiate next step (simulator or influxDB).

        :param schedules: (dict) Dictionary where key is the id of the EV and value is a list of scheduled charging rates.
        :return: None
        """
        self.simulator.schedule = schedules
        pass