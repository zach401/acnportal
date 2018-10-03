"""
This module contains methods for directly interacting with the simulator. 
"""


class Interface:

    def __init__(self, simulator):
        self.simulator = simulator
        pass

    def get_active_EVs(self):
        """ Returns a list of active EVs for use by the algorithm.

        :return: List of EVs currently plugged in and not finished charging
        :rtype: list(EV)
        """
        active_EVs = self.simulator.get_active_EVs()
        return active_EVs

    def get_max_charging_rate(self):
        '''
        Returns the maximum charging rate that is allowed in the simulation.

        :return: The maximum charging rate
        :rtype: float
        '''
        return self.simulator.garage.max_rate

    def get_allowable_pilot_signals(self, station_id):
        '''
        Get the allowable pilot signal levels for the specified EVSE.

        :param string station_id: The station ID
        :return: A list with the allowable pilot signal levels. The values are sorted in increasing order.
        :rtype: list(int)
        '''
        return self.simulator.garage.get_allowable_rates(station_id)

    def get_last_applied_pilot_signals(self):
        '''
        Get the pilot signals that were applied in the last iteration of the simulation.

        :return: A dictionary with the session ID as key and the pilot signal as value.
        :rtype: dict
        '''
        return self.simulator.get_last_applied_pilot_signals()

    def get_last_actual_charging_rate(self):
        '''
        Get the actual charging rates that the EVs are charged with.

        :return: A dictionary with the session ID as key and actual charging rate as value.
        :rtype: dict
        '''
        return self.simulator.get_last_actual_charging_rate()

    def get_current_time(self):
        '''
        Get the current time (the current iteration) of the simulator.

        :return: The current iteration time in the simulator.
        :rtype: int
        '''
        return self.simulator.iteration

    def submit_schedules(self, schedules):
        """
        Sends scheduled charging rates to the simulator.

        This function is called internally. The schedules are the same as returned from
        the ``schedule`` function, so to submit the schedules when writing a charging algorithm just
        make the ``schedule`` function return them.

        :param dict schedules: Dictionary where key is the id of the EV and value is a list of scheduled charging rates.
        :return: None
        """
        self.simulator.update_schedules(schedules)
        pass

    def submit_log(self, text):
        '''
        Submits a text log to the simulator. This can be useful when debugging a custom
        scheduling algorithm

        :param string text: String that should be logged
        :return: None
        '''
        self.simulator.submit_log_event(text)
