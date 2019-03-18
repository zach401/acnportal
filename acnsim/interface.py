"""
This module contains methods for directly interacting with the simulator. 
"""


class Interface:

    def __init__(self, simulator):
        self.simulator = simulator
        pass

    def get_active_evs(self):
        """ Returns a list of active EVs for use by the algorithm.

        :return: List of EVs currently plugged in and not finished charging
        :rtype: list(EV)
        """
        active_evs = self.simulator.get_active_evs()
        return active_evs

    def get_max_aggregate_limit(self):
        """
        Returns the maximum charging rate that is allowed in the simulation.

        :return: The maximum charging rate
        :rtype: float
        """
        return self.simulator.network.aggregate_max

    def get_allowable_pilot_signals(self, station_id):
        """
        Get the allowable pilot signal levels for the specified EVSE.

        :param string station_id: The station ID
        :return: A list with the allowable pilot signal levels. The values are sorted in increasing order.
        :rtype: list(int)
        """
        return self.simulator.network.EVSEs[station_id].allowable_rates

    def get_last_applied_pilot_signals(self):
        """
        Get the pilot signals that were applied in the last iteration of the simulation for all active EVs.
        Does not include EVs that arrived in the current iteration.

        :return: A dictionary with the session ID as key and the pilot signal as value.
        :rtype: dict
        """
        active_evs = self.get_active_evs()
        i = self.simulator.iteration - 1
        if i > 0:
            return {ev.session_id: self.simulator.pilot_signals[ev.station_id][i] for ev in active_evs if
                    ev.arrival <= i}
        else:
            return {}

    def get_last_actual_charging_rate(self):
        """
        Get the actual charging rates in the last period for all active EVs.

        :return: A dictionary with the session ID as key and actual charging rate as value.
        :rtype: dict
        """
        active_evs = self.get_active_evs()
        return {ev.session_id: ev.current_charging_rate for ev in active_evs}

    def get_current_time(self):
        """
        Get the current time (the current iteration) of the simulator.

        :return: The current iteration time in the simulator.
        :rtype: int
        """
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

    def get_prices(self, start, length):
        """
        Get a vector of prices beginning at time start and continuing for length periods.

        :param int start: Time step of the simulation where price vector should begin.
        :param int length: Number of elements in the prices vector. One entry per period.
        :return: vector of floats of length length where each entry is a price which is valid for one period.
        """
        if self.simulator.prices is not None:
            return self.simulator.prices.get_prices(start, length)
        else:
            raise ValueError('No pricing method is specified.')

    def get_demand_charge(self, schedule_len):
        """
        Get the demand charge scaled according to the length of the scheduling period.

        :param int schedule_len: length of the schedule in number of periods.
        :return: float Demand charge scaled for the scheduling period.
        """
        if self.simulator.prices is not None:
            return self.simulator.prices.get_normalized_demand_charge(self.simulator.period, schedule_len)
        else:
            raise ValueError('No pricing method is specified.')

    def get_revenue(self):
        """
        Get the per unit revenue of energy.

        :return: float Revenue per unit of energy.
        """
        if self.simulator.prices is not None:
            return self.simulator.prices.revenue
        else:
            raise ValueError('No pricing method is specified.')

    def get_prev_peak(self):
        """
        Get the highest aggregate peak demand so far in the simulation.

        :return: peak demand so far in the simulation.
        :rtype: float
        """
        return self.simulator.peak
