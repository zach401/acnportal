"""
This module contains methods for directly interacting with the _simulator.
"""


class Interface:
    """ Interface between algorithms and the ACN Simulation Environment."""

    def __init__(self, simulator):
        self._simulator = simulator
        pass

    @property
    def active_evs(self):
        """ Returns a list of active EVs for use by the algorithm.

        Returns:
            List[EV]: List of EVs currently plugged in and not finished charging
        """
        return self._simulator.get_active_evs()

    @property
    def last_applied_pilot_signals(self):
        """ Return the pilot signals that were applied in the last _iteration of the simulation for all active EVs.

        Does not include EVs that arrived in the current _iteration.

        Returns:
            Dict[str, number]: A dictionary with the session ID as key and the pilot signal as value.
        """
        i = self._simulator.iteration - 1
        if i > 0:
            return {ev.session_id: self._simulator.pilot_signals[ev.station_id][i] for ev in self.active_evs if
                    ev.arrival <= i}
        else:
            return {}

    @property
    def last_actual_charging_rate(self):
        """ Return the actual charging rates in the last period for all active EVs.

        Returns:
            Dict[str, number]:  A dictionary with the session ID as key and actual charging rate as value.
        """
        return {ev.session_id: ev.current_charging_rate for ev in self.active_evs}

    @property
    def current_time(self):
        """ Get the current time (the current _iteration) of the simulator.

        Returns:
            int: The current _iteration of the simulator.
        """
        return self._simulator.iteration

    @property
    def max_recompute_time(self):
        """ Return the maximum recompute time of the simulator.

        Returns:
            int: Maximum recompute time of the simulator in number of periods.
        """
        return self._simulator.max_recompute

    def get_allowable_pilot_signals(self, station_id):
        """ Returns the allowable pilot signal levels for the specified EVSE.

        Args:
            station_id (str): The ID of the station for which the allowable rates should be returned.

        Returns:
            List[number]: A list of the allowable pilot signal levels. Values are sorted in increasing order.
        """
        return self._simulator.network.EVSEs[station_id].allowable_rates

    def is_feasible(self, load_currents, t=0, linear=False):
        """ Return if a set of current magnitudes for each load are feasible.

        Wraps Network's is_feasible method.

        Args:
            load_currents (Dict[str, List[number]]): Dictionary mapping load_ids to schedules of charging rates.
            t (int): Index into the charging rate schedule where feasibility should be checked.
            linear (bool): If True, linearize all constraints to a more conservative but easier to compute constraint by
                ignoring the phase angle and taking the absolute value of all load coefficients. Default False.

        Returns:
            bool: If load_currents is feasible at time t according to this constraint set.
        """
        return self._simulator.network.is_feasible(load_currents, t, linear)

    # def get_prices(self, start, length):
    #     """
    #     Get a vector of prices beginning at time start and continuing for length periods.
    #
    #     :param int start: Time step of the simulation where price vector should begin.
    #     :param int length: Number of elements in the prices vector. One entry per period.
    #     :return: vector of floats of length length where each entry is a price which is valid for one period.
    #     """
    #     if self._simulator.prices is not None:
    #         return self._simulator.prices.get_prices(start, length)
    #     else:
    #         raise ValueError('No pricing method is specified.')
    #
    # def get_demand_charge(self, schedule_len):
    #     """
    #     Get the demand charge scaled according to the length of the scheduling period.
    #
    #     :param int schedule_len: length of the schedule in number of periods.
    #     :return: float Demand charge scaled for the scheduling period.
    #     """
    #     if self._simulator.prices is not None:
    #         return self._simulator.prices.get_normalized_demand_charge(self._simulator.period, schedule_len)
    #     else:
    #         raise ValueError('No pricing method is specified.')
    #
    # def get_revenue(self):
    #     """
    #     Get the per unit revenue of energy.
    #
    #     :return: float Revenue per unit of energy.
    #     """
    #     if self._simulator.prices is not None:
    #         return self._simulator.prices.revenue
    #     else:
    #         raise ValueError('No pricing method is specified.')
    #
    # def get_prev_peak(self):
    #     """
    #     Get the highest aggregate peak demand so far in the simulation.
    #
    #     :return: peak demand so far in the simulation.
    #     :rtype: float
    #     """
    #     return self._simulator.peak
    #
    # def get_max_recompute_period(self):
    #     return self._simulator.max_recompute

