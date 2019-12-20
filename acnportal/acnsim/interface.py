"""
This module contains methods for directly interacting with the _simulator.
"""
import numpy as np
from datetime import timedelta
from collections import namedtuple


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
            return {ev.session_id: self._simulator.pilot_signals[self._simulator.index_of_evse(ev.station_id), i] for ev in self.active_evs if
                ev.arrival <= i}
        else:
            return {}

    @property
    def last_actual_charging_rate(self):
        # TODO: change to "rates"?
        # TODO: doesn't include charging rates of those evs that unplugged in the last period
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
    def period(self):
        """ Return the length of each timestep in the simulation.

        Returns:
            int: Length of each time interval in the simulation. [minutes]
        """
        return self._simulator.period

    @property
    def max_recompute_time(self):
        """ Return the maximum recompute time of the simulator.

        Returns:
            int: Maximum recompute time of the simulator in number of periods. [periods]
        """
        return self._simulator.max_recompute

    def allowable_pilot_signals(self, station_id):
        """ Returns the allowable pilot signal levels for the specified EVSE.
        One may assume an EVSE pilot signal of 0 is allowed regardless
        of this function's return values.

        Args:
            station_id (str): The ID of the station for which the allowable rates should be returned.

        Returns:
            bool: If the range is continuous or not
            list[float]: The sorted set of acceptable pilot signals. If continuous this range will have 2 values
                the min and the max acceptable values. [A]
        """
        evse = self._simulator.network._EVSEs[station_id]
        return evse.is_continuous, evse.allowable_pilot_signals

    def max_pilot_signal(self, station_id):
        """ Returns the maximum allowable pilot signal level for the specified EVSE.

        Args:
            station_id (str): The ID of the station for which the allowable rates should be returned.

        Returns:
            float: the maximum pilot signal supported by this EVSE. [A]
        """
        return self._simulator.network._EVSEs[station_id].max_rate

    def min_pilot_signal(self, station_id):
        """ Returns the minimum allowable pilot signal level for the specified EVSE.

        Args:
            station_id (str): The ID of the station for which the allowable rates should be returned.

        Returns:
            float: the minimum pilot signal supported by this EVSE. [A]
        """
        return self._simulator.network._EVSEs[station_id].min_rate

    def evse_voltage(self, station_id):
        """ Returns the voltage of the EVSE.

        Args:
            station_id (str): The ID of the station for which the allowable rates should be returned.

        Returns:
            float: voltage of the EVSE. [V]
        """
        return self._simulator.network.voltages[station_id]

    def evse_phase(self, station_id):
        """ Returns the phase angle of the EVSE.

        Args:
            station_id (str): The ID of the station for which the allowable rates should be returned.

        Returns:
            float: phase angle of the EVSE. [degrees]
        """
        return self._simulator.network.phase_angles[station_id]

    def remaining_amp_periods(self, ev):
        """ Return the EV's remaining demand in A*periods.

        Returns:
            float: the EV's remaining demand in A*periods.
        """
        return self._convert_to_amp_periods(ev.remaining_demand, ev.station_id)

    def _convert_to_amp_periods(self, kwh, station_id):
        """ Convert the given energy in kWh to A*periods based on the voltage at EVSE station_id.

        Returns:
            float: kwh in A*periods.

        """
        return kwh * 1000 / self.evse_voltage(station_id) * 60 / self.period

    def get_constraints(self):
        """ Get the constraint matrix and corresponding EVSE ids for the network.

        Returns:
            np.ndarray: Matrix representing the constraints of the network. Each row is a constraint and each
        """
        Constraint = namedtuple('Constraint', ['constraint_matrix', 'magnitudes', 'constraint_index', 'evse_index'])
        network = self._simulator.network
        return Constraint(network.constraint_matrix, network.magnitudes, network.constraint_index, network.station_ids)

    def is_feasible(self, load_currents, linear=False):
        """ Return if a set of current magnitudes for each load are feasible.

        Wraps Network's is_feasible method.

        Args:
            load_currents (Dict[str, List[number]]): Dictionary mapping load_ids to schedules of charging rates.
            linear (bool): If True, linearize all constraints to a more conservative but easier to compute constraint by
                ignoring the phase angle and taking the absolute value of all load coefficients. Default False.

        Returns:
            bool: If load_currents is feasible at time t according to this set of constraints.
        """
        if len(load_currents) == 0:
            return True

        # Check that all schedules are the same length
        schedule_lengths = set(len(x) for x in load_currents.values())
        if len(schedule_lengths) > 1:
            raise InvalidScheduleError('All schedules should have the same length.')
        schedule_length = schedule_lengths.pop()

        # Convert input schedule into its matrix representation
        schedule_matrix = np.array(
            [load_currents[evse_id] if evse_id in load_currents else [0] * schedule_length for evse_id in self._simulator.network.station_ids])
        return self._simulator.network.is_feasible(schedule_matrix, linear)

    def get_prices(self, length, start=None):
        """ Get a vector of prices beginning at time start and continuing for length periods. ($/kWh)

        Args:
            length (int): Number of elements in the prices vector. One entry per period.
            start (int): Time step of the simulation where price vector should begin. If None, uses the current timestep
                of the simulation. Default None.

        Returns:
            np.ndarray[float]: Array of floats where each entry is the price for the corresponding period. ($/kWh)
        """
        if 'tariff' in self._simulator.signals:
            if start is None:
                start = self.current_time
            price_start = self._simulator.start + timedelta(minutes=self.period)*start
            return np.array(self._simulator.signals['tariff'].get_tariffs(price_start, length, self.period))
        else:
            raise ValueError('No pricing method is specified.')

    def get_demand_charge(self, start=None):
        """ Get the demand charge for the given period. ($/kW)

        Args:
            start (int): Time step of the simulation where price vector should begin. If None, uses the current timestep
                of the simulation. Default None.

        Returns:
            float: Demand charge for the given period. ($/kW)
        """
        if 'tariff' in self._simulator.signals:
            if start is None:
                start = self.current_time
            price_start = self._simulator.start + timedelta(minutes=self.period) * start
            return self._simulator.signals['tariff'].get_demand_charge(price_start)
        else:
            raise ValueError('No pricing method is specified.')

    def get_prev_peak(self):
        """ Get the highest aggregate peak demand so far in the simulation.

        Returns:
            float: Peak demand so far in the simulation. (A)
        """
        return self._simulator.peak


class GymInterface(Interface):
    """ Interface between OpenAI Environments and the ACN Simulation Environment.

    """

    @classmethod
    def from_interface(cls, interface):
        self = GymInterface(interface._simulator)
        return self

    @property
    def last_predicted_timestamp(self):
        """ Return the timestamp of the last EV departure in the queue.

        Returns:
            int: Last timestamp of an EV departure in the event queue
        """
        return self._simulator.event_queue.get_last_predicted_timestamp()

    @property
    def station_ids(self):
        """ Return a list of space ids of stations the in the network.

        Returns:
            List[str]: List of station ids in the newtork.
        """
        return self._simulator.network.station_ids

    @property
    def num_evses(self):
        """ Return the number of EVSEs in the network.

        Returns:
            int: Number of EVSEs in the network.
        """
        return len(self._simulator.network.station_ids)

    @property
    def evse_list(self):
        """ Return the list of EVSEs in the network.

        Returns:
            List[EVSE]: List of EVSEs in the network.
        """
        return list(self._simulator.network._EVSEs.values())

    @property
    def network_constraints(self):
        """ Return the network constraints.

        Returns:
            np.Array: constraint matrix
            np.Array: magnitudes vector
        """
        return self._simulator.network.constraint_matrix, self._simulator.network.magnitudes

    @property
    def is_done(self):
        """ Returns if the simulation is complete (i.e. event queue is empty).

        Returns:
            bool: True if simulation is complete.
        """
        return self._simulator.event_queue.empty()

    def step(self, new_schedule):
        """ Step the simulation using the input new_schedule until the simulator
        requests a new charging schedule.
        # TODO: somehow remove this function as it is unrealistic
        Args:
            new_schedule (Dict[str, List[number]]): Dictionary mappding station ids to a schedule of pilot signals.

        Returns:
            bool: True if the simulation is completed
        """
        return self._simulator.step(new_schedule, check_feasibility=False)

    def last_energy_delivered(self):
        """ Return the actual energy delivered in the last period, in amp-periods

        Returns:
            number: Total energy delivered in the last period, in amp-periods
        """
        return sum([ev.current_charging_rate for ev in self.active_evs])

    def constraint_currents(self, input_schedule):
        # TODO: standardize schedule input format for interface functions
        return abs(self._simulator.network.constraint_current(input_schedule, time_indices=[0]))
    # TODO: Add a function that gives the EVs unplugged in the last time step

class InvalidScheduleError(Exception):
    """ Raised when the schedule passed to the simulator is invalid. """
    pass

