"""
This module contains methods for directly interacting with the _simulator.
"""
import numpy as np
from datetime import timedelta
from collections import namedtuple
import warnings


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
        # TODO: doesn't include charging rates of those evs that unplugged
        #  in the last period if called from a reward function.
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

    def is_feasible(self, load_currents, linear=False, violation_tolerance=None, relative_tolerance=None):
        """ Return if a set of current magnitudes for each load are feasible.

        Wraps Network's is_feasible method.

        For a given constraint, the larger of the violation_tolerance
        and relative_tolerance is used to evaluate feasibility.

        Args:
            load_currents (Dict[str, List[number]]): Dictionary mapping load_ids to schedules of charging rates.
            linear (bool): If True, linearize all constraints to a more conservative but easier to compute constraint by
                ignoring the phase angle and taking the absolute value of all load coefficients. Default False.
            violation_tolerance (float): Absolute amount by which
                schedule may violate network constraints. Default
                None, in which case the network's violation_tolerance
                attribute is used.
            relative_tolerance (float): Relative amount by which
                schedule may violate network constraints. Default
                None, in which case the network's relative_tolerance
                attribute is used.

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
        return self._simulator.network.is_feasible(schedule_matrix, linear, violation_tolerance, relative_tolerance)

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
    # TODO: In PR, ask which of these functions (or if all of them) should
    #  be moved into Interface.
    """ Interface between OpenAI Environments and the ACN Simulation
     Environment.
    """

    @classmethod
    def from_interface(cls, interface):
        gym_interface = GymInterface(interface._simulator)
        return gym_interface

    @property
    def station_ids(self):
        """ Return a list of space ids of stations the in the network.

        Returns:
            List[str]: List of station ids in the network.
        """
        return self._simulator.network.station_ids

    @property
    def active_station_ids(self):
        """ Returns a list of active EVSE station ids for use by the
        algorithm.

        Returns:
            List[str]: List of EVSE station ids with an EV plugged in
                that is not finished charging.
        """
        return self._simulator.network.active_station_ids

    @property
    def is_done(self):
        """ Returns if the simulation is complete (i.e. event queue is
        empty).

        Returns:
            bool: True if simulation is complete.
        """
        return self._simulator.event_queue.empty()

    @property
    def charging_rates(self):
        """ Returns the charging_rates of the simulator at all times.

        Returns:
            np.ndarray: numpy array of all charging rates. Each row
                represents the charging rates of a station; each column
                represents the charging rates at an iteration.
        """
        return self._simulator.charging_rates

    def is_feasible_evse(self, load_currents):
        """
        Return if each EVSE in load_currents can accept the pilots
        assigned to it.

        Args:
            load_currents (Dict[str, List[number]]): Dictionary mapping
                load_ids to schedules of charging rates.

        Returns: bool: True if all pilots are valid for the EVSEs to
            which they are sent.
        """
        # TODO: Should this check apply to all schedules, just the ones in
        #  the current iteration, or some intermediate?
        evse_satisfied = True
        for station_id in load_currents:
            # Check that each EVSE in the schedule is actually in the network.
            if station_id not in self.station_ids:
                raise KeyError(
                    f'Station {station_id} in schedule but not found in '
                    f'network.'
                )
            # Check that none of the EVSE pilot signal limits are violated.
            evse_is_continuous, evse_allowable_pilots = \
                self.allowable_pilot_signals(station_id)
            if evse_is_continuous:
                min_rate = evse_allowable_pilots[0]
                max_rate = evse_allowable_pilots[1]
                evse_satisfied = not np.all(np.array([
                    (min_rate <= pilot <= max_rate) or pilot == 0
                    for pilot in load_currents[station_id]
                ]))
            else:
                evse_satisfied = not np.all(np.isin(
                    np.array(load_currents[station_id]),
                    np.array(evse_allowable_pilots)
                ))
        return evse_satisfied

    def is_feasible(self, load_currents, linear=False,
                    violation_tolerance=None, relative_tolerance=None):
        """ Overrides Interface.is_feasible with extra feasibility
        checks. These include:

        - Checking for stations in a schedule but not in the network.
        - Checking that the schedule doesn't violate any constraints on
        EVSE charging rates.
        """
        # Check if conditions of standard Interface.is_feasible are violated.
        constraints_satisfied = super().is_feasible(
            load_currents,
            linear=linear,
            violation_tolerance=violation_tolerance,
            relative_tolerance=relative_tolerance
        )

        evse_satisfied = self.is_feasible_evse(load_currents)

        return constraints_satisfied and evse_satisfied

    def step(self, new_schedule, force_feasibility=True):
        """ Step the simulation using the input new_schedule until the
        simulator requires a new charging schedule. If the provided
        schedule is infeasible, steps the simulation only if
        `force_feasibility` is `False`, otherwise doesn't step the
        simulation.

        Args:
            new_schedule (Dict[str, List[number]]): Dictionary mapping
            station ids to a schedule of pilot signals.
            force_feasibility (bool): If True, do not allow an

        Returns:
            bool: True if the simulation is completed
            bool: True if the schedule was feasible

        Warns:
            UserWarning: If the length of the new schedule is less than
                the Simulator's `max_recompute` parameter. This warning
                is raised because stepping the Simulator with a schedule
                of length less than `max_recompute` could cause the
                pilot signals to be updated with 0's after the schedule
                runs out of entries.
        """
        # Check that length of new schedules is not less than max_recompute
        if (len(new_schedule) == 0
                or len(new_schedule.values()[0])
                < self._simulator.max_recompute):
            warnings.warn(
                f"Length of schedules {len(new_schedule.values()[0])} is less "
                f"than this simulation's max_recompute parameter "
                f"{self._simulator.max_recompute}. Pilots may be updated with"
                f"zeros."
            )

        schedule_is_feasible = self.is_feasible(new_schedule)
        if force_feasibility and not schedule_is_feasible:
            return self._simulator.event_queue.empty(), schedule_is_feasible
        return self._simulator.step(new_schedule), schedule_is_feasible

    def last_energy_delivered(self):
        """ Return the actual energy delivered in the last period, in
        amp-periods.

        Returns:
            number: Total energy delivered in the last period, in
                amp-periods.
        """
        return sum([ev.current_charging_rate for ev in self.active_evs])

    def constraint_currents(self, input_schedule):
        # TODO: standardize schedule input format for interface functions
        return abs(self._simulator.network.constraint_current(
            input_schedule, time_indices=[0]))
    # TODO: Add a function that gives the EVs unplugged in the last time step


class InvalidScheduleError(Exception):
    """ Raised when the schedule passed to the simulator is invalid. """
    pass
