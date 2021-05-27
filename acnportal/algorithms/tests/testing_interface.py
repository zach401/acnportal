# coding=utf-8
"""
This module contains methods for accessing a static simulator mocked through a data
file.
"""
from typing import Dict, Any, List, Tuple, Optional
from warnings import warn

import numpy as np
from acnportal.acnsim.interface import (
    Interface,
    SessionInfo,
    InfrastructureInfo,
    Constraint,
)
from acnportal.algorithms import infrastructure_constraints_feasible


class TestingInterface(Interface):
    """ Static interface based on an attributes dict for testing algorithms without a
        full environment.
    """

    data: Dict[str, Any]

    def __init__(self, data: Dict[str, Any]) -> None:
        super().__init__(None)
        self.data = data

    def _get_or_error(self, field: str) -> Any:
        if field in self.data:
            return self.data[field]
        else:
            raise NotImplementedError(f"No data provided for {field}.")

    @property
    def last_applied_pilot_signals(self) -> Dict[str, float]:
        """ Return the pilot signals that were applied in the last _iteration of the
        simulation for all active EVs.

        Does not include EVs that arrived in the current _iteration.

        Returns:
            Dict[str, number]: A dictionary with the session ID as key and the pilot
                signal as value.
        """
        return self._get_or_error("last_applied_pilot_signals")

    @property
    def last_actual_charging_rate(self) -> Dict[str, float]:
        """ Return the actual charging rates in the last period for all active EVs.

        Returns:
            Dict[str, number]:  A dictionary with the session ID as key and actual
                charging rate as value.
        """
        return self._get_or_error("last_actual_charging_rate")

    @property
    def current_time(self) -> int:
        """ Get the current time (the current _iteration) of the simulator.

        Returns:
            int: The current _iteration of the simulator.
        """
        return self._get_or_error("current_time")

    @property
    def period(self) -> float:
        """ Return the length of each timestep in the simulation.

        Returns:
            int: Length of each time interval in the simulation. [minutes]
        """
        return self._get_or_error("period")

    def active_sessions(self) -> List[SessionInfo]:
        """ Return a list of SessionInfo objects describing the currently charging EVs.

        Returns:
            List[SessionInfo]: List of currently active charging sessions.
        """
        active_sessions = self._get_or_error("active_sessions")
        return [
            SessionInfo(current_time=self.current_time, **session)
            for session in active_sessions
        ]

    def infrastructure_info(self) -> InfrastructureInfo:
        """ Returns an InfrastructureInfo object generated from interface.

        Returns:
            InfrastructureInfo: A description of the charging infrastructure.
        """
        infrastructure = self._get_or_error("infrastructure_info")
        return InfrastructureInfo(
            np.array(infrastructure["constraint_matrix"]),
            np.array(infrastructure["constraint_limits"]),
            np.array(infrastructure["phases"]),
            np.array(infrastructure["voltages"]),
            infrastructure["constraint_ids"],
            infrastructure["station_ids"],
            np.array(infrastructure["max_pilot"]),
            np.array(infrastructure["min_pilot"]),
            [np.array(allowable) for allowable in infrastructure["allowable_pilots"]],
            np.array(infrastructure["is_continuous"]),
        )

    def allowable_pilot_signals(self, station_id: str) -> Tuple[bool, List[float]]:
        """ Returns the allowable pilot signal levels for the specified EVSE.
        One may assume an EVSE pilot signal of 0 is allowed regardless
        of this function's return values.

        Args:
            station_id (str): The ID of the station for which the allowable rates
                should be returned.

        Returns:
            bool: If the range is continuous or not
            list[float]: The sorted set of acceptable pilot signals. If continuous this
                range will have 2 values the min and the max acceptable values. [A]
        """
        infrastructure = self._get_or_error("infrastructure_info")
        i = infrastructure["station_ids"].index(station_id)
        return (
            infrastructure["is_continuous"][i],
            infrastructure["allowable_pilots"][i],
        )

    def max_pilot_signal(self, station_id: str) -> float:
        """ Returns the maximum allowable pilot signal level for the specified EVSE.

        Args:
            station_id (str): The ID of the station for which the allowable rates
                should be returned.

        Returns:
            float: the maximum pilot signal supported by this EVSE. [A]
        """
        infrastructure = self._get_or_error("infrastructure_info")
        i = infrastructure["station_ids"].index(station_id)
        return infrastructure["max_pilot"][i]

    def min_pilot_signal(self, station_id: str) -> float:
        """ Returns the minimum allowable pilot signal level for the specified EVSE.

        Args:
            station_id (str): The ID of the station for which the allowable rates
                should be returned.

        Returns:
            float: the minimum pilot signal supported by this EVSE. [A]
        """
        infrastructure = self._get_or_error("infrastructure_info")
        i = infrastructure["station_ids"].index(station_id)
        return infrastructure["min_pilot"][i]

    def evse_voltage(self, station_id: str) -> float:
        """ Returns the voltage of the EVSE.

        Args:
            station_id (str): The ID of the station for which the allowable rates
                should be returned.

        Returns:
            float: voltage of the EVSE. [V]
        """
        infrastructure = self._get_or_error("infrastructure_info")
        i = infrastructure["station_ids"].index(station_id)
        return infrastructure["voltages"][i]

    def evse_phase(self, station_id: str) -> float:
        """ Returns the phase angle of the EVSE.

        Args:
            station_id (str): The ID of the station for which the allowable rates
                should be returned.

        Returns:
            float: phase angle of the EVSE. [degrees]
        """
        infrastructure = self._get_or_error("infrastructure_info")
        i = infrastructure["station_ids"].index(station_id)
        return infrastructure["phases"][i]

    def get_constraints(self) -> Constraint:
        """ Get the constraint matrix and corresponding EVSE ids for the network.

        Returns:
            np.ndarray: Matrix representing the constraints of the network. Each row is
                a constraint and each
        """
        infrastructure = self._get_or_error("infrastructure_info")
        return Constraint(
            np.array(infrastructure["constraint_matrix"]),
            np.array(infrastructure["constraint_limits"]),
            infrastructure["constraint_ids"],
            infrastructure["station_ids"],
        )

    def is_feasible(
        self,
        load_currents: Dict[str, List[float]],
        linear: bool = False,
        violation_tolerance: float = 1e-5,
        relative_tolerance: float = 1e-7,
    ) -> bool:
        """ Return if a set of current magnitudes for each load are feasible.

        Wraps Network's is_feasible method.

        For a given constraint, the larger of the violation_tolerance
        and relative_tolerance is used to evaluate feasibility.

        Args:
            load_currents (Dict[str, List[number]]): Dictionary mapping load_ids to
                schedules of charging rates.
            linear (bool): If True, linearize all constraints to a more conservative
                but easier to compute constraint by ignoring the phase angle and taking
                the absolute value of all load coefficients. Default False.
            violation_tolerance (float): Absolute amount by which
                schedule may violate network constraints. Default
                None, in which case the network's violation_tolerance
                attribute is used.
            relative_tolerance (float): Relative amount by which
                schedule may violate network constraints. Default
                None, in which case the network's relative_tolerance
                attribute is used.

        Returns:
            bool: If load_currents is feasible at time t according to this set of
                constraints.
        """
        if len(load_currents) == 0:
            return True
        # Check that all schedules are the same length
        schedule_lengths = set(len(x) for x in load_currents.values())
        if len(schedule_lengths) > 1:
            raise InvalidScheduleError("All schedules should have the same length.")
        schedule_length = schedule_lengths.pop()

        # Convert input schedule into its matrix representation
        infrastructure = self._get_or_error("infrastructure_info")
        station_ids = infrastructure["station_ids"]
        schedule_matrix = np.array(
            [
                load_currents[station_id]
                if station_id in load_currents
                else [0] * schedule_length
                for station_id in station_ids
            ]
        )
        infrastructure = self.infrastructure_info()
        return infrastructure_constraints_feasible(
            schedule_matrix,
            infrastructure,
            linear,
            violation_tolerance,
            relative_tolerance,
        )

    def get_prices(self, length: int, start: Optional[int] = None) -> np.ndarray:
        """ Get a vector of prices beginning at time start and continuing for length
        periods. ($/kWh)

        Args:
            length (int): Number of elements in the prices vector. One entry per period.
            start (int): Time step of the simulation where price vector should begin.
                If None, uses the current timestep of the simulation. Default None.

        Returns:
            np.ndarray[float]: Array of floats where each entry is the price for the
                corresponding period. ($/kWh)
        """
        tariff = self._get_or_error("tariff")
        if start is None:
            start = self.current_time
        return np.array(tariff["tou_prices"][start : start + length])

    def get_demand_charge(self, start: Optional[int] = None) -> float:
        """ Get the demand charge for the given period. ($/kW)

        Args:
            start (int): Time step of the simulation where price vector should begin.
                If None, uses the current timestep of the simulation. Default None.

        Returns:
            float: Demand charge for the given period. ($/kW)
        """
        if start is not None:
            warn(
                "start time specified in get_demand_charge. "
                "JSONInterface currently only supports static demand charge"
            )
        tariff = self._get_or_error("tariff")
        return tariff["demand_charge"]

    def get_prev_peak(self) -> float:
        """ Get the highest aggregate peak demand so far in the simulation.

        Returns:
            float: Peak demand so far in the simulation. (A)
        """
        return self._get_or_error("prev_peak")


class InvalidScheduleError(Exception):
    """ Raised when the schedule passed to the simulator is invalid. """
