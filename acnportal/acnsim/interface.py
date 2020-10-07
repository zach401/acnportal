# coding=utf-8
"""
This module contains methods for directly interacting with the _simulator.
"""
from copy import deepcopy
from typing import List, Union, Optional, Dict, Set, Tuple

import numpy as np
from datetime import timedelta, datetime
from collections import namedtuple
from warnings import warn

from .models import EV
from .network import ChargingNetwork


class SessionInfo:
    """ Class to store information relevant to a charging session.

        Args:
            station_id (str): Unique identifier of the station (EVSE) where the
                session takes place.
            session_id (str): Unique identifier of the charging session.
            requested_energy (float): Energy requested by the user during the
                session. [kWh]
            energy_delivered (float): Energy delivered already during the
                session. [kWh]
            arrival (int): Time index when the session begins.
            departure (int): Time index when the session ends.
            estimated_departure (int): Time index when the user estimates the session
                will end.
            current_time (int): Time index of the current time.
            min_rates (Union[float, List[float]): Lower bound for the charging
                rate of the session. If List (or np.array) length should be
                departure - arrival and each entry is a lower bound for the
                corresponding time period.
            max_rates (Union[float, List[float]): Upper bound for the charging
                rate of the session. If List (or np.array) length should be
                departure - arrival and each entry is a upper bound for the
                corresponding time period.
    """

    station_id: str
    session_id: str
    requested_energy: float
    energy_delivered: float
    arrival: int
    departure: int
    estimated_departure: int
    current_time: int
    min_rates: np.ndarray
    max_rates: np.ndarray

    def __init__(
        self,
        station_id: str,
        session_id: str,
        requested_energy: float,
        energy_delivered: float,
        arrival: int,
        departure: int,
        estimated_departure: Optional[int] = None,
        current_time: int = 0,
        min_rates: Union[float, List[float]] = 0,
        max_rates: Union[float, List[float]] = float("inf"),
    ):
        self.station_id = station_id
        self.session_id = session_id
        self.requested_energy = requested_energy
        self.energy_delivered = energy_delivered
        self.arrival = arrival
        self.departure = departure
        if self.departure <= self.arrival:
            raise ValueError(
                f"Departure must be later than arrival."
                f"\nArrival:{self.arrival}\n"
                f"Departure:{self.departure}"
            )

        if estimated_departure is None:
            self.estimated_departure = departure
        else:
            self.estimated_departure = estimated_departure

        if self.estimated_departure <= self.arrival:
            raise ValueError(
                "Departure must be later than arrival."
                f"\nArrival:{self.arrival}\n"
                f"Estimated Departure:{self.estimated_departure}"
            )

        self.current_time = current_time

        if np.isscalar(min_rates):
            self.min_rates = np.array([min_rates] * self.remaining_time)
        elif len(min_rates) == self.remaining_time:
            self.min_rates = np.array(min_rates)
        else:
            raise ValueError(
                "min_rates must be a scalar or list-like with length "
                "equal to the remaining_time of the session.\n"
                f"Length of min_rates: {len(min_rates)}\n"
                f"Remaining time: {self.remaining_time}"
            )

        if np.isscalar(max_rates):
            self.max_rates = np.array([max_rates] * self.remaining_time)
        elif len(max_rates) == self.remaining_time:
            self.max_rates = np.array(max_rates)
        else:
            raise ValueError(
                "max_rates must be a scalar or list-like with length "
                "equal to the remaining_time of the session.\n"
                f"Length of max_rates: {len(max_rates)}\n"
                f"Remaining time: {self.remaining_time}"
            )

    @property
    def remaining_demand(self) -> float:
        """ Return the Session's remaining demand in kWh."""
        return self.requested_energy - self.energy_delivered

    @property
    def arrival_offset(self) -> int:
        """ Return the time (in periods) until the arrival of the EV represented by
        this Session, or 0 if the EV has already arrived.
        """
        return max(self.arrival - self.current_time, 0)

    @property
    def remaining_time(self) -> int:
        """ Return the time remaining until the EV represented by this Session departs,
        or the time between the EV's departure and arrival if the EV has not arrived
        yet. If the EV has already departed, return 0.
        """
        remaining = min(
            self.departure - self.arrival, self.departure - self.current_time
        )
        return max(remaining, 0)


class InfrastructureInfo:
    """ Class to store information about the electrical infrastructure.

    Args:
        constraint_matrix (np.array[float]): M x N array relating the
            individual station currents to aggregate currents each of which
            is subject to a constraint. M is the number of constraints and N
            is the number of stations.
        constraint_limits (np.array[float]): Limits on each constrained link.
            Length M.
        phases (np.array[float]): Phase angle of the current at each station
            (EVSE). Length N. [deg]
        voltages (np.array[float]): Voltage of each station. Length N. [V]
        constraint_ids (List[str]): Unique identifier of each constraint.
        station_ids (List[str]): Unique identifier of each station.
        max_pilot (np.array[float]: Maximum pilot signal supported by each
            station.
        min_pilot (np.array[float]: Minimum pilot signal supported by each
            station. A non-zero min_pilot indicates that the station does
            not support any charging rates between 0 and min_pilot.  It is
            implied that all EVSEs support a pilot signal of 0, even if
            min_pilot > 0.
        allowable_pilots (List[np.array[float]): Pilot signals which each
            station supports. The allowable pilot signals for station i are
            stored in allowable_pilots[i]. If a station supports continuous
            pilot signals, the list is of length 2, where the first value is
            the lower bound on the continuous interval and the second is the
            upper bound. In continuous case, it is implied that all EVSEs
            support a pilot signal of 0, even if the continuous interval does
            not include 0.
        is_continuous (np.array[bool]): True if a station supports continuous
            pilot signals, False otherwise.
    """

    constraint_matrix: np.ndarray
    constraint_limits: np.ndarray
    phases: np.ndarray
    voltages: np.ndarray
    constraint_ids: List[str]
    station_ids: List[str]
    max_pilot: np.ndarray
    min_pilot: np.ndarray
    allowable_pilots: Union[List[np.ndarray], List[None]]
    is_continuous: np.ndarray
    _station_ids_dict: Dict[str, int]

    def __init__(
        self,
        constraint_matrix: np.ndarray,
        constraint_limits: np.ndarray,
        phases: np.ndarray,
        voltages: np.ndarray,
        constraint_ids: List[str],
        station_ids: List[str],
        max_pilot: np.ndarray,
        min_pilot: np.ndarray,
        allowable_pilots: Optional[List[np.ndarray]] = None,
        is_continuous: Optional[np.ndarray] = None,
    ):
        self.constraint_matrix = constraint_matrix
        self.constraint_limits = constraint_limits
        self.phases = phases
        self.voltages = voltages
        self.constraint_ids = constraint_ids
        self.station_ids = station_ids
        self._station_ids_dict = {
            station_id: i for i, station_id in enumerate(self.station_ids)
        }
        self.max_pilot = max_pilot
        self.min_pilot = min_pilot
        self.allowable_pilots = (
            allowable_pilots
            if allowable_pilots is not None
            else [None] * self.num_stations
        )
        # Cast an input is_continuous array to dtype bool
        if is_continuous is not None and is_continuous.dtype != "bool":
            is_continuous = is_continuous.astype("bool")
        self.is_continuous = (
            is_continuous
            if is_continuous is not None
            else np.ones(self.num_stations, dtype=bool)
        )
        self._validate()

    @property
    def num_stations(self) -> int:
        """ Return total number of stations in this network."""
        return len(self.station_ids)

    def get_station_index(self, station_id: str) -> int:
        """ Get the numerical index of a given station_id in the network."""
        return self._station_ids_dict[station_id]

    def _validate(self) -> None:
        """ Raise error if attributes do not have consistent shapes."""
        # Check number of stations
        num_stations_set: Set[int] = {
            self.constraint_matrix.shape[1],
            len(self.phases),
            len(self.voltages),
            len(self.station_ids),
            len(self.max_pilot),
            len(self.min_pilot),
            len(self.allowable_pilots),
            len(self.is_continuous),
        }
        num_constraints_set: Set[int] = {
            self.constraint_matrix.shape[0],
            len(self.constraint_limits),
            len(self.constraint_ids),
        }

        errors: List[str] = []
        if len(num_stations_set) > 1:
            errors.append(
                "Number of stations should be consistent between inputs.\n"
                "Stations implied by argument:\n"
                f"constraint_matrix: {self.constraint_matrix.shape[1]},\n"
                f"phases: {len(self.phases)},\n"
                f"voltages: {len(self.voltages)},\n"
                f"max_pilot: {len(self.max_pilot)},\n"
                f"min_pilot: {len(self.min_pilot)},\n"
                f"allowable_pilots: {len(self.allowable_pilots)},\n"
                f"is_continuous: {len(self.is_continuous)},\n"
            )
        if len(num_constraints_set) > 1:
            errors.append(
                "Number of constraints should be consistent between inputs.\n"
                "Constraints implied by argument:\n"
                f"constraint_matrix: {self.constraint_matrix.shape[0]},\n"
                f"constraint_limits: {len(self.constraint_limits)},\n"
                f"constraint_ids: {len(self.constraint_ids)},\n"
            )

        if len(errors) > 0:
            raise ValueError("\n---\n".join(errors))


Constraint = namedtuple(
    "Constraint", ["constraint_matrix", "magnitudes", "constraint_index", "evse_index"],
)


class Interface:
    """ Interface between algorithms and the ACN Simulation Environment."""

    # noinspection PyUnresolvedReferences
    _simulator: "Simulator"

    # noinspection PyUnresolvedReferences
    def __init__(self, simulator: "Simulator"):
        self._simulator = simulator

    # TODO: How do we handle default tolerances in a layer-agnostic way?
    #  Below is a proposed solution. Alternatively, InfrastructureInfo could
    #  be passed with tolerances.
    @property
    def _violation_tolerance(self) -> float:
        """ Returns the intrinsic absolute constraint violation tolerance of
        this network, for internal use by Interface.is_feasible if the algorithm does
        not provide tolerances in its call to is_feasible.

        Returns:
            float: Absolute constraint violation tolerance of the network. Submitted
                schedules may exceed constraint limits by no more than this number.
        """
        return self._simulator.network.violation_tolerance

    @property
    def _relative_tolerance(self) -> float:
        """ Returns the intrinsic relative constraint violation tolerance of
        this network, for internal use by Interface.is_feasible if the algorithm does
        not provide tolerances in its call to is_feasible.

        Returns:
            float: Relative constraint violation tolerance of the network. Submitted
                schedules may exceed constraint limits by no more than this number
                times the constraint limits.
        """
        return self._simulator.network.relative_tolerance

    @property
    def active_evs(self) -> List[EV]:
        """ Returns a list of active EVs for use by the algorithm.

        Returns:
            List[EV]: List of EVs currently plugged in and not finished.
        """
        warn(
            "Property active_evs is depreciated and will be removed in a "
            "future version. Please use active_sessions instead, "
            "as it provides a read-only copy of charging session related "
            "information."
        )
        return self._active_evs

    @property
    def _active_evs(self) -> List[EV]:
        """ Returns a list of active EVs for use by the algorithm.

        Returns:
            List[EV]: List of EVs currently plugged in and not finished.
        """
        return self._simulator.get_active_evs()

    @property
    def last_applied_pilot_signals(self) -> Dict[str, float]:
        """ Return the pilot signals that were applied in the last _iteration
            of the simulation for all active EVs.

        Does not include EVs that arrived in the current _iteration.

        Returns:
            Dict[str, float]: A dictionary with the session ID as key and the
                pilot signal as value.
        """
        i = self._simulator.iteration - 1
        if i > 0:
            return {
                ev.session_id: self._simulator.pilot_signals[
                    self._simulator.index_of_evse(ev.station_id), i
                ]
                for ev in self._active_evs
                if ev.arrival <= i
            }
        else:
            return {}

    @property
    def last_actual_charging_rate(self) -> Dict[str, float]:
        """ Return the actual charging rates in the last period for all
            active sessions.

        Returns:
            Dict[str, number]:  A dictionary with the session ID as key and actual
                charging rate as value.
        """
        return {ev.session_id: ev.current_charging_rate for ev in self._active_evs}

    @property
    def current_time(self) -> int:
        """ Get the current time (the current _iteration) of the simulator.

        Returns:
            int: The current _iteration of the simulator.
        """
        return self._simulator.iteration

    @property
    def current_datetime(self) -> datetime:
        """ Get the simulated wall time of the simulator.

        Returns:
            datetime: The datetime corresponding to the  current time step of
                the simulator.
        """
        return (
            self._simulator.start + timedelta(minutes=self.period) * self.current_time
        )

    @property
    def period(self) -> float:
        """ Return the length of each timestep in the simulation.

        Returns:
            float: Length of each time interval in the simulation. [minutes]
        """
        return self._simulator.period

    @property
    def max_recompute_time(self) -> int:
        """ Return the maximum recompute time of the simulator.

        Returns:
            int: Maximum recompute time of the simulation in number of periods.
                [periods]
        """
        return self._simulator.max_recompute

    def _active_sessions(self) -> List[SessionInfo]:
        """ Return a list of SessionInfo objects describing the currently
            charging EVs.

        Returns:
            List[SessionInfo]: List of currently active charging sessions.
        """
        return [
            SessionInfo(
                ev.station_id,
                ev.session_id,
                ev.requested_energy,
                ev.energy_delivered,
                ev.arrival,
                ev.departure,
                ev.estimated_departure,
                self.current_time,
            )
            for ev in self._active_evs
        ]

    def active_sessions(self) -> List[SessionInfo]:
        """ Return a copy of the list of SessionInfo objects describing the currently
            charging EVs.

        Returns:
            List[SessionInfo]: List of currently active charging sessions.
        """
        return deepcopy(self._active_sessions())

    def _infrastructure_info(self) -> InfrastructureInfo:
        """ Returns an InfrastructureInfo object generated from interface.

        Returns:
            InfrastructureInfo: A description of the charging infrastructure.
        """
        network: ChargingNetwork = self._simulator.network
        station_ids = network.station_ids
        max_pilot_signals = network.max_pilot_signals
        min_pilot_signals = network.min_pilot_signals
        allowable_rates = network.allowable_rates
        is_continuous = network.is_continuous
        return InfrastructureInfo(
            network.constraint_matrix,
            network.magnitudes,
            network._phase_angles,
            network._voltages,
            network.constraint_index,
            station_ids,
            max_pilot_signals,
            min_pilot_signals,
            allowable_rates,
            is_continuous,
        )

    def infrastructure_info(self) -> InfrastructureInfo:
        """ Returns a copy of the InfrastructureInfo object generated from interface.

        Returns:
            InfrastructureInfo: A description of the charging infrastructure.
        """
        return deepcopy(self._infrastructure_info())

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
        infrastructure_info: InfrastructureInfo = self._infrastructure_info()
        continuity: bool = infrastructure_info.is_continuous[
            infrastructure_info.get_station_index(station_id)
        ].tolist()
        # Numpy tolist method returns "object", so type checking will fail here.
        # noinspection PyTypeChecker
        allowable_pilots: List[float] = infrastructure_info.allowable_pilots[
            infrastructure_info.get_station_index(station_id)
        ].tolist()
        return continuity, allowable_pilots

    def max_pilot_signal(self, station_id: str) -> float:
        """ Returns the maximum allowable pilot signal level for the specified EVSE.

        Args:
            station_id (str): The ID of the station.

        Returns:
            float: the maximum pilot signal supported by this EVSE. [A]
        """
        infrastructure_info: InfrastructureInfo = self._infrastructure_info()
        return infrastructure_info.max_pilot[
            infrastructure_info.get_station_index(station_id)
        ]

    def min_pilot_signal(self, station_id: str) -> float:
        """ Returns the minimum allowable pilot signal level for the EVSE.
        A zero pilot signal is always assumed to be allowed; the minimum allowable pilot
        signal returned here is the minimum nonzero pilot signal allowed by the EVSE if
        said EVSE is non-continuous.

        Args:
            station_id (str): The ID of the station.

        Returns:
            float: the minimum pilot signal supported by this EVSE. [A]
        """
        infrastructure_info: InfrastructureInfo = self._infrastructure_info()
        return infrastructure_info.min_pilot[
            infrastructure_info.get_station_index(station_id)
        ]

    def evse_voltage(self, station_id: str) -> float:
        """ Returns the voltage of the EVSE.

        Args:
            station_id (str): The ID of the station.

        Returns:
            float: voltage of the EVSE. [V]
        """
        infrastructure_info: InfrastructureInfo = self._infrastructure_info()
        return infrastructure_info.voltages[
            infrastructure_info.get_station_index(station_id)
        ]

    def evse_phase(self, station_id: str) -> float:
        """ Returns the phase angle of the EVSE.

        Args:
            station_id (str): The ID of the station.

        Returns:
            float: phase angle of the EVSE. [degrees]
        """
        infrastructure_info: InfrastructureInfo = self._infrastructure_info()
        return infrastructure_info.phases[
            infrastructure_info.get_station_index(station_id)
        ]

    def remaining_amp_periods(self, ev: SessionInfo) -> float:
        """ Return the EV's remaining demand in A*periods.

        Args:
            ev (SessionInfo): The SessionInfo object for which to get remaining demand.

        Returns:
            float: the EV's remaining demand in A*periods.
        """
        return self._convert_to_amp_periods(ev.remaining_demand, ev.station_id)

    def _convert_to_amp_periods(self, kwh: float, station_id: str) -> float:
        """ Convert the given energy in kWh to A*periods based on the voltage
            at EVSE station_id.

        Returns:
            float: kwh in A*periods.

        """
        return kwh * 1000 / self.evse_voltage(station_id) * 60 / self.period

    def get_constraints(self) -> Constraint:
        """ Get the constraint matrix and corresponding EVSE ids for the network.

        Returns:
            Constraint: namedtuple including the following attributes:
                constraint_matrix (np.ndarray): Matrix representing the constraints
                    of the network. Each row is a constraint and each
                    column is an index.
                constraint_limits (np.ndarray): Vector of bounding limits for each
                    constraint (1 for each constraint).
                constraint_ids (List[str]): Names of each constraint.
                station_ids (List[str]): Names of each station.

        """
        infrastructure_info: InfrastructureInfo = self._infrastructure_info()
        return Constraint(
            infrastructure_info.constraint_matrix,
            infrastructure_info.constraint_limits,
            infrastructure_info.constraint_ids,
            infrastructure_info.station_ids,
        )

    def is_feasible(
        self,
        load_currents: Dict[str, List[float]],
        linear: bool = False,
        violation_tolerance: Optional[float] = None,
        relative_tolerance: Optional[float] = None,
    ) -> bool:
        # TODO: Should Interface.is_feasible replace network is_feasible?
        # TODO: ACN-Live should not accept violation_tolerance, relative_tolerance
        #  args that are less than the built-in network tols.
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
        if violation_tolerance is None:
            violation_tolerance = self._violation_tolerance
        if relative_tolerance is None:
            relative_tolerance = self._relative_tolerance

        if len(load_currents) == 0:
            return True

        # Check that all schedules are the same length
        schedule_lengths = set(len(x) for x in load_currents.values())
        if len(schedule_lengths) > 1:
            raise InvalidScheduleError("All schedules should have the same length.")
        schedule_length = schedule_lengths.pop()

        # Convert input schedule into its matrix representation
        schedule_matrix = np.array(
            [
                load_currents[evse_id]
                if evse_id in load_currents
                else [0] * schedule_length
                for evse_id in self._simulator.network.station_ids
            ]
        )
        return self._simulator.network.is_feasible(
            schedule_matrix, linear, violation_tolerance, relative_tolerance
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
        if "tariff" in self._simulator.signals:
            if start is None:
                start = self.current_time
            price_start = self._simulator.start + timedelta(minutes=self.period) * start
            return np.array(
                self._simulator.signals["tariff"].get_tariffs(
                    price_start, length, self.period
                )
            )
        else:
            raise ValueError("No pricing method is specified.")

    def get_demand_charge(self, start: Optional[int] = None) -> float:
        """ Get the demand charge for the given period. ($/kW)

        Args:
            start (int): Time step of the simulation where price vector should begin.
                If None, uses the current timestep of the simulation. Default None.

        Returns:
            float: Demand charge for the given period. ($/kW)
        """
        if "tariff" in self._simulator.signals:
            if start is None:
                start = self.current_time
            price_start = self._simulator.start + timedelta(minutes=self.period) * start
            return self._simulator.signals["tariff"].get_demand_charge(price_start)
        else:
            raise ValueError("No pricing method is specified.")

    def get_prev_peak(self) -> float:
        """ Get the highest aggregate peak demand so far in the simulation.

        Returns:
            float: Peak demand so far in the simulation. (A)
        """
        return self._simulator.peak


class InvalidScheduleError(Exception):
    """ Raised when the schedule passed to the simulator is invalid. """
