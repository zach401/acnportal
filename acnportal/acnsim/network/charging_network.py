# coding=utf-8
""" Class to represent the physical model of an adaptive charging network. """
from typing import Optional, List, Dict, Any, Tuple

from .current import Current
import pandas as pd
import numpy as np
from collections import OrderedDict
import warnings

from ..models import EV, BaseEVSE
from ..base import BaseSimObj


class ChargingNetwork(BaseSimObj):
    """
    The ChargingNetwork class describes the infrastructure of the
    charging network with information about the types of the charging
    station_schedule.

    Args:
        violation_tolerance (float): Absolute amount by which an input
            charging schedule may violate network constraints (A).
        relative_tolerance (float): Relative amount by which an input
            charging schedule may violate network constraints (A).
    """

    _EVSEs: OrderedDict
    constraint_matrix: Optional[np.ndarray]
    magnitudes: np.ndarray
    constraint_index: List[str]
    _voltages: np.ndarray
    _phase_angles: np.ndarray
    violation_tolerance: float
    relative_tolerance: float
    _station_ids_dict: Dict[str, int]
    max_pilot_signals: np.ndarray
    min_pilot_signals: np.ndarray
    allowable_rates: List[np.ndarray]
    is_continuous: np.ndarray

    def __init__(
        self, violation_tolerance: float = 1e-5, relative_tolerance: float = 1e-7
    ):
        self._EVSEs = OrderedDict()
        # Matrix of constraints
        self.constraint_matrix = None
        # Vector of limiting magnitudes
        self.magnitudes = np.array([])
        # List of constraints in order of addition to network
        self.constraint_index = []
        self._voltages = np.array([])
        self._phase_angles = np.array([])
        self.violation_tolerance = violation_tolerance
        self.relative_tolerance = relative_tolerance

        # Cached information-storing objects for use by Interface.
        (
            self._station_ids_dict,
            self.max_pilot_signals,
            self.min_pilot_signals,
            self.allowable_rates,
            self.is_continuous,
        ) = self._update_info_store()

    def _update_info_store(
        self,
    ) -> Tuple[Dict[str, int], np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
        station_ids: List[str] = self.station_ids
        self._station_ids_dict = {
            station_id: i for i, station_id in enumerate(station_ids)
        }
        self.max_pilot_signals = np.array(
            [self._EVSEs[station_id].max_rate for station_id in station_ids]
        )
        self.min_pilot_signals = np.array(
            [self._EVSEs[station_id].min_rate for station_id in station_ids]
        )
        allowable_rates = []
        is_continuous = []
        for station_id in station_ids:
            # Get allowable pilot signals and continuity for this EVSE.
            evse = self._EVSEs[station_id]
            continuous, allowable = evse.is_continuous, evse.allowable_pilot_signals
            allowable_rates.append(np.array(allowable))
            is_continuous.append(continuous)
        is_continuous = np.array(is_continuous, dtype="bool")
        self.allowable_rates = allowable_rates
        self.is_continuous = is_continuous
        return (
            self._station_ids_dict,
            self.max_pilot_signals,
            self.min_pilot_signals,
            self.allowable_rates,
            self.is_continuous,
        )

    @property
    def current_charging_rates(self) -> np.ndarray:
        """Return the current actual charging rate of all EVSEs in the network. If
        no EV is attached to a given EVSE, that EVSE's charging rate is 0. In the
        returned array, the charging rates are given in the same order as the list of
        EVSEs given by station_ids

        Returns:
            np.Array: numpy ndarray of actual charging rates of all EVSEs in the
                network.
        """
        return np.array(
            [
                evse.ev.current_charging_rate if evse.ev is not None else 0
                for evse in self._EVSEs.values()
            ]
        )

    @property
    def station_ids(self) -> List[str]:
        """Return the IDs of all registered EVSEs.

        Returns:
            List[str]: List of all registered EVSE IDs.
        """
        return list(self._EVSEs.keys())

    @property
    def active_evs(self) -> List[EV]:
        """Return all EVs which are connected to an EVSE and which are not already
        fully charged.

        Returns:
            List[EV]: List of EVs which can currently be charged.
        """
        return [
            evse.ev
            for evse in self._EVSEs.values()
            if evse.ev is not None and not evse.ev.fully_charged
        ]

    @property
    def active_station_ids(self) -> List[str]:
        """Return IDs for all stations which have an active EV attached.

        Returns:
            List[str]: List of the station_id of all stations which have an
                active EV attached.
        """
        return [
            evse.station_id
            for evse in self._EVSEs.values()
            if evse.ev is not None and not evse.ev.fully_charged
        ]

    @property
    def voltages(self) -> Dict[str, float]:
        """Return dictionary of voltages for all EVSEs in the network.

        Returns:
            Dict[str, float]: Dictionary mapping EVSE ids their input voltage. [V]
        """
        return {
            self.station_ids[i]: self._voltages[i] for i in range(len(self._voltages))
        }

    @property
    def phase_angles(self) -> Dict[str, float]:
        """Return dictionary of phase angles for all EVSEs in the network.

        Returns:
            Dict[str, float]: Dictionary mapping EVSE ids their input phase angle. [
                degrees]
        """
        return {
            self.station_ids[i]: self._phase_angles[i]
            for i in range(len(self._phase_angles))
        }

    def register_evse(self, evse: BaseEVSE, voltage: float, phase_angle: float) -> None:
        """Register an EVSE with the network so it will be accessible to the rest of
        the simulation. This can only be called before any constraints have been
        registered in order to prevent dimensionality mismatch between the EVSE list
        and the

        Args:
            evse (EVSE): An EVSE object.
            voltage (float): Voltage feeding the EVSE (V).
            phase_angle (float): Phase angle of the voltage/current feeding the EVSE
                (degrees).

        Returns:
            None
        """
        # Only allow registering of EVSEs before any constraints have been added.
        if self.constraint_matrix is not None:
            raise EVSERegistrationError(
                "Attempting to register an EVSE after constraints have been added. "
                "Please register all EVSEs with the network before adding constraints."
            )
        self._EVSEs[evse.station_id] = evse
        self._voltages = np.append(self._voltages, voltage)
        self._phase_angles = np.append(self._phase_angles, phase_angle)
        # Cached information-storing objects for use by Interface.
        _ = self._update_info_store()

    def constraints_as_df(self) -> pd.DataFrame:
        """Returns the network constraints in a pandas DataFrame.

        The index is the constraint IDs, and the columns are station IDs. The
        magnitudes (constraint limits) must be accessed separately.

        Returns:
            pd.DataFrame: The network constraints as a DataFrame.
        """
        return pd.DataFrame(
            self.constraint_matrix,
            columns=self.station_ids,
            index=self.constraint_index,
        )

    def add_constraint(
        self, current: Current, limit: float, name: Optional[str] = None
    ) -> None:
        """Add an additional constraint to the constraint DataFrame.

        Args:
            current (Current): Aggregate current which is constrained.
                See Current for more info.
            limit (float): Upper limit on the aggregate current.
            name (str): Name of this constraint.

        Returns:
            None
        """
        if name is None:
            name = "_const_{0}".format(len(self.constraint_index))
        if name in self.constraint_index:
            warnings.warn(
                f"Constraint {name} already added. Adding input constraint as new "
                f"constraint. Use network.update_constraint to update constraint "
                f"{name}.",
                UserWarning,
            )
            name += "_v2"
        for station_id in current.index:
            if station_id not in self._EVSEs:
                raise KeyError(
                    f"Station {station_id} not found. Register station {station_id} to "
                    f"add constraint {name} to network. "
                )
        current.name = name
        self.magnitudes = np.append(self.magnitudes, limit)
        # Make a DataFrame for the constraint matrix for easy addition of the new
        # constraint
        constraint_frame: pd.DataFrame = self.constraints_as_df()

        if pd.__version__ > "1.4.0":
            if len(constraint_frame) == 0:
                constraint_frame_ = current.to_frame().T
                for col in constraint_frame:
                    if col not in constraint_frame_:
                        constraint_frame_[col] = 0
                constraint_frame = constraint_frame_
            else:
                constraint_frame = pd.concat(
                    [constraint_frame, current.to_frame().T]
                ).fillna(0)
        else:
            constraint_frame = constraint_frame.append(current).fillna(0)

            warnings.warn(
                f"Compatability with pandas <=1.4.0 may not be supported in "
                f"a future version of acnportal.",
                DeprecationWarning,
            )

        # Maintain a list of constraint ids for use with constraint_current.
        self.constraint_index = list(constraint_frame.index)
        # Update the numpy matrix of constraints by reconstructing it from
        # constraint_frame.
        self.constraint_matrix = constraint_frame.reindex(
            columns=self.station_ids
        ).to_numpy()
        # Cached information-storing objects for use by Interface.
        _ = self._update_info_store()

    def remove_constraint(self, name: str) -> None:
        """Remove a network constraint.

        Args:
            name (str): Name of constraint to remove.

        Returns:
            None
        """
        if name not in self.constraint_index:
            raise KeyError(f"Cannot remove constraint {name}: not found in network.")
        del_index: int = self.constraint_index.index(name)
        self.constraint_matrix = np.delete(self.constraint_matrix, del_index, axis=0)
        self.magnitudes = np.delete(self.magnitudes, del_index, axis=0)
        self.constraint_index.remove(name)
        # Cached information-storing objects for use by Interface.
        _ = self._update_info_store()

    def update_constraint(
        self, name: str, current: Current, limit: float, new_name: Optional[str] = None
    ) -> None:
        """Update a network constraint with a new aggregate current, limit, and name.

        Args:
            name (str): Name of constraint to update.
            current (Current): New current to update constraint with
            limit (float): New upper limit to update constraint with
            new_name (str): New name to give constraint

        Returns:
            None
        """
        if new_name is None:
            new_name: str = name
        if name not in self.constraint_index:
            raise KeyError(f"Cannot update constraint {name}: not found in network.")
        self.remove_constraint(name)
        self.add_constraint(current, limit, name=new_name)
        # Cached information-storing objects for use by Interface.
        _ = self._update_info_store()

    def plugin(self, ev: EV, station_id: str = None) -> None:
        """Attach EV to a specific EVSE.

        Args:
            ev (EV): EV object which will be attached to the EVSE.
            [Depreciated]
            station_id (str): ID of the EVSE.

        Returns:
            None

        Raises:
            KeyError: Raised when the station id has not yet been registered.
        """
        if station_id is not None:
            warnings.warn(
                "plugin arg 'station_id' is deprecated. Plugin now uses the value of "
                "ev.station_id directly.",
                DeprecationWarning,
            )
        if ev.station_id in self._EVSEs:
            self._EVSEs[ev.station_id].plugin(ev)
        else:
            raise KeyError("Station {0} not found.".format(ev.station_id))

    def unplug(self, station_id: str, session_id: str = None) -> None:
        """Detach EV from a specific EVSE.

        Args:
            station_id (str): ID of the EVSE.
            session_id (str): ID of the session to be unplugged.
        Returns:
            None

        Raises:
            KeyError: Raised when the station id has not yet been registered.
        """
        if station_id in self._EVSEs:
            if session_id is None:
                warnings.warn(
                    f"Calling ChargingNetwork.unplug without a session_id argument is "
                    f"deprecated. Please include a session_id argument as this will be "
                    f"required in a future release. Unplugging EV at station "
                    f"{station_id}."
                )
                self._EVSEs[station_id].unplug()
            elif self._EVSEs[station_id].ev is None:
                warnings.warn(
                    f"Tried to remove EV with session_id {session_id} which was not "
                    f"present at station {station_id}. Found no EV instead."
                )
            elif session_id == self._EVSEs[station_id].ev.session_id:
                self._EVSEs[station_id].unplug()
            else:
                warnings.warn(
                    f"Tried to remove EV with session_id {session_id} which was not "
                    f"present at station {station_id}. Found EV with session_id "
                    f"{self._EVSEs[station_id].ev.session_id} instead."
                )
        else:
            raise KeyError("Station {0} not found.".format(station_id))

    def get_ev(self, station_id: str) -> EV:
        """Return the EV attached to the specified EVSE.

        Args:
            station_id (str): ID of the EVSE.

        Returns:
            EV: The EV attached to the specified station.

        """
        if station_id in self._EVSEs:
            return self._EVSEs[station_id].ev
        else:
            raise KeyError("Station {0} not found.".format(station_id))

    def update_pilots(self, pilots: np.ndarray, i: int, period: float) -> None:
        """Update the pilot signal sent to each EV. Also triggers the EVs to charge
        at the specified rate.

        Note that if a pilot is not sent to an EVSE the associated EV WILL NOT charge
        during that period. If station_id is pilots or a list does not include the
        current time index, a 0 pilot signal is passed to the EVSE. Station IDs not
        registered in the network are silently ignored.

        Args:
            pilots (np.Array): numpy array with a row for each station_id
                and a column for each time. Each entry in the Array corresponds to
                a charging rate (in A) at the station given by the row at a time given
                by the column.
            i (int): Current time index of the simulation.
            period (float): Length of the charging period. [minutes]

        Returns:
            None
        """
        ids: List[str] = self.station_ids
        for station_number in range(len(ids)):
            new_rate = pilots[station_number, i]
            self._EVSEs[ids[station_number]].set_pilot(
                new_rate, self._voltages[station_number], period
            )

    def constraint_current(
        self,
        input_schedule: np.ndarray,
        constraints: Optional[List[str]] = None,
        time_indices: Optional[List[int]] = None,
        linear: bool = False,
    ):
        """Return the aggregate currents subject to the given constraints. If
        constraints=None, return all aggregate currents.

        Args:
            input_schedule (np.Array): 2-D matrix with each row corresponding to an
                EVSE and each column corresponding to a time index in the schedule.
            constraints (List[str]): List of constraint id's for which to calculate
                aggregate current. If None, calculates aggregate currents for all
                constraints.
            time_indices (List[int]): List of time indices for which to calculate
                aggregate current. If None, calculates aggregate currents for all
                timesteps.
            linear (bool): If True, linearize all constraints to a more conservative
                but easier to compute constraint by ignoring the phase angle and taking
                the absolute value of all load coefficients. Default False.

        Returns:
            np.ndarray: Aggregate currents subject to the given constraints.
        """
        schedule_matrix: np.ndarray = np.array(input_schedule)
        # Convert list of constraint id's to list of indices in constraint matrix
        if constraints is not None:
            constraint_indices: List[int] = [
                i
                for i in range(len(self.constraint_index))
                if self.constraint_index[i] in constraints
            ]
        else:
            constraint_indices: List[int] = list(range(len(self.constraint_index)))

        # If we only want the constraint currents at specific time indices,
        # index schedule_matrix columns using these indices
        if time_indices is not None:
            schedule_matrix = schedule_matrix[:, time_indices]

        if linear:
            return np.abs(
                self.constraint_matrix[constraint_indices] @ schedule_matrix
            ).astype("complex")
        else:
            # build vector of phase angles on EVSE
            angle_coeffs: np.ndarray = np.exp(1j * np.deg2rad(self._phase_angles))

            # multiply schedule by angles matrix element-wise
            phasor_schedule: np.ndarray = (schedule_matrix.T * angle_coeffs).T

            # multiply constraint matrix by current schedule, shifted by the phases
            return self.constraint_matrix[constraint_indices] @ phasor_schedule

    def is_feasible(
        self,
        schedule_matrix: np.ndarray,
        linear: bool = False,
        violation_tolerance: Optional[float] = None,
        relative_tolerance: Optional[float] = None,
    ) -> bool:
        """Return if a set of current magnitudes for each load are feasible.

        For a given constraint, the larger of the violation_tolerance
        and relative_tolerance is used to evaluate feasibility.

        Args:
            schedule_matrix (np.Array): 2-D matrix with each row corresponding to
                an EVSE and each column corresponding to a time index in the schedule.

            linear (bool): If True, linearize all constraints to a more conservative
                but easier to compute constraint by ignoring the phase angle and taking
                the absolute value of all load coefficients. Default False.
            violation_tolerance (float): Absolute amount by which
                schedule_matrix may violate network constraints. Default
                None, in which case the network's violation_tolerance
                attribute is used.
            relative_tolerance (float): Relative amount by which
                schedule_matrix may violate network constraints. Default
                None, in which case the network's relative_tolerance
                attribute is used.

        Returns:
            bool: If load_currents is feasible at time t according to this set of
                constraints.
        """
        # If no violation_tolerance is specified, default to the network's
        # violation_tolerance.
        if violation_tolerance is None:
            violation_tolerance = self.violation_tolerance
        if relative_tolerance is None:
            relative_tolerance = self.relative_tolerance
        rel_magnitude_tol = self.magnitudes * relative_tolerance

        # If there are no constraints (magnitudes vector is empty) return True
        if not len(self.magnitudes):
            return True

        # Calculate aggregate currents for each constraint
        aggregate_currents = self.constraint_current(schedule_matrix, linear=linear)

        # Ensure each aggregate current is less than its limit, returning False if not
        schedule_length = schedule_matrix.shape[1]
        return np.all(
            np.tile(
                self.magnitudes + np.maximum(violation_tolerance, rel_magnitude_tol),
                (schedule_length, 1),
            ).T
            >= np.abs(aggregate_currents)
        )

    def post_charging_update(self):
        """Hook to define actions to take after the charging update."""
        pass

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Implements BaseSimObj._to_dict."""

        attribute_dict = {}
        # Serialize non-nested attributes.
        nn_attr_lst = [
            "violation_tolerance",
            "relative_tolerance",
            "constraint_matrix",
            "magnitudes",
            "_voltages",
            "_phase_angles",
            "constraint_index",
            "_station_ids_dict",
            "max_pilot_signals",
            "min_pilot_signals",
            "is_continuous",
        ]
        for attr in nn_attr_lst:
            attribute_dict[attr] = getattr(self, attr)

        evses = {}
        for station_id, evse in self._EVSEs.items():
            # noinspection PyProtectedMember
            registry, context_dict = evse._to_registry(context_dict=context_dict)
            evses[station_id] = registry["id"]
        attribute_dict["_EVSEs"] = evses

        allowable_rates_list: List = []
        for allowable_rates_array in self.allowable_rates:
            allowable_rates_list.append(allowable_rates_array.tolist())
        attribute_dict["allowable_rates"] = allowable_rates_list

        return attribute_dict, context_dict

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, BaseSimObj]] = None,
    ) -> Tuple[BaseSimObj, Dict[str, BaseSimObj]]:
        """Implements BaseSimObj._from_dict."""
        out_obj = cls(
            violation_tolerance=attribute_dict["violation_tolerance"],
            relative_tolerance=attribute_dict["relative_tolerance"],
        )

        evses = {}
        for station_id, evse in attribute_dict["_EVSEs"].items():
            # noinspection PyProtectedMember
            evse_elt, loaded_dict = BaseSimObj._build_from_id(
                evse, context_dict, loaded_dict=loaded_dict
            )
            evses[station_id] = evse_elt
        out_obj._EVSEs = evses

        if attribute_dict["constraint_matrix"] is not None:
            out_obj.constraint_matrix = np.array(attribute_dict["constraint_matrix"])
        else:
            out_obj.constraint_matrix = attribute_dict["constraint_matrix"]
        out_obj.magnitudes = np.array(attribute_dict["magnitudes"])
        out_obj._voltages = np.array(attribute_dict["_voltages"])
        out_obj._phase_angles = np.array(attribute_dict["_phase_angles"])
        out_obj.constraint_index = attribute_dict["constraint_index"]

        # If the original ChargingNetwork had info stores encoded, overwrite defaults
        # with these.
        try:
            (
                _station_ids_dict,
                max_rates_list,
                min_rates_list,
                allowable_rates_list,
                is_continuous_list,
            ) = (
                attribute_dict["_station_ids_dict"],
                attribute_dict["max_pilot_signals"],
                attribute_dict["min_pilot_signals"],
                attribute_dict["allowable_rates"],
                attribute_dict["is_continuous"],
            )
        except KeyError:
            _ = out_obj._update_info_store()
        else:
            out_obj._station_ids_dict = _station_ids_dict
            out_obj.max_pilot_signals = np.array(max_rates_list)
            out_obj.min_pilot_signals = np.array(min_rates_list)
            out_obj.is_continuous = np.array(is_continuous_list, dtype="bool")
            allowable_rates_arrays: List[np.ndarray] = []
            for allowable_rates in allowable_rates_list:
                allowable_rates_arrays.append(np.array(allowable_rates))
            out_obj.allowable_rates = allowable_rates_arrays

        return out_obj, loaded_dict


class StationOccupiedError(Exception):
    """Exception which is raised when trying to add an EV to an EVSE which is
    already occupied."""


class EVSERegistrationError(Exception):
    """Exception which is raised when trying to add an EVSE to the network after
    constraints have already been added."""
