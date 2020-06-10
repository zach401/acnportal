from typing import Optional

from .current import Current
import pandas as pd
import numpy as np
from collections import OrderedDict
import warnings
from ..base import BaseSimObj


class ChargingNetwork(BaseSimObj):
    """
    The ChargingNetwork class describes the infrastructure of the
    charging network with information about the types of the charging
    station_schedule.

    Args:
        violation_tolerance (float): Absolute amount by which an input
            charging schedule may violate network constrants (A).
        relative_tolerance (float): Relative amount by which an input
            charging schedule may violate network constrants (A).
    """

    def __init__(self, violation_tolerance=1e-5, relative_tolerance=1e-7):
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

    @property
    def current_charging_rates(self):
        """ Return the current actual charging rate of all EVSEs in the network. If no EV is
        attached to a given EVSE, that EVSE's charging rate is 0. In the returned array, the
        charging rates are given in the same order as the list of EVSEs given by station_ids

        Returns:
            np.Array: numpy ndarray of actual charging rates of all EVSEs in the network.
        """
        return np.array(
            [
                evse.ev.current_charging_rate if evse.ev is not None else 0
                for evse in self._EVSEs.values()
            ]
        )

    @property
    def station_ids(self):
        """ Return the IDs of all registered EVSEs.

        Returns:
            List[str]: List of all registered EVSE IDs.
        """
        return list(self._EVSEs.keys())

    @property
    def active_evs(self):
        """ Return all EVs which are connected to an EVSE and which are not already fully charged.

        Returns:
            List[EV]: List of EVs which can currently be charged.
        """
        return [
            evse.ev
            for evse in self._EVSEs.values()
            if evse.ev is not None and not evse.ev.fully_charged
        ]

    @property
    def active_station_ids(self):
        """ Return IDs for all stations which have an active EV attached.

        Returns:
            List[str]: List of the station_id of all stations which have an active EV attached.
        """
        return [
            evse.station_id
            for evse in self._EVSEs.values()
            if evse.ev is not None and not evse.ev.fully_charged
        ]

    @property
    def voltages(self):
        """ Return dictionary of voltages for all EVSEs in the network.

        Returns:
            Dict[str, float]: Dictionary mapping EVSE ids their input voltage. [V]
        """
        return {
            self.station_ids[i]: self._voltages[i] for i in range(len(self._voltages))
        }

    @property
    def phase_angles(self):
        """ Return dictionary of phase angles for all EVSEs in the network.

        Returns:
            Dict[str, float]: Dictionary mapping EVSE ids their input phase angle. [degrees]
        """
        return {
            self.station_ids[i]: self._phase_angles[i]
            for i in range(len(self._phase_angles))
        }

    def register_evse(self, evse, voltage, phase_angle):
        """ Register an EVSE with the network so it will be accessible to the rest of the simulation.

        Args:
            evse (EVSE): An EVSE object.
            voltage (float): Voltage feeding the EVSE (V).
            phase_angle (float): Phase angle of the voltage/current feeding the EVSE (degrees).

        Returns:
            None
        """
        self._EVSEs[evse.station_id] = evse
        self._voltages = np.append(self._voltages, voltage)
        self._phase_angles = np.append(self._phase_angles, phase_angle)

    def constraints_as_df(self):
        return pd.DataFrame(
            self.constraint_matrix,
            columns=self.station_ids,
            index=self.constraint_index,
        )

    def add_constraint(
        self, current: Current, limit: float, name: Optional[str] = None
    ) -> None:
        """ Add an additional constraint to the constraint DataFrame.

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
                "Constraint {0} already added. Adding input constraint as new constraint. Use network.update_constraint to update constraint {0}".format(
                    name
                ),
                UserWarning,
            )
            name = name + "_v2"
        for station_id in current.index:
            if station_id not in self._EVSEs:
                raise KeyError(
                    "Station {0} not found. Register station {0} to add constraint {1} to network.".format(
                        station_id, name
                    )
                )
        current.name = name
        self.magnitudes = np.append(self.magnitudes, limit)
        # Make a dataframe for the constraint matrix for easy addition of the new constraint
        constraint_frame = self.constraints_as_df()
        constraint_frame = constraint_frame.append(current).fillna(0)
        # Maintain a list of constraint ids for use with constraint_current.
        self.constraint_index = list(constraint_frame.index)
        # Update the numpy matrix of constraints by reconstructing it from constraint_frame.
        self.constraint_matrix = constraint_frame.reindex(
            columns=self.station_ids
        ).to_numpy()

    def remove_constraint(self, name):
        """ Remove a network constraint.

        Args:
            name (str): Name of constriant to remove.

        Returns:
            None
        """
        if name not in self.constraint_index:
            raise KeyError(
                "Cannot remove constraint {0}: not found in network.".format(name)
            )
        del_index = self.constraint_index.index(name)
        self.constraint_matrix = np.delete(self.constraint_matrix, (del_index), axis=0)
        self.magnitudes = np.delete(self.magnitudes, (del_index), axis=0)
        self.constraint_index.remove(name)

    def update_constraint(self, name, current: Current, limit, new_name=None):
        """ Update a network constraint with a new aggregate current, limit, and name.

        Args:
            name (str): Name of constriant to update.
            current (Current): New current to update constraint with
            limit (float): New upper limit to update constraint with
            new_name (str): New name to give constraint

        Returns:
            None
        """
        if new_name is None:
            new_name = name
        if name not in self.constraint_index:
            raise KeyError(
                "Cannot update constraint {0}: not found in network.".format(name)
            )
        self.remove_constraint(name)
        self.add_constraint(current, limit, name=new_name)

    def plugin(self, ev, station_id):
        """ Attach EV to a specific EVSE.

        Args:
            ev (EV): EV object which will be attached to the EVSE.
            station_id (str): ID of the EVSE.

        Returns:
            None

        Raises:
            KeyError: Raised when the station id has not yet been registered.
        """
        if station_id in self._EVSEs:
            self._EVSEs[station_id].plugin(ev)
        else:
            raise KeyError("Station {0} not found.".format(station_id))

    def unplug(self, station_id):
        """ Detach EV from a specific EVSE.

        Args:
            station_id (str): ID of the EVSE.

        Returns:
            None

        Raises:
            KeyError: Raised when the station id has not yet been registered.
        """
        if station_id in self._EVSEs:
            self._EVSEs[station_id].unplug()
        else:
            raise KeyError("Station {0} not found.".format(station_id))

    def get_ev(self, station_id):
        """ Return the EV attached to the specified EVSE.

        Args:
            station_id (str): ID of the EVSE.

        Returns:
            EV: The EV attached to the specified station.

        """
        if station_id in self._EVSEs:
            return self._EVSEs[station_id].ev
        else:
            raise KeyError("Station {0} not found.".format(station_id))

    def update_pilots(self, pilots, i, period):
        """ Update the pilot signal sent to each EV. Also triggers the EVs to charge at the specified rate.

        Note that if a pilot is not sent to an EVSE the associated EV WILL NOT charge during that period.
        If station_id is pilots or a list does not include the current time index, a 0 pilot signal is passed to the
        EVSE.
        Station IDs not registered in the network are silently ignored.

        Args:
            pilots (np.Array): numpy array with a row for each station_id
                and a column for each time. Each entry in the Array corresponds to
                a charging rate (in A) at the staion given by the row at a time given by the column.
            i (int): Current time index of the simulation.
            period (float): Length of the charging period. [minutes]

        Returns:
            None
        """
        ids = self.station_ids
        for station_number in range(len(ids)):
            new_rate = pilots[station_number, i]
            self._EVSEs[ids[station_number]].set_pilot(
                new_rate, self._voltages[station_number], period
            )

    def constraint_current(
        self, input_schedule, constraints=None, time_indices=None, linear=False
    ):
        """ Return the aggregate currents subject to the given constraints. If constraints=None,
        return all aggregate currents.

        Args:
            input_schedule (np.Array): 2-D matrix with each row corresponding to an EVSE and each
                column corresponding to a time index in the schedule.
            constraints (List[str]): List of constraint id's for which to calculate aggregate current. If
                None, calculates aggregate currents for all constraints.
            time_indices (List[int]): List of time indices for which to calculate aggregate current. If None,
                calculates aggregate currents for all timesteps.
            linear (bool): If True, linearize all constraints to a more conservative but easier to compute constraint by
                ignoring the phase angle and taking the absolute value of all load coefficients. Default False.

        Returns:
            np.Array: Aggregate currents subject to the given constraints.
        """
        schedule_matrix = np.array(input_schedule)
        # Convert list of constraint id's to list of indices in constraint matrix
        if constraints is not None:
            constraint_indices = [
                i
                for i in range(len(self.constraint_index))
                if self.constraint_index[i] in constraints
            ]
        else:
            constraint_indices = list(range(len(self.constraint_index)))

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
            angle_coeffs = np.exp(1j * np.deg2rad(self._phase_angles))

            # multiply schedule by angles matrix element-wise
            phasor_schedule = (schedule_matrix.T * angle_coeffs).T

            # multiply constraint matrix by current schedule, shifted by the phases
            return self.constraint_matrix[constraint_indices] @ phasor_schedule

    def is_feasible(
        self,
        schedule_matrix,
        linear=False,
        violation_tolerance=None,
        relative_tolerance=None,
    ):
        """ Return if a set of current magnitudes for each load are feasible.

        For a given constraint, the larger of the violation_tolerance
        and relative_tolerance is used to evaluate feasibility.

        Args:
            schedule_matrix (np.Array): 2-D matrix with each row corresponding to an EVSE and each
                column corresponding to a time index in the schedule.
            linear (bool): If True, linearize all constraints to a more conservative but easier to compute constraint by
                ignoring the phase angle and taking the absolute value of all load coefficients. Default False.
            violation_tolerance (float): Absolute amount by which
                schedule_matrix may violate network constraints. Default
                None, in which case the network's violation_tolerance
                attribute is used.
            relative_tolerance (float): Relative amount by which
                schedule_matrix may violate network constraints. Default
                None, in which case the network's relative_tolerance
                attribute is used.

        Returns:
            bool: If load_currents is feasible at time t according to this set of constraints.
        """
        # If no violation_tolerance is specified, default to the network's violation_tolerance.
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

    def _to_dict(self, context_dict=None):
        """ Implements BaseSimObj._to_dict. """

        attribute_dict = {}
        # Serialize non-nested attributes.
        nn_attr_lst = ["violation_tolerance", "relative_tolerance"]
        for attr in nn_attr_lst:
            attribute_dict[attr] = getattr(self, attr)

        evses = {}
        for station_id, evse in self._EVSEs.items():
            # noinspection PyProtectedMember
            registry, context_dict = evse._to_registry(context_dict=context_dict)
            evses[station_id] = registry["id"]
        attribute_dict["_EVSEs"] = evses

        if self.constraint_matrix is not None:
            attribute_dict["constraint_matrix"] = self.constraint_matrix.tolist()
        else:
            attribute_dict["constraint_matrix"] = self.constraint_matrix
        attribute_dict["magnitudes"] = self.magnitudes.tolist()
        attribute_dict["_voltages"] = self._voltages.tolist()
        attribute_dict["_phase_angles"] = self._phase_angles.tolist()
        attribute_dict["constraint_index"] = self.constraint_index
        return attribute_dict, context_dict

    @classmethod
    def _from_dict(cls, attribute_dict, context_dict, loaded_dict=None):
        """ Implements BaseSimObj._from_dict. """
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

        return out_obj, loaded_dict


class StationOccupiedError(Exception):
    """ Exception which is raised when trying to add an EV to an EVSE which is already occupied."""
