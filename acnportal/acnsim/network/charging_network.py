from .current import Current
import pandas as pd
import numpy as np

class ChargingNetwork:
    """
    The ChargingNetwork class describes the infrastructure of the charging network with
    information about the types of the charging station_schedule.
    """

    def __init__(self, constraints=None, magnitudes=None):
        self._EVSEs = {}
        self.constraints = constraints if constraints is not None else pd.DataFrame()
        self.magnitudes = magnitudes if magnitudes is not None else pd.Series()
        # Matrix of constraints
        self.constraint_matrix = None
        # Vector of limiting magnitudes
        self.magnitude_vector = None
        # Dict of constraint_id to row index in constraint matrix
        self.constraint_index = {}
        self._voltages = {}
        self._phase_angles = pd.Series()
        self.angles_vector = None
        pass

    @property
    def current_charging_rates(self):
        """ Return the current actual charging rate of all EVSEs in the network. If no EV is
        attached to a given EVSE, that EVSE's charging rate is 0. In the returned array, the
        charging rates are given in lexicographical ordering by EVSE name.

        Returns:
            np.Array: numpy ndarray of actual charging rates of all EVSEs in the network.
        """
        current_rates = np.zeros(len(self._EVSEs))
        i = 0
        for station_id, evse in sorted(self._EVSEs.items()):
            if evse.ev is not None:
                current_rates[i] = evse.ev.current_charging_rate
            i += 1
        return current_rates

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
        return [evse.ev for evse in self._EVSEs.values() if evse.ev is not None and not evse.ev.fully_charged]

    @property
    def active_station_ids(self):
        """ Return IDs for all stations which have an active EV attached.

        Returns:
            List[str]: List of the station_id of all stations which have an active EV attached.
        """
        return [evse.station_id for evse in self._EVSEs.values() if evse.ev is not None and not evse.ev.fully_charged]

    @property
    def voltages(self):
        """ Return dictionary of voltages for all EVSEs in the network.

        Returns:
            Dict[str, float]: Dictionary mapping EVSE ids their input voltage. [V]
        """
        return self._voltages

    @property
    def phase_angles(self):
        """ Return dictionary of phase angles for all EVSEs in the network.

        Returns:
            Dict[str, float]: Dictionary mapping EVSE ids their input phase angle. [degrees]
        """
        return self._phase_angles

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
        self._voltages[evse.station_id] = voltage
        self._phase_angles[evse.station_id] = phase_angle
        # Update the numpy vector of angles by reconstructing it.
        self.angles_vector = np.array([angle for _, angle in sorted(self._phase_angles.items())])

    def add_constraint(self, current, limit, name=None):
        """ Add an additional constraint to the constraint DataFrame.

        Args:
            current (Current): Aggregate current which is constrained. See Current for more info.
            limit (float): Upper limit on the aggregate current.
            name (str): Name of this constraint.

        Returns:
            None
        """
        if name is None:
            name = '_const_{0}'.format(len(self.constraints.index))
        for station_id in current.index:
            if station_id not in self._EVSEs:
                raise KeyError('Station {0} not found. Register station {0} to add constraint {1} to network.'.format(station_id, name))
        current.name = name
        self.magnitudes[name] = limit
        # Update numpy vector of magnitudes (limits on aggregate currents) by reconstructing it.
        self.magnitude_vector = self.magnitudes.sort_index().to_numpy()
        self.constraints = self.constraints.append(current).sort_index(axis=1).fillna(0)
        # Update the numpy matrix of constraints by reconstructing it.
        self.constraint_matrix = self.constraints.sort_index(axis=0).to_numpy()
        # Maintain a dictoinary mapping constraints to row indices in the constraint_matrix, for use with constraint_current method
        self.constraint_index = {self.constraints.index.to_list()[i] : i for i in range(len(self.constraints.index))}

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
            raise KeyError('Station {0} not found.'.format(station_id))

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
            raise KeyError('Station {0} not found.'.format(station_id))

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
            raise KeyError('Station {0} not found.'.format(station_id))

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
        ids = sorted(self.station_ids)
        for station_number in range(len(ids)):
            new_rate = pilots[station_number, i]
            self._EVSEs[ids[station_number]].set_pilot(new_rate, self._voltages[ids[station_number]], period)

    def constraint_current(self, load_currents, constraints=None, time_indices=None, linear=False):
        """ Return the aggregate currents subject to the given constraints. If constraints=None,
        return all aggregate currents.

        Args:
            load_currents (Dict[str, List[number]]): Dictionary mapping load_ids to schedules of charging rates.
            constraints (List[str]): List of constraint id's for which to calculate aggregate current. If
                None, calculates aggregate currents for all constraints.
            time_indices (List[int]): List of time indices for which to calculate aggregate current. If None, 
                calculates aggregate currents for all timesteps.
            linear (bool): If True, linearize all constraints to a more conservative but easier to compute constraint by
                ignoring the phase angle and taking the absolute value of all load coefficients. Default False.

        Returns:
            List[complex]: Aggregate currents subject to the given constraints.
        """
        # Convert list of constraint id's to list of indices in constraint matrix
        if constraints:
            constraint_indices = [self.constraint_index[constraint_id] for constraint_id in constraints]
        else:
            constraint_indices = list(self.constraint_index.values())
        if time_indices:
            schedule_length = len(time_indices)
            schedule_matrix = np.array([[load_currents[evse_id][i] for i in time_indices] if evse_id in load_currents else [0] * schedule_length for evse_id, _ in sorted(self._EVSEs.items())])
        else:
            # Gets the length of the schedules given in load_currents
            schedule_length = len(list(load_currents.values())[0])
            schedule_matrix = np.array([load_currents[evse_id] if evse_id in load_currents else [0] * schedule_length for evse_id, _ in sorted(self._EVSEs.items())])
        if linear:
            return complex(np.abs(self.constraint_matrix[constraint_indices]@schedule_matrix))
        else:
            # build vector of phase angles on EVSE
            angle_coeffs = np.exp(1j*np.deg2rad(self.angles_vector))

            # multiply schedule by angles matrix element-wise
            shifted_schedule = (schedule_matrix.T * angle_coeffs).T

            # multiply constraint matrix by current schedule, shifted by the phases
            return self.constraint_matrix[constraint_indices]@shifted_schedule

    def is_feasible(self, load_currents, t=0, linear=False):
        """ Return if a set of current magnitudes for each load are feasible.

        Args:
            load_currents (Dict[str, List[number]]): Dictionary mapping load_ids to schedules of charging rates.
            t (int): Index into the charging rate schedule where feasibility should be checked.
            linear (bool): If True, linearize all constraints to a more conservative but easier to compute constraint by
                ignoring the phase angle and taking the absolute value of all load coefficients. Default False.

        Returns:
            bool: If load_currents is feasible at time t according to this set of constraints.
        """
        # build schedule matrix, ensuring rows in order of EVSE list
        schedule_lengths = set(len(x) for x in load_currents.values())
        schedule_length = schedule_lengths.pop()
        schedule_matrix = np.array([load_currents[evse_id] if evse_id in load_currents else [0] * schedule_length for evse_id, _ in sorted(self._EVSEs.items())])
        if linear:
            return np.all(np.tile(self.magnitude_vector) - np.abs(self.constraint_matrix@schedule_matrix) >= 0)
        else:
            # build vector of phase angles on EVSE
            angle_coeffs = np.exp(1j*np.deg2rad(self.angles_vector))

            # multiply schedule by angles matrix element-wise
            shifted_schedule = (schedule_matrix.T * angle_coeffs).T

            # multiply constraint matrix by current schedule, shifted by the phases
            curr_mags = self.constraint_matrix@shifted_schedule
            # compare with tiled magnitude matrix
            return np.all(np.tile(self.magnitude_vector, (schedule_length, 1)).T >= np.abs(curr_mags))


class StationOccupiedError(Exception):
    """ Exception which is raised when trying to add an EV to an EVSE which is already occupied."""
    pass
