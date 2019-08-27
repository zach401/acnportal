from .current import Current
import pandas as pd
import numpy as np

# TODO: Fix docs here
class ChargingNetwork:
    """
    The ChargingNetwork class describes the infrastructure of the charging network with
    information about the types of the charging station_schedule.
    """

    def __init__(self, constraints=None, magnitudes=None):
        self._EVSEs = {}
        # TODO: is the case where constraints, magnitudes != None still relevant?
        self.constraints = constraints if constraints is not None else pd.DataFrame()
        self.magnitudes = magnitudes if magnitudes is not None else pd.Series()
        # Matrix of constraints
        self.constraint_matrix = None
        # Vector of limiting magnitudes
        self.magnitude_vector = None
        self._voltages = {}
        self._phase_angles = pd.Series()
        self.angles_vector = None
        # TODO: Instead of updating both constraints and constraint_matrix every time anything
        # is added, have a flag that's read or write: writing updates constraints, switching from
        # write to read generates matrix. Switching from read to write deletes matrix.
        pass

    @property
    def current_charging_rates(self):
        """ Return the current actual charging rate of all EVSEs in the network.

        Returns:
            Dict[str, number]: Dictionary mapping station_id to the current actual charging rate of the EV attached to
                that EVSE.
        """
        # TODO: Update to df
        current_rates = {}
        for station_id, evse in self._EVSEs.items():
            if evse.ev is not None:
                current_rates[station_id] = evse.ev.current_charging_rate
            else:
                current_rates[station_id] = 0
        return current_rates

    @property
    def space_ids(self):
        #TODO: Change to station_ids?
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
        self.angles_vector = np.array([angle for _, angle in sorted(self._phase_angles.items())])

    # def add_constraint(self, current, limit, name=None):
    #     """ Add constraint to the network's constraint set.

    #     Wraps ConstraintSet add_constraint method, see its description for more info.
    #     """
    #     self.constraint_set.add_constraint(current, limit, name)

    def add_constraint(self, current, limit, name=None):
        """ Add an additional constraint to the constraint DataFrame.

        Args:
            See Constraint.

        Returns:
            None
        """
        # TODO: Check if each EVSE in current is already registered
        if name is None:
            name = '_const_{0}'.format(len(self.constraints.index))
        current.name = name
        self.magnitudes[name] = limit
        self.magnitude_vector = self.magnitudes.sort_index().to_numpy()
        self.constraints = self.constraints.append(current).sort_index(axis=1).fillna(0)
        self.constraint_matrix = self.constraints.sort_index(axis=0).to_numpy()

    # def is_feasible(self, load_currents, t=0, linear=False):
    #     """ Return if a set of current magnitudes for each load are feasible at the given time, t.

    #     Wraps ConstraintSet is_feasible method, see its description for more info.
    #     """
    #     return self.constraint_set.is_feasible(load_currents, self._phase_angles, t, linear)

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
            pilots (Dict[str, List[number]]): Dictionary mapping station_ids to lists of pilot signals. Each index in
                the array corresponds to an a period of the simulation. [A]
            i (int): Current time index of the simulation.
            period (float): Length of the charging period. [minutes]

        Returns:
            None
        """
        for station_id in self._EVSEs:
            if station_id in pilots.columns and i < len(pilots[station_id]):
                new_rate = pilots[station_id][i]
            else:
                new_rate = 0
            self._EVSEs[station_id].set_pilot(new_rate, self._voltages[station_id], period)

    # def add_constraint(self, current, limit, name=None):
    #     """ Add an additional constraint to the constraint set.

    #     Args:
    #         See Constraint.

    #     Returns:
    #         None
    #     """
    #     # TODO: Directly use Series for current
    #     if name is None:
    #         name = '_const_{0}'.format(len(self.constraints.index))
    #     current.name = name
    #     self.magnitudes[name] = limit
    #     self.constraints = self.constraints.append(current).sort_index(axis=1).fillna(0)

    # TODO: Fix this function
    def constraint_current(self, constraint, load_currents, angles, t=0, linear=False):
        # TODO: refactor as input set of constraint ids, or None, and calculate accordingly
        """ Return the current subject to the given constraint.

        Args:
            constraint (Constraint): Constraint object describing the current.
            load_currents (Dict[str, List[number]]): Dictionary mapping load_ids to schedules of charging rates.
            angles (Dict[str, float]): Dictionary mapping load_ids to the phase angle of the voltage feeding them.
            t (int): Index into the charging rate schedule where feasibility should be checked.
            linear (bool): If True, linearize all constraints to a more conservative but easier to compute constraint by
                ignoring the phase angle and taking the absolute value of all load coefficients. Default False.

        Returns:
            complex: Current subject to the given constraint.
        """
        # TODO: pass constraint ids, return currents that are passed (or all if None passed)
        currents = pd.Series({elt : load_currents[elt][t] for elt in load_currents.keys()}).sort_index()
        if linear:
            return complex(np.abs(constraint[currents.index].to_numpy()).dot(currents.to_numpy()))
        else:
            angles_series = angles[currents.index].sort_index()
            ret = constraint[currents.index].to_numpy().dot(currents.to_numpy() * np.exp(1j*np.deg2rad(angles_series.to_numpy())))
            return ret
        # acc = 0
        # for load_id in constraint.loads:
        #     if load_id in load_currents:
        #         if linear:
        #             acc += abs(constraint.loads[load_id]) * load_currents[load_id][t]
        #         else:
        #             acc += cmath.rect(constraint.loads[load_id] * load_currents[load_id][t],
        #                               math.radians(angles[load_id]))
        # return complex(acc)

    def is_feasible(self, load_currents, t=0, linear=False):
        """ Return if a set of current magnitudes for each load are feasible.

        Args:
            load_currents (Dict[str, List[number]]): Dictionary mapping load_ids to schedules of charging rates.
            angles (Dict[str, float]): Dictionary mapping load_ids to the phase angle of the voltage feeding them.
            t (int): Index into the charging rate schedule where feasibility should be checked.
            linear (bool): If True, linearize all constraints to a more conservative but easier to compute constraint by
                ignoring the phase angle and taking the absolute value of all load coefficients. Default False.

        Returns:
            bool: If load_currents is feasible at time t according to this set of constraints.
        """
        # build schedule matrix, ensuring rows in order of EVSE list
        schedule_length = len(next(iter(load_currents.values())))
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
