from .constraint_set import ConstraintSet


class ChargingNetwork:
    """
    The ChargingNetwork class describes the infrastructure of the charging network with
    information about the types of the charging station_schedule.
    """

    def __init__(self):
        self._EVSEs = {}
        self.constraint_set = ConstraintSet()
        self._voltages = {}
        self._phase_angles = {}
        pass

    @property
    def current_charging_rates(self):
        """ Return the current actual charging rate of all EVSEs in the network.

        Returns:
            Dict[str, number]: Dictionary mapping station_id to the current actual charging rate of the EV attached to
                that EVSE.
        """
        current_rates = {}
        for station_id, evse in self._EVSEs.items():
            if evse.ev is not None:
                current_rates[station_id] = evse.ev.current_charging_rate
            else:
                current_rates[station_id] = 0
        return current_rates

    @property
    def space_ids(self):
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

    def add_constraint(self, current, limit, name=None):
        """ Add constraint to the network's constraint set.

        Wraps ConstraintSet add_constraint method, see its description for more info.
        """
        self.constraint_set.add_constraint(current, limit, name)

    def is_feasible(self, load_currents, t=0, linear=False):
        """ Return if a set of current magnitudes for each load are feasible at the given time, t.

        Wraps ConstraintSet is_feasible method, see its description for more info.
        """
        return self.constraint_set.is_feasible(load_currents, self._phase_angles, t, linear)

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
            if station_id in pilots and i < len(pilots[station_id]):
                new_rate = pilots[station_id][i]
            else:
                new_rate = 0
            self._EVSEs[station_id].set_pilot(new_rate, self._voltages[station_id], period)


class StationOccupiedError(Exception):
    """ Exception which is raised when trying to add an EV to an EVSE which is already occupied."""
    pass
