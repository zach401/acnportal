from builtins import property


class EV:
    """Class to model the behavior of an Electrical Vehicle (ev).

    Attributes:
        arrival (int): Arrival time of the ev. [periods]
        departure (int): Departure time of the ev. [periods]
        requested_energy (float): Energy requested by the ev on arrival. [acnsim units]
        max_rate (float): Maximum charging rate of the ev. [acnsim units]
        session_id (str): Identifier of the session belonging to this ev.
        station_id (str): Identifier of the station used by this ev.
        energy_delivered (float): Amount of energy delivered to the ev so far. [acnsim units]
        current_charging_rate (float): Charging rate at the current time step. [acnsim units]
        remaining_demand (float): Energy which still needs to be delivered to meet requested_energy. [acnsim units]
        fully_charged (bool): If the ev is fully charged.
        percent_remaining (float): remaining_demand as a percent of requested_energy.
    """

    def __init__(self, arrive, depart, requested_energy, station_id, session_id, battery):
        # User Defined Parameters
        self._arrival = arrive
        self._departure = depart
        self._requested_energy = requested_energy
        self._session_id = session_id
        self._station_id = station_id

        # Internal State
        self._battery = battery
        self._energy_delivered = 0

    @property
    def arrival(self):
        return self._arrival

    @arrival.setter
    def arrival(self, value):
        self._arrival = value

    @property
    def departure(self):
        return self._departure

    @departure.setter
    def departure(self, value):
        self._departure = value

    @property
    def requested_energy(self):
        return self._requested_energy

    @property
    def max_rate(self):
        return self._battery.max_rate

    @property
    def session_id(self):
        return self._session_id

    @property
    def station_id(self):
        return self._station_id

    @property
    def energy_delivered(self):
        return self._energy_delivered

    @property
    def current_charging_rate(self):
        return self._battery.current_charging_rate

    @property
    def remaining_demand(self):
        return self.requested_energy - self.energy_delivered

    @property
    def fully_charged(self):
        return not (self.remaining_demand > 1e-3)

    @property
    def percent_remaining(self):
        return self.remaining_demand / self.requested_energy

    def charge(self, pilot):
        """ Method to "charge" the ev.

        Args:
            pilot (float): Pilot signal pass to the ev.

        Returns:
            float: Actual charging rate of the ev.
        """
        charge_rate = self._battery.charge(pilot)
        self._energy_delivered += charge_rate
        return charge_rate

    def reset(self):
        """ Reset battery back to its initialization. Also reset energy delivered.

        Returns:
            None.
        """
        self._energy_delivered = 0
        self._battery.reset()
