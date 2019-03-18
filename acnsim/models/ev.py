from builtins import property
import numpy as np


class EV:
    '''
    This class models the behavior of an Electrical Vehicle (EV).

    :ivar int arrival: Arrival time of the EV [periods]
    :ivar int departure: Departure time of the EV [periods]
    :ivar float requested_energy: The energy the EV has requested upon arrival [kWh]
    :ivar float energy_delivered: The energy that has been delivered to the EV [kWh]
    :ivar float max_rate: Max charging rate for the EV [A]
    :ivar string station_id: The ID for the charging station
    :ivar string session_id: The ID for the charging session
    :ivar int finishing_time: The time the EV finished charging [periods]
    '''

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
        """ Method to "charge" the EV

        :param pilot: Pilot signal passed to the EV
        :return: Actual charging rate of the EV
        """
        charge_rate = self._battery.charge(pilot)
        self._energy_delivered += charge_rate
        return charge_rate

    def reset(self):
        self._energy_delivered = 0
        self._battery.reset()
