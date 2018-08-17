from builtins import property

import numpy as np
import sys

# Piecewise linear: https://www.sciencedirect.com/science/article/pii/S0378775316317396

class EV:
    '''
    This class models the behavior of an Electrical Vehicle (EV).

    The battery charging is currently described by a piecewise linear model, which charges
    at full speed to 80% and after that with a linearly decreasing rate.

    :ivar int arrival: Arrival time of the EV [period]
    :ivar int departure: Departure time of the EV [period]
    :ivar float requested_energy: The energy the EV has requested upon arrival [kWh]
    :ivar float energy_delivered: The energy that has been delivered to the EV [kWh]
    :ivar float max_rate: Max charging rate for the EV [A]
    :ivar string station_id: The ID for the charging station
    :ivar string session_id: The ID for the charging session
    :ivar int finishing_time: The time the EV finished charging [period]
    '''

    def __init__(self, arrive, depart, requested_energy, max_rate, station, session):
        self.arrival = arrive
        self.departure = depart
        self.requested_energy = requested_energy
        self.energy_delivered = 0
        self.max_rate = max_rate
        self.station_id = station
        self.session_id = session
        self.finishing_time = sys.maxsize

    def charge(self, pilot, tail=False, noise_level=0):
        """ Method to "charge" the EV

        :param pilot: Pilot signal passed to the EV
        :param tail: Default: False. True for 2-phase charging model, false for ideal charging model
        :param noise_level: Default: 0. If noise level > 0, add gaussian noise with sigma^2 = noise_level
        :return: Actual charging rate of the EV
        """
        # Implement 2-phase charging model
        if tail:
            if self.energy_delivered / self.requested_energy <= .8:
                charge_rate = min(pilot, self.max_rate)
                if noise_level > 0:
                    charge_rate -= abs(np.random.normal(0, noise_level))
            else:
                charge_rate = min(pilot, ((1 - (self.energy_delivered / self.requested_energy)) / 0.2)*self.max_rate)
                if noise_level > 0:
                    charge_rate += np.random.normal(0, noise_level)
        # Ideal charging model
        else:
            charge_rate = min(pilot, self.max_rate)
            if noise_level > 0:
                charge_rate -= abs(np.random.normal(0, noise_level))

        # Ensure that noise did not move charge rate out of allowable range
        charge_rate = min([charge_rate, pilot, self.max_rate])

        #Update energy delivered
        self.energy_delivered += charge_rate
        return charge_rate

    def reset(self):
        self.energy_delivered = 0

    @property
    def remaining_demand(self):
        return self.requested_energy - self.energy_delivered

    @property
    def fully_charged(self):
        return not (self.remaining_demand > 1)