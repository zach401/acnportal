import numpy as np


class EV:
    def __init__(self, arrive, depart, requested_energy, max_rate, station, session):
        self.arrival = arrive
        self.departure = depart
        self.requested_energy = requested_energy
        self.energy_delivered = 0
        self.max_rate = max_rate
        self.station_id = station
        self.session_id = session


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

    @property
    def remaining_demand(self):
        return self.requested_energy - self.energy_delivered