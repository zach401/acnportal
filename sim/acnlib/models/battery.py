import numpy as np

IDEAL = 'Ideal'
NOISY = 'Noisy'
TWO_STAGE = 'TwoStage'

#
# def battery_factory(type, capacity, init_charge, max_rate):
#     if type == IDEAL:
#         return Battery(capacity, init_charge, max_rate)
#     elif type == NOISY:
#         return NoisyBattery(capacity, init_charge, max_rate, n)
#     elif type == TWO_STAGE:
#         return Linear2StageBattery(capacity, init_charge, max_rate)


class Battery:
    def __init__(self, capacity, init_charge, max_rate):
        if init_charge > capacity:
            raise ValueError('Initial Charge cannot be greater than capacity.')
        self._capacity = capacity
        self._current_charge = init_charge
        self._max_rate = max_rate
        self._current_charging_rate = 0


    @property
    def _soc(self):
        return self._current_charge / self._capacity

    @property
    def max_rate(self):
        return self._max_rate

    @property
    def current_charging_rate(self):
        return self._current_charging_rate

    def charge(self, pilot):
        """ Method to "charge" the battery

        :param pilot: Pilot signal passed to the battery
        :return: Actual charging rate of the battery
        """
        charge_rate = min([pilot, self.max_rate, self._capacity - self._current_charge])
        self._current_charge += charge_rate
        self._current_charging_rate = charge_rate
        return charge_rate

    def reset(self, init_charge):
        if init_charge > self._capacity:
            raise ValueError('Initial Charge cannot be greater than capacity.')
        self._current_charge = init_charge
        self._current_charging_rate = 0


class NoisyBattery(Battery):
    def __init__(self, capacity, init_charge, max_rate, noise_level=0):
        super().__init__(capacity, init_charge, max_rate)
        self._noise_level = noise_level

    def charge(self, pilot):
        """ Method to "charge" the battery

        :param pilot: Pilot signal passed to the battery
        :return: Actual charging rate of the battery
        """
        charge_rate = min([pilot, self.max_rate, self._capacity - self._current_charge])
        if self._noise_level > 0:
            charge_rate -= abs(np.random.normal(0, self._noise_level))
        self._current_charge += charge_rate
        self._current_charging_rate = charge_rate
        return charge_rate


# Piecewise linear: https://www.sciencedirect.com/science/article/pii/S0378775316317396
class Linear2StageBattery(NoisyBattery):
    def charge(self, pilot):
        """ Method to "charge" the battery

        :param pilot: Pilot signal passed to the battery
        :return: Actual charging rate of the battery
        """
        # Implement 2-phase charging model
        if self._soc < 0.8:
            charge_rate = min(pilot, self.max_rate)
            if self._noise_level > 0:
                charge_rate -= abs(np.random.normal(0, self._noise_level))
        else:
            charge_rate = min(pilot, ((1 - self._soc) / 0.2) * self.max_rate)
            if self._noise_level > 0:
                charge_rate += np.random.normal(0, self._noise_level)
            charge_rate = min([charge_rate, pilot, self.max_rate])
        self._current_charge += charge_rate
        self._current_charging_rate = charge_rate
        return charge_rate
