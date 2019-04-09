import numpy as np

IDEAL = 'Ideal'
NOISY = 'Noisy'
TWO_STAGE = 'TwoStage'


class Battery:
    """This class models the behavior of a battery and battery management system (BMS).

    Args:
        capacity (float): Capacity of the battery [acnsim units]
        init_charge (float): Initial charge of the battery [acnsim units]
        max_rate (float): Maximum charging rate of the battery [acnsim units]
    """

    def __init__(self, capacity, init_charge, max_rate):
        if init_charge > capacity:
            raise ValueError('Initial Charge cannot be greater than capacity.')
        self._capacity = capacity
        self._current_charge = init_charge
        self._max_rate = max_rate
        self._current_charging_rate = 0

    @property
    def _soc(self):
        """ Returns the state of charge of the battery as a percent."""
        return self._current_charge / self._capacity

    @property
    def max_rate(self):
        """ Returns the maximum charging rate of the battery. """
        return self._max_rate

    @property
    def current_charging_rate(self):
        """ Returns the current charging rate of the battery. """
        return self._current_charging_rate

    def charge(self, pilot):
        """ Method to "charge" the battery

        Args:
            pilot (float): Pilot signal passed to the battery.

        Returns:
            float: actual charging rate of the battery.

        """
        charge_rate = min([pilot, self.max_rate, self._capacity - self._current_charge])
        self._current_charge += charge_rate
        self._current_charging_rate = charge_rate
        return charge_rate

    def reset(self, init_charge):
        """ Reset battery to initial state.

        Args:
            init_charge (float): charge battery should be reset to. [acnsim units]

        Returns:
            None
        """
        if init_charge > self._capacity:
            raise ValueError('Initial Charge cannot be greater than capacity.')
        self._current_charge = init_charge
        self._current_charging_rate = 0


class Linear2StageBattery(Battery):
    """ Extends NoisyBattery with a simple piecewise linear model of battery dynamics based on SoC.

    Battery model based on a piecewise linear approximation of battery behavior. The battery will charge at the
    minimum of max_rate and the pilot until it reaches _transition_soc. After this, the maximum charging rate of the
    battery will decrease linearly to 0 at 100% state of charge.

    For more info on model: https://www.sciencedirect.com/science/article/pii/S0378775316317396

    All public attributes are the same as Battery.
    """

    def __init__(self, capacity, init_charge, max_rate, noise_level=0, transition_soc=0.8):
        super().__init__(capacity, init_charge, max_rate)
        self._noise_level = noise_level
        self._transition_soc = transition_soc

    def charge(self, pilot):
        """ Method to "charge" the battery based on a two-stage linear battery model.

        Args:
            pilot (float): Pilot signal passed to the battery.

        Returns:
            float: actual charging rate of the battery.

        """
        if self._soc < self._transition_soc:
            charge_rate = min([pilot, self.max_rate, self._capacity - self._current_charge])
            if self._noise_level > 0:
                charge_rate -= abs(np.random.normal(0, self._noise_level))
        else:
            charge_rate = min([pilot,
                               ((1 - self._soc) / (1 - self._transition_soc) * self.max_rate),
                               self._capacity - self._current_charge])
            if self._noise_level > 0:
                charge_rate += np.random.normal(0, self._noise_level)
            charge_rate = min([charge_rate, pilot, self.max_rate])
        self._current_charge += charge_rate
        self._current_charging_rate = charge_rate
        return charge_rate


class NoisyBattery(Linear2StageBattery):
    """ Extends Battery to model noise in the charging process.

    Noise here is modeled as rectified Gaussian noise which is subtracted from the normal maximum charging
    rate. This noise roughly models the behavior of the battery management system which might restrict battery
    charging rate during the charging process.

    All public attributes are the same as Battery.
    """

    def __init__(self, capacity, init_charge, max_rate, noise_level=0):
        super().__init__(capacity, init_charge, max_rate, noise_level, transition_soc=1)

    def charge(self, pilot):
        """ Method to "charge" the battery include subtractive rectified Gaussian noise.

        Args:
            pilot (float): Pilot signal passed to the battery.

        Returns:
            float: actual charging rate of the battery.

        """
        charge_rate = min([pilot, self.max_rate, self._capacity - self._current_charge])
        if self._noise_level > 0:
            charge_rate -= abs(np.random.normal(0, self._noise_level))
        self._current_charge += charge_rate
        self._current_charging_rate = charge_rate
        return charge_rate