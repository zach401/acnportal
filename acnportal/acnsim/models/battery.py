import numpy as np

IDEAL = 'Ideal'
NOISY = 'Noisy'
TWO_STAGE = 'TwoStage'


class Battery:
    """This class models the behavior of a battery and battery management system (BMS).

    Args:
        capacity (float): Capacity of the battery [kWh]
        init_charge (float): Initial charge of the battery [kWh]
        max_power (float): Maximum charging rate of the battery [kW]
    """

    def __init__(self, capacity, init_charge, max_power):
        if init_charge > capacity:
            raise ValueError('Initial Charge cannot be greater than capacity.')
        self._capacity = capacity
        self._current_charge = init_charge
        self._init_charge = init_charge
        self._max_power = max_power
        self._current_charging_power = 0

    @property
    def _soc(self):
        """ Returns the state of charge of the battery as a percent."""
        return self._current_charge / self._capacity

    @property
    def max_charging_power(self):
        """ Returns the maximum charging power of the Battery."""
        return self._max_power

    @property
    def current_charging_power(self):
        """ Returns the current draw of the battery on the AC side."""
        return self._current_charging_power

    def charge(self, pilot, voltage, period):
        """ Method to "charge" the battery

        Args:
            pilot (float): Pilot signal passed to the battery. [A]
            voltage (float): AC voltage provided to the battery charger. [V]
            period (float): Length of the charging period. [minutes]

        Returns:
            float: actual charging rate of the battery. [A]

        Raises:
            ValueError: if voltage or period are <= 0.
        """
        if voltage <= 0:
            raise ValueError('Voltage must be greater than 0. Got {0}'.format(voltage))
        if period <= 0:
            raise ValueError('period must be greater than 0. Got {0}'.format(voltage))

        # Rate which would fill the battery in period minutes.
        rate_to_full = (self._capacity - self._current_charge) / (period / 60)

        charge_power = min([pilot*voltage / 1000, self._max_power, rate_to_full])
        self._current_charge += charge_power * (period / 60)
        self._current_charging_power = charge_power
        return charge_power * 1000 / voltage

    def reset(self, init_charge=None):
        """ Reset battery to initial state. If init_charge is not
        given (is None), the battery is reset to its initial charge
        on initialization.

        Args:
            init_charge (float): charge battery should be reset to. [acnsim units]

        Returns:
            None
        """
        if init_charge is None:
            self._current_charge = self._init_charge
        else:
            if init_charge > self._capacity:
                raise ValueError('Initial Charge cannot be greater than capacity.')
            self._current_charge = init_charge
        self._current_charging_power = 0


class Linear2StageBattery(Battery):
    """ Extends Battery with a simple piecewise linear model of battery dynamics based on SoC.

    Battery model based on a piecewise linear approximation of battery behavior. The battery will charge at the
    minimum of max_rate and the pilot until it reaches _transition_soc. After this, the maximum charging rate of the
    battery will decrease linearly to 0 at 100% state of charge.

    For more info on model: https://www.sciencedirect.com/science/article/pii/S0378775316317396

    All public attributes are the same as Battery.

    Args:
        noise_level (float): Standard deviation of the noise to add to the charging process. (kW)
        transition_soc (float): State of charging when transitioning from constant current to constraint voltage.
    """

    def __init__(self, capacity, init_charge, max_power, noise_level=0, transition_soc=0.8):
        super().__init__(capacity, init_charge, max_power)
        self._noise_level = noise_level
        self._transition_soc = transition_soc

    def charge(self, pilot, voltage, period):
        """ Method to "charge" the battery based on a two-stage linear battery model.

        Args:
            pilot (float): Pilot signal passed to the battery. [A]
            voltage (float): AC voltage provided to the battery charger. [V]
            period (float): Length of the charging period. [minutes]

        Returns:
            float: actual charging rate of the battery.

        """
        if voltage <= 0:
            raise ValueError('Voltage must be greater than 0. Got {0}'.format(voltage))
        if period <= 0:
            raise ValueError('period must be greater than 0. Got {0}'.format(voltage))

        # Rate which would fill the battery in period minutes.
        rate_to_full = (self._capacity - self._current_charge) / (period / 60)

        if self._soc < self._transition_soc:
            charge_power = min([pilot * voltage / 1000, self._max_power, rate_to_full])
            if self._noise_level > 0:
                charge_power -= abs(np.random.normal(0, self._noise_level))
        else:
            charge_power = min([pilot * voltage / 1000,
                                (1 - self._soc) / (1 - self._transition_soc) * self._max_power,
                                rate_to_full])
            if self._noise_level > 0:
                charge_power += np.random.normal(0, self._noise_level)
                # ensure that noise does not cause the battery to violate any hard limits.
                charge_power = min([charge_power, pilot * voltage / 1000, self._max_power, rate_to_full])
        self._current_charge += charge_power * (period / 60)
        self._current_charging_power = charge_power
        return charge_power * 1000 / voltage


def batt_cap_fn(requested_energy, stay_dur, voltage, period):
    def _get_init_cap(requested_energy, stay_dur, cap, voltage, period, max_rate=32):
        batt = Linear2StageBattery(cap, 0, max_rate, noise_level=0)
        actual_rates = []
        while True:
            actual_rates.append(batt.charge(max_rate, voltage, period))
            if actual_rates[-1] < 0.1:
                break

        for t in range(len(actual_rates) - 1, -1, -1):
            if t < 0:
                raise ValueError('t should never go below 0')
            if sum(actual_rates[t: max(len(actual_rates), t + stay_dur - 1)]) * voltage / 1000 / (
                    60 / period) >= requested_energy:
                return sum(actual_rates[:t]) * voltage / 1000 / (60 / period)
        return -1

    potential_caps = np.array([8, 24, 40, 60, 85, 100])
    for cap in potential_caps:
        if requested_energy > cap:
            continue
        init = _get_init_cap(requested_energy, stay_dur, cap, voltage, period)
        if init >= 0:
            return cap, init
    raise ValueError('No feasible battery size found.')