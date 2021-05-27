from typing import Optional, Dict, Any, Tuple

import numpy as np
import warnings
from ..base import BaseSimObj

IDEAL = "Ideal"
NOISY = "Noisy"
TWO_STAGE = "TwoStage"


class Battery(BaseSimObj):
    """This class models the behavior of a battery and battery management system (BMS).

    Args:
        capacity (float): Capacity of the battery [kWh]
        init_charge (float): Initial charge of the battery [kWh]
        max_power (float): Maximum charging rate of the battery [kW]
    """

    def __init__(self, capacity, init_charge, max_power):
        if init_charge > capacity:
            raise ValueError("Initial Charge cannot be greater than capacity.")
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
            raise ValueError("Voltage must be greater than 0. Got {0}".format(voltage))
        if period <= 0:
            raise ValueError("period must be greater than 0. Got {0}".format(voltage))

        # Rate which would fill the battery in period minutes.
        rate_to_full = (self._capacity - self._current_charge) / (period / 60)

        charge_power = min([pilot * voltage / 1000, self._max_power, rate_to_full])
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
                raise ValueError("Initial Charge cannot be greater than capacity.")
            self._current_charge = init_charge
        self._current_charging_power = 0

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ Implements BaseSimObj._to_dict. """
        attribute_dict = {}
        nn_attr_lst = [
            "_max_power",
            "_current_charging_power",
            "_current_charge",
            "_capacity",
            "_init_charge",
        ]
        for attr in nn_attr_lst:
            attribute_dict[attr] = getattr(self, attr)
        return attribute_dict, context_dict

    @classmethod
    def _from_dict_helper(cls, out_obj, attribute_dict):
        out_obj._current_charging_power = attribute_dict["_current_charging_power"]
        out_obj._current_charge = attribute_dict["_current_charge"]

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, BaseSimObj]] = None,
    ) -> Tuple[BaseSimObj, Dict[str, BaseSimObj]]:
        """ Implements BaseSimObj._from_dict. """
        out_obj = cls(
            attribute_dict["_capacity"],
            attribute_dict["_init_charge"],
            attribute_dict["_max_power"],
        )
        cls._from_dict_helper(out_obj, attribute_dict)
        return out_obj, loaded_dict


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
        charge_calculation (str): If 'stepwise', use the charging
            method from a previous version of acnportal, which
            assumes a constant maximal charging rate for the entire
            timestep during which the pilot signal is input. This
            charging method is less accurate than the _charge method,
            and should only be used for reproducing results from
            older versions of acnportal.

            If 'continuous' or not provided, use the _charge method,
            which assumes a continuously varying maximal charging rate.
    """

    charging_methods = ["continuous", "stepwise"]

    def __init__(
        self,
        capacity,
        init_charge,
        max_power,
        noise_level=0,
        transition_soc=0.8,
        charge_calculation="continuous",
    ):
        super().__init__(capacity, init_charge, max_power)
        self._noise_level = noise_level
        if transition_soc < 0:
            raise ValueError(
                f"transition_soc must be non-negative. " f"Got {transition_soc}."
            )
        elif transition_soc >= 1:
            raise ValueError(
                f"transition_soc must be less than 1. " f"Got {transition_soc}."
            )
        self._transition_soc = transition_soc
        if charge_calculation not in self.charging_methods:
            raise ValueError(
                f"Charging method {charge_calculation} specified in "
                f"charge_calculation attribute not recognized by "
                f"Linear2StageBattery class. use one of "
                f"{self.charging_methods}."
            )
        self.charge_calculation = charge_calculation

    def charge(self, pilot, voltage, period):
        """ Method to "charge" the battery based on a two-stage linear
        battery model.

        Uses one of
        {_charge, _charge_stepwise}
        to charge the battery depending on the value of the
        charge_calculation attribute of this object.
        """
        if self.charge_calculation == "stepwise":
            return self._charge_stepwise(pilot, voltage, period)
        elif self.charge_calculation == "continuous":
            return self._charge(pilot, voltage, period)
        else:
            raise ValueError(
                f"Charging method {self.charge_calculation} specified "
                f"in charge_calculation attribute not recognized by "
                f"Linear2StageBattery class. use one of "
                f"{self.charging_methods}."
            )

    def _charge(self, pilot, voltage, period):
        """ Method to "charge" the battery based on a two-stage linear
        battery model.

        All calculations are done in terms fo battery state of charge
        (SoC). Results are converted back to power units at the end.

        Args:
            pilot (float): Pilot signal passed to the battery. [A]
            voltage (float): AC voltage provided to the battery
                charger. [V]
            period (float): Length of the charging period. [minutes]

        Returns:
            float: average charging rate of the battery over this single
                period.

        """
        if voltage <= 0:
            raise ValueError(f"Voltage must be greater than 0. Got {voltage}.")
        if period <= 0:
            raise ValueError(f"Period must be greater than 0. Got {period}.")
        if pilot == 0:
            self._current_charging_power = 0
            return 0
        # All calculations are done in terms of battery SoC, so we
        # convert pilot signal and max power into pilot and max rate of
        # change of SoC.
        pilot_dsoc = pilot * voltage / 1000 / self._capacity / (60 / period)
        max_dsoc = self._max_power / self._capacity / (60 / period)

        if pilot_dsoc > max_dsoc:
            pilot_dsoc = max_dsoc

        # The pilot SoC rate of change has a new transition SoC at
        # which decreasing of max charging rate occurs.
        pilot_transition_soc = self._transition_soc + (
            pilot_dsoc - max_dsoc
        ) / max_dsoc * (self._transition_soc - 1)

        if pilot < 0:
            warnings.warn(
                f"Negative pilot signal input. Battery models"
                f"may not be accurate for pilot {pilot} A."
            )

        # The charging equation depends on whether the current SoC of
        # the battery is above or below the new transition SoC.
        if self._soc < pilot_transition_soc:
            # In the pre-rampdown region, the charging equation changes
            # depending on whether charging the battery over this
            # time period causes the battery to transition between
            # charging regions.
            if 1 <= (pilot_transition_soc - self._soc) / pilot_dsoc:
                curr_soc = pilot_dsoc + self._soc
            else:
                curr_soc = 1 + np.exp(
                    (pilot_dsoc + self._soc - pilot_transition_soc)
                    / (pilot_transition_soc - 1)
                ) * (pilot_transition_soc - 1)
        else:
            curr_soc = 1 + np.exp(pilot_dsoc / (pilot_transition_soc - 1)) * (
                self._soc - 1
            )

        # Add subtractive noise to the final SoC, scaling the noise
        # such that _noise_level is the standard deviation of the noise
        # in the battery charging power.
        if self._noise_level > 0:
            raw_noise = np.random.normal(0, self._noise_level)
            scaled_noise = raw_noise * (period / 60) / self._capacity
            curr_soc -= abs(scaled_noise)

        dsoc = curr_soc - self._soc
        self._current_charge = curr_soc * self._capacity

        # For charging power and charging rate (current), we use the
        # the average over this time period.
        self._current_charging_power = dsoc * self._capacity / (period / 60)
        return self._current_charging_power * 1000 / voltage

    def _charge_stepwise(self, pilot, voltage, period):
        """ Method to "charge" the battery based on a two-stage linear
        battery model.

        This is a legacy charging method from an older version of
        acnportal. This method assumes a constant maximal charging rate
        for the entire timestep during which the pilot signal is input.
        This charging method is less accurate than the _charge method,
        and should only be used for reproducing results from  older
        versions of acnportal.

        Args:
            pilot (float): Pilot signal passed to the battery. [A]
            voltage (float): AC voltage provided to the battery charger. [V]
            period (float): Length of the charging period. [minutes]

        Returns:
            float: actual charging rate of the battery.

        """
        if voltage <= 0:
            raise ValueError("Voltage must be greater than 0. Got {0}".format(voltage))
        if period <= 0:
            raise ValueError("period must be greater than 0. Got {0}".format(voltage))

        # Rate which would fill the battery in period minutes.
        rate_to_full = (self._capacity - self._current_charge) / (period / 60)

        if self._soc < self._transition_soc:
            charge_power = min([pilot * voltage / 1000, self._max_power, rate_to_full])
            if self._noise_level > 0:
                charge_power = max(
                    charge_power - abs(np.random.normal(0, self._noise_level)), 0
                )
        else:
            charge_power = min(
                [
                    pilot * voltage / 1000,
                    (1 - self._soc) / (1 - self._transition_soc) * self._max_power,
                    rate_to_full,
                ]
            )
            if self._noise_level > 0:
                charge_power = min(
                    max(charge_power + np.random.normal(0, self._noise_level), 0),
                    pilot * voltage / 1000,
                    self._max_power,
                    rate_to_full,
                )
                # ensure that noise does not cause the battery to violate any hard limits.
                charge_power = min(
                    [
                        charge_power,
                        pilot * voltage / 1000,
                        self._max_power,
                        rate_to_full,
                    ]
                )
        self._current_charge += charge_power * (period / 60)
        self._current_charging_power = charge_power
        return charge_power * 1000 / voltage

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ Implements BaseSimObj._to_dict. """
        attribute_dict, context_dict = super()._to_dict(context_dict)
        attribute_dict["_noise_level"] = self._noise_level
        attribute_dict["_transition_soc"] = self._transition_soc
        attribute_dict["charge_calculation"] = self.charge_calculation
        return attribute_dict, context_dict

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, BaseSimObj]] = None,
    ) -> Tuple[BaseSimObj, Dict[str, BaseSimObj]]:
        """ Implements BaseSimObj._from_dict. """
        out_obj = cls(
            attribute_dict["_capacity"],
            attribute_dict["_init_charge"],
            attribute_dict["_max_power"],
            noise_level=attribute_dict["_noise_level"],
            transition_soc=attribute_dict["_transition_soc"],
        )
        if "charge_calculation" in attribute_dict:
            out_obj.charge_calculation = attribute_dict["charge_calculation"]
        cls._from_dict_helper(out_obj, attribute_dict)
        return out_obj, loaded_dict


def batt_cap_fn(requested_energy, stay_dur, voltage, period):
    """ This function takes as input a requested energy, stay duration,
    and measurement parameters (voltage & period) and calculates the
    minimum capacity linear 2 stage battery such that it is
    feasible to deliver requested_energy in stay_dur periods.

    The function returns this minimum total capacity along with an
    initial capacity, which is the maximum initial capacity the battery
    with given total capacity may have such that it is feasible (after
    charging at max rate) to deliver requested_energy in stay_dur
    periods. Thus, the returned total and initial capacities maximize
    the amount of time the battery behaves non-ideally during charging.

    Args:
        requested_energy (float): Energy requested by this EV. If this
            fit uses data from ACN-Data, this requested_energy is the
            amount of energy actually delivered in real life.
        stay_dur (float): Number of periods the EV stayed.
        voltage (float): Voltage at which the battery is charged (V).
        period (float): Number of minutes in a period (minutes).

    """

    def _get_init_cap(battery_cap, max_rate=32, transition_soc=0.8):
        """ Given a requested energy, stay duration, battery capacity
        (kWh), charging voltage, period, max charging rate, and the
        state of charge (SoC) at which non-ideal behavior begins, finds
        the maximum initial capacity the battery may have such that
        requested_energy is delivered in stay_dur periods, assuming
        linear 2 stage battery behavior.

        """
        delta_soc = requested_energy / battery_cap
        # Maximum rate of change of SoC in per period.
        max_dsoc = max_rate * voltage / 1000 / battery_cap / (60 / period)

        # First, assume init_soc is after the transition_soc. In this
        # case, the max init_soc has a closed form solution.
        init_soc = 1 + delta_soc / (
            np.exp(max_dsoc * stay_dur / (transition_soc - 1)) - 1
        )
        if init_soc >= transition_soc:
            return init_soc

        # If that didn't work, search over all possible init_soc for the
        # largest init_soc that still allows for delta_soc to be
        # delivered in stay_dur periods.
        def delta_soc_from_init_soc(init_soc_guess):
            if stay_dur <= (transition_soc - init_soc_guess) / max_dsoc:
                return max_dsoc * stay_dur
            return (
                1
                + np.exp(
                    (max_dsoc * stay_dur + init_soc_guess - transition_soc)
                    / (transition_soc - 1)
                )
                * (transition_soc - 1)
                - init_soc_guess
            )

        # Before that, we make sure that starting at init_soc of 0, it's
        # possible to deliver the requested energy with this battery.
        if delta_soc_from_init_soc(0) < delta_soc:
            return -1

        # Since max energy delivered is decreasing in init_soc, we
        # use a binary search for a decreasing function. The default
        # tolerance is set to 1e-9 to satisfy the tests'
        # assertAlmostEqual default tolerance.
        def binsearch(f, lb, ub, target, tol=1e-9):
            mid = (lb + ub) / 2
            val = f(mid)
            if abs(val - target) < tol:
                return mid
            elif val - target > 0:
                return binsearch(f, mid, ub, target, tol=tol)
            else:
                return binsearch(f, lb, mid, target, tol=tol)

        # Below an initial capacity of
        # (transition_soc - max_dsoc * stay_dur),
        # the same amount of charge is delivered, as charging stays
        # entirely within the ideal region. So, we can start the binary
        # search here.
        init_soc = binsearch(
            delta_soc_from_init_soc, transition_soc - max_dsoc * stay_dur, 1, delta_soc
        )
        return init_soc * battery_cap

    potential_caps = np.array([8, 24, 40, 60, 85, 100])
    for cap in potential_caps:
        if requested_energy > cap:
            continue
        init = _get_init_cap(cap)
        if init >= 0:
            return cap, init
    raise ValueError("No feasible battery size found.")
