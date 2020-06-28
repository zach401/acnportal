import unittest
from unittest import TestCase
from unittest.mock import patch

from acnportal.acnsim.models import Battery, Linear2StageBattery, batt_cap_fn


class TestBatteryBase(TestCase):
    def setUp(self):
        self.init_charge = 50
        self.batt = Battery(100, self.init_charge, 32)

    def test_charge_negative_voltage(self):
        with self.assertRaises(ValueError):
            self.batt.charge(32, -240, 5)

    def test_charge_negative_period(self):
        with self.assertRaises(ValueError):
            self.batt.charge(32, 240, -5)

    def test_valid_reset_default(self):
        self.batt.charge(16, 240, 5)
        self.batt.reset()
        self.assertEqual(self.batt.current_charging_power, 0)
        self.assertEqual(self.batt._current_charge, self.init_charge)

    def test_valid_reset(self):
        self.batt.charge(16, 240, 5)
        self.batt.reset(45)
        self.assertEqual(self.batt.current_charging_power, 0)
        self.assertEqual(self.batt._current_charge, 45)

    def test_reset_over_capacity(self):
        with self.assertRaises(ValueError):
            self.batt.reset(110)


class TestBattery(TestBatteryBase):
    def setUp(self):
        self.init_charge = 50
        self.batt = Battery(100, self.init_charge, 7.68)

    def test_valid_charge(self):
        rate = self.batt.charge(16, 240, 5)
        self.assertEqual(rate, 16)
        self.assertEqual(self.batt.current_charging_power, 3.84)
        self.assertAlmostEqual(self.batt._current_charge, 50.32)

    def test_charge_over_max_rate(self):
        rate = self.batt.charge(55, 240, 5)
        self.assertEqual(rate, 32)
        self.assertEqual(self.batt.current_charging_power, 7.68)
        self.assertAlmostEqual(self.batt._current_charge, 50.64)

    def test_charge_over_capacity(self):
        self.batt = Battery(50, 49.8, 7.68)
        rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 10)
        self.assertAlmostEqual(self.batt.current_charging_power, 2.4)
        self.assertAlmostEqual(self.batt._current_charge, 50)


class TestLinear2StageBatteryStepwiseCharge(TestBatteryBase):
    def setUp(self):
        self.init_charge = 0
        self.batt = Linear2StageBattery(
            100, self.init_charge, 7.68, charge_calculation="stepwise"
        )

    def test_invalid_charge_calculation_method(self):
        with self.assertRaises(ValueError):
            self.batt = Linear2StageBattery(
                100, 0, 7.6, charge_calculation="invalid_method"
            )

    def test_negative_transition_soc(self):
        with self.assertRaises(ValueError):
            self.batt = Linear2StageBattery(
                100, 0, 7.68, transition_soc=-0.1, charge_calculation="stepwise"
            )

    def test_one_transition_soc(self):
        with self.assertRaises(ValueError):
            self.batt = Linear2StageBattery(
                100, 0, 7.68, transition_soc=1, charge_calculation="stepwise"
            )

    def test_over_one_transitions_soc(self):
        with self.assertRaises(ValueError):
            self.batt = Linear2StageBattery(
                100, 0, 7.68, transition_soc=1.1, charge_calculation="stepwise"
            )

    def test_zero_pilot_charge(self):
        self.batt = Linear2StageBattery(100, 0, 7.68, charge_calculation="stepwise")
        with patch("numpy.random.normal", return_value=1.2):
            rate = self.batt.charge(0, 240, 5)
        self.assertAlmostEqual(rate, 0)
        self.assertAlmostEqual(self.batt.current_charging_power, 0)
        self.assertAlmostEqual(self.batt._current_charge, 0)

    def test_valid_charge_no_noise_not_tail(self):
        self.batt = Linear2StageBattery(100, 0, 7.68, charge_calculation="stepwise")
        with patch("numpy.random.normal", return_value=1.2):
            rate = self.batt.charge(16, 240, 5)
        self.assertAlmostEqual(rate, 16)
        self.assertAlmostEqual(self.batt.current_charging_power, 3.84)
        self.assertAlmostEqual(self.batt._current_charge, 0.32)

    def test_valid_charge_positive_noise_not_tail(self):
        self.batt = Linear2StageBattery(
            100, 0, 7.68, noise_level=1, charge_calculation="stepwise"
        )
        with patch("numpy.random.normal", return_value=0.288):
            rate = self.batt.charge(16, 240, 5)
        self.assertAlmostEqual(rate, 14.8)
        self.assertAlmostEqual(self.batt.current_charging_power, 3.552)
        self.assertAlmostEqual(self.batt._current_charge, 0.296)

    def test_valid_charge_negative_noise_not_tail(self):
        self.batt = Linear2StageBattery(
            100, 0, 7.68, noise_level=1, charge_calculation="stepwise"
        )
        with patch("numpy.random.normal", return_value=-0.288):
            rate = self.batt.charge(16, 240, 5)
        self.assertAlmostEqual(rate, 14.8)
        self.assertAlmostEqual(self.batt.current_charging_power, 3.552)
        self.assertAlmostEqual(self.batt._current_charge, 0.296)

    def test_valid_charge_no_noise_tail(self):
        self.batt = Linear2StageBattery(100, 85, 7.68, charge_calculation="stepwise")
        rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 24)
        self.assertAlmostEqual(self.batt.current_charging_power, 5.76)
        self.assertAlmostEqual(self.batt._current_charge, 85.48)

    def test_valid_charge_positive_noise_tail(self):
        self.batt = Linear2StageBattery(
            100, 85, 7.68, noise_level=1, charge_calculation="stepwise"
        )
        with patch("numpy.random.normal", return_value=0.288):
            rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 25.2)
        self.assertAlmostEqual(self.batt.current_charging_power, 6.048)
        self.assertAlmostEqual(self.batt._current_charge, 85.504)

    def test_valid_charge_negative_noise_tail(self):
        self.batt = Linear2StageBattery(
            100, 85, 7.68, noise_level=1, charge_calculation="stepwise"
        )
        with patch("numpy.random.normal", return_value=-0.288):
            rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 22.8)
        self.assertAlmostEqual(self.batt.current_charging_power, 5.472)
        self.assertAlmostEqual(self.batt._current_charge, 85.456)

    def test_charge_over_max_rate_not_tail(self):
        self.batt = Linear2StageBattery(100, 0, 7.68, charge_calculation="stepwise")
        rate = self.batt.charge(40, 240, 5)
        self.assertAlmostEqual(rate, 32)
        self.assertAlmostEqual(self.batt.current_charging_power, 7.68)
        self.assertAlmostEqual(self.batt._current_charge, 0.64)

    def test_charge_over_capacity(self):
        self.batt = Linear2StageBattery(100, 99, 7.68, charge_calculation="stepwise")
        rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 1.6)
        self.assertAlmostEqual(self.batt.current_charging_power, 0.384)
        self.assertAlmostEqual(self.batt._current_charge, 99.032)


class TestLinear2StageBattery(TestLinear2StageBatteryStepwiseCharge):
    def setUp(self):
        self.init_charge = 0
        self.batt = Linear2StageBattery(100, self.init_charge, 7.68)

    def test_valid_charge_no_noise_tail(self):
        self.batt = Linear2StageBattery(100, 85, 7.68)
        rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 23.62006344060197)
        self.assertAlmostEqual(self.batt.current_charging_power, 5.668815225744472)
        self.assertAlmostEqual(self.batt._current_charge, 85.472401268812)

    def test_valid_charge_positive_noise_tail(self):
        self.batt = Linear2StageBattery(100, 85, 7.68, noise_level=1)
        with patch("numpy.random.normal", return_value=0.288):
            rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 22.42006344060197)
        self.assertAlmostEqual(self.batt.current_charging_power, 5.380815225744472)
        self.assertAlmostEqual(self.batt._current_charge, 85.448401268812)

    def test_valid_charge_negative_noise_tail(self):
        self.batt = Linear2StageBattery(100, 85, 7.68, noise_level=1)
        with patch("numpy.random.normal", return_value=-0.288):
            rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 22.42006344060197)
        self.assertAlmostEqual(self.batt.current_charging_power, 5.380815225744472)
        self.assertAlmostEqual(self.batt._current_charge, 85.448401268812)

    def test_charge_over_max_rate_not_tail(self):
        self.batt = Linear2StageBattery(100, 0, 7.68)
        rate = self.batt.charge(40, 240, 5)
        self.assertAlmostEqual(rate, 32)
        self.assertAlmostEqual(self.batt.current_charging_power, 7.68)
        self.assertAlmostEqual(self.batt._current_charge, 0.64)

    def test_charge_over_capacity(self):
        self.batt = Linear2StageBattery(100, 99, 7.68)
        rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 1.574670896040131)
        self.assertAlmostEqual(self.batt.current_charging_power, 0.3779210150496315)
        self.assertAlmostEqual(self.batt._current_charge, 99.0314934179208)

    def test_charge_at_threshold(self):
        self.batt = Linear2StageBattery(100, 80, 7.68)
        rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 31.49341792080207)
        self.assertAlmostEqual(self.batt.current_charging_power, 7.558420300992497)
        self.assertAlmostEqual(self.batt._current_charge, 80.629868358416)

    def test_charge_cross_threshold(self):
        self.batt = Linear2StageBattery(100, 79.9, 7.68)
        rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 31.63875847566334)
        self.assertAlmostEqual(self.batt.current_charging_power, 7.5933020341592)
        self.assertAlmostEqual(self.batt._current_charge, 80.5327751695133)


class TestBatteryFit(TestCase):
    def battery_feasible(self, request, duration, voltage, period):
        cap, init = batt_cap_fn(request, duration, voltage, period)
        batt = Linear2StageBattery(cap, init, 32 * voltage / 1000)
        rates = [batt.charge(32, voltage, period) for _ in range(duration)]
        self.assertAlmostEqual((batt._current_charge - init), request)
        self.assertAlmostEqual(request, sum(rates) * voltage / 1000 * (period / 60))

    def test_no_laxity(self):
        voltage = 208
        period = 5
        max_power = 32 * voltage / 1000
        for dur in [12, 24, 32, 64]:
            self.battery_feasible(max_power * dur / (60 / period), dur, voltage, period)

    def test_half_laxity(self):
        voltage = 208
        period = 5
        max_power = 32 * voltage / 1000
        for dur in [12, 24, 32, 64]:
            self.battery_feasible(
                max_power * dur / (60 / period) / 2, dur, voltage, period
            )

    def test_almost_no_laxity(self):
        voltage = 208
        period = 5
        max_power = 32 * voltage / 1000
        for dur in [12, 24, 32, 64]:
            self.battery_feasible(
                max_power * dur / (60 / period) / 1.001, dur, voltage, period
            )


if __name__ == "__main__":
    unittest.main()
