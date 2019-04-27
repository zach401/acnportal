import unittest
from unittest import TestCase
from unittest.mock import patch

from acnportal.acnsim.models import Battery, Linear2StageBattery


class TestBatteryBase(TestCase):
    def setUp(self):
        self.batt = Battery(100, 50, 32)

    def test_charge_negative_voltage(self):
        with self.assertRaises(ValueError):
            self.batt.charge(32, -240, 5)

    def test_charge_negative_period(self):
        with self.assertRaises(ValueError):
            self.batt.charge(32, 240, -5)

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
        self.batt = Battery(100, 50, 7.68)

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


class TestLinear2StageBattery(TestBatteryBase):
    def setUp(self):
        self.batt = Linear2StageBattery(100, 0, 7.68, 0)

    def test_valid_charge_no_noise_not_tail(self):
        self.batt = Linear2StageBattery(100, 0, 7.68, 0)
        with patch('numpy.random.normal', return_value=1.2):
            rate = self.batt.charge(16, 240, 5)
        self.assertAlmostEqual(rate, 16)
        self.assertAlmostEqual(self.batt.current_charging_power, 3.84)
        self.assertAlmostEqual(self.batt._current_charge, 0.32)

    def test_valid_charge_positive_noise_not_tail(self):
        self.batt = Linear2StageBattery(100, 0, 7.68, 1)
        with patch('numpy.random.normal', return_value=0.288):
            rate = self.batt.charge(16, 240, 5)
        self.assertAlmostEqual(rate, 14.8)
        self.assertAlmostEqual(self.batt.current_charging_power, 3.552)
        self.assertAlmostEqual(self.batt._current_charge, 0.296)

    def test_valid_charge_negative_noise_not_tail(self):
        self.batt = Linear2StageBattery(100, 0, 7.68, 1)
        with patch('numpy.random.normal', return_value=-0.288):
            rate = self.batt.charge(16, 240, 5)
        self.assertAlmostEqual(rate, 14.8)
        self.assertAlmostEqual(self.batt.current_charging_power, 3.552)
        self.assertAlmostEqual(self.batt._current_charge, 0.296)

    def test_valid_charge_no_noise_tail(self):
        self.batt = Linear2StageBattery(100, 85, 7.68, 0)
        rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 24)
        self.assertAlmostEqual(self.batt.current_charging_power, 5.76)
        self.assertAlmostEqual(self.batt._current_charge, 85.48)

    def test_valid_charge_positive_noise_tail(self):
        self.batt = Linear2StageBattery(100, 85, 7.68, 1)
        with patch('numpy.random.normal', return_value=0.288):
            rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 25.2)
        self.assertAlmostEqual(self.batt.current_charging_power, 6.048)
        self.assertAlmostEqual(self.batt._current_charge, 85.504)

    def test_valid_charge_negative_noise_tail(self):
        self.batt = Linear2StageBattery(100, 85, 7.68, 1)
        with patch('numpy.random.normal', return_value=-0.288):
            rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 22.8)
        self.assertAlmostEqual(self.batt.current_charging_power, 5.472)
        self.assertAlmostEqual(self.batt._current_charge, 85.456)

    def test_charge_over_max_rate_not_tail(self):
        self.batt = Linear2StageBattery(100, 0, 7.68, 0)
        rate = self.batt.charge(40, 240, 5)
        self.assertAlmostEqual(rate, 32)
        self.assertAlmostEqual(self.batt.current_charging_power, 7.68)
        self.assertAlmostEqual(self.batt._current_charge, 0.64)

    def test_charge_over_capacity(self):
        self.batt = Linear2StageBattery(100, 99, 7.68, 0)
        rate = self.batt.charge(32, 240, 5)
        self.assertAlmostEqual(rate, 1.6)
        self.assertAlmostEqual(self.batt.current_charging_power, 0.384)
        self.assertAlmostEqual(self.batt._current_charge, 99.032)


if __name__ == '__main__':
    unittest.main()
