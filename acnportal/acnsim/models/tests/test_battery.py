import unittest
from unittest import TestCase
from unittest.mock import patch

from acnportal.acnsim.models import Battery, NoisyBattery, Linear2StageBattery


class TestBattery(TestCase):
    def setUp(self):
        self.batt = Battery(100, 50, 32)

    def test_valid_charge(self):
        rate = self.batt.charge(16)
        self.assertEqual(rate, 16)
        self.assertEqual(self.batt.current_charging_rate, 16)
        self.assertEqual(self.batt._current_charge, 66)

    def test_charge_over_max_rate(self):
        rate = self.batt.charge(55)
        self.assertEqual(rate, 32)
        self.assertEqual(self.batt.current_charging_rate, 32)
        self.assertEqual(self.batt._current_charge, 82)

    def test_charge_over_capacity(self):
        self.batt = Battery(100, 90, 32)
        rate = self.batt.charge(16)
        self.assertEqual(rate, 10)
        self.assertEqual(self.batt.current_charging_rate, 10)
        self.assertAlmostEqual(self.batt._current_charge, 100)


class TestNoisyBattery(TestCase):
    def setUp(self):
        self.batt = NoisyBattery(100, 50, 32, 1)

    def test_invalid_init_capacity(self):
        with self.assertRaises(ValueError):
            NoisyBattery(100, 110, 32, 1)

    def test_soc(self):
        self.assertAlmostEqual(self.batt._soc, 0.5)

    def test_valid_charge_positive_noise(self):
        with patch('numpy.random.normal', return_value=1.2):
            rate = self.batt.charge(16)
        self.assertEqual(rate, 14.8)
        self.assertEqual(self.batt.current_charging_rate, 14.8)
        self.assertEqual(self.batt._current_charge, 64.8)

    def test_valid_charge_negative_noise(self):
        with patch('numpy.random.normal', return_value=-1.2):
            rate = self.batt.charge(16)
        self.assertEqual(rate, 14.8)
        self.assertEqual(self.batt.current_charging_rate, 14.8)
        self.assertEqual(self.batt._current_charge, 64.8)

    def test_charge_over_max_rate(self):
        with patch('numpy.random.normal', return_value=1.2):
            rate = self.batt.charge(40)
        self.assertEqual(rate, 30.8)
        self.assertEqual(self.batt.current_charging_rate, 30.8)
        self.assertEqual(self.batt._current_charge, 80.8)

    def test_charge_over_capacity(self):
        almost_full_batt = NoisyBattery(100, 90, 32, 1)
        with patch('numpy.random.normal', return_value=1.2):
            rate = almost_full_batt.charge(16)
        self.assertEqual(rate, 8.8)
        self.assertEqual(almost_full_batt.current_charging_rate, 8.8)
        self.assertAlmostEqual(almost_full_batt._current_charge, 98.8)

    def test_valid_reset(self):
        self.batt.charge(16)
        self.batt.reset(45)
        self.assertEqual(self.batt.current_charging_rate, 0)
        self.assertEqual(self.batt._current_charge, 45)

    def test_reset_over_capacity(self):
        with self.assertRaises(ValueError):
            self.batt.reset(110)


class TestLinear2StageBattery(TestCase):
    def setUp(self):
        self.batt = Linear2StageBattery(100, 0, 32, 1)

    def test_valid_charge_no_noise_not_tail(self):
        self.batt = Linear2StageBattery(100, 0, 32, 0)
        with patch('numpy.random.normal', return_value=1.2):
            rate = self.batt.charge(16)
        self.assertEqual(rate, 16)
        self.assertEqual(self.batt.current_charging_rate, 16)
        self.assertEqual(self.batt._current_charge, 16)

    def test_valid_charge_positive_noise_not_tail(self):
        with patch('numpy.random.normal', return_value=1.2):
            rate = self.batt.charge(16)
        self.assertEqual(rate, 14.8)
        self.assertEqual(self.batt.current_charging_rate, 14.8)
        self.assertEqual(self.batt._current_charge, 14.8)

    def test_valid_charge_negative_noise_not_tail(self):
        with patch('numpy.random.normal', return_value=-1.2):
            rate = self.batt.charge(16)
        self.assertEqual(rate, 14.8)
        self.assertEqual(self.batt.current_charging_rate, 14.8)
        self.assertEqual(self.batt._current_charge, 14.8)

    def test_valid_charge_no_noise_tail(self):
        self.batt = Linear2StageBattery(1000, 850, 32, 0)
        rate = self.batt.charge(32)
        self.assertAlmostEqual(rate, 24)
        self.assertAlmostEqual(self.batt.current_charging_rate, 24)
        self.assertAlmostEqual(self.batt._current_charge, 874)

    def test_valid_charge_positive_noise_tail(self):
        self.batt = Linear2StageBattery(1000, 850, 32, 1)
        with patch('numpy.random.normal', return_value=1.2):
            rate = self.batt.charge(32)
        self.assertAlmostEqual(rate, 25.2)
        self.assertAlmostEqual(self.batt.current_charging_rate, 25.2)
        self.assertAlmostEqual(self.batt._current_charge, 875.2)

    def test_valid_charge_negative_noise_tail(self):
        self.batt = Linear2StageBattery(1000, 850, 32, 1)
        with patch('numpy.random.normal', return_value=-1.2):
            rate = self.batt.charge(32)
        self.assertAlmostEqual(rate, 22.8)
        self.assertAlmostEqual(self.batt.current_charging_rate, 22.8)
        self.assertAlmostEqual(self.batt._current_charge, 872.8)

    def test_charge_over_max_rate_not_tail(self):
        self.batt = Linear2StageBattery(100, 0, 32, 0)
        rate = self.batt.charge(40)
        self.assertEqual(rate, 32)
        self.assertEqual(self.batt.current_charging_rate, 32)
        self.assertEqual(self.batt._current_charge, 32)

    def test_charge_over_capacity(self):
        self.batt = Linear2StageBattery(1000, 990, 32, 0)
        rate = self.batt.charge(32)
        self.assertAlmostEqual(rate, 1.6)
        self.assertAlmostEqual(self.batt.current_charging_rate, 1.6)
        self.assertAlmostEqual(self.batt._current_charge, 991.6)

    def test_valid_reset(self):
        self.batt.charge(16)
        self.batt.reset(45)
        self.assertEqual(self.batt.current_charging_rate, 0)
        self.assertEqual(self.batt._current_charge, 45)

    def test_reset_over_capacity(self):
        with self.assertRaises(ValueError):
            self.batt.reset(110)


if __name__ == '__main__':
    unittest.main()
