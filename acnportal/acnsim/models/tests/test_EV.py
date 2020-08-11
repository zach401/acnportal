from unittest import TestCase
from unittest.mock import Mock, create_autospec

from acnportal.acnsim.models import Battery
from acnportal.acnsim.models import EV


class TestEV(TestCase):
    def setUp(self):
        basicBatt = create_autospec(Battery)
        self.ev = EV(0, 10, 25.0, "PS-001", "0001", basicBatt)

    def test_charge_valid_rate(self):
        self.ev._battery.charge = Mock(return_value=16)
        rate = self.ev.charge(16, 240, 5)
        self.assertEqual(rate, 16)
        self.assertAlmostEqual(self.ev.energy_delivered, 0.32)
        self.ev._battery.charge.assert_called_once()

    def test_charge_over_max_rate(self):
        self.ev._battery.charge = Mock(return_value=32)
        rate = self.ev.charge(40, 240, 5)
        self.assertEqual(rate, 32)
        self.assertAlmostEqual(self.ev.energy_delivered, 0.64)
        self.ev._battery.charge.assert_called_once()

    def test_reset(self):
        self.ev.reset()
        self.assertEqual(self.ev.energy_delivered, 0)
        self.ev._battery.reset.assert_called_once()
