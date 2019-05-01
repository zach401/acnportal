from unittest import TestCase
from unittest.mock import create_autospec

from acnportal.acnsim.models import EV
from acnportal.acnsim.models import EVSE, DeadbandEVSE, FiniteRatesEVSE, InvalidRateError, StationOccupiedError


class TestEVSE(TestCase):
    def setUp(self):
        self.evse = EVSE('0001')

    def test_plugin_unoccupied(self):
        ev = create_autospec(EV)
        self.evse.plugin(ev)
        self.assertEqual(self.evse.ev, ev)

    def test_plugin_occupied(self):
        ev = create_autospec(EV)
        ev2 = create_autospec(EV)
        self.evse.plugin(ev)
        self.assertEqual(self.evse.ev, ev)
        with self.assertRaises(StationOccupiedError):
            self.evse.plugin(ev2)

    def test_unplug_occupied(self):
        ev = create_autospec(EV)
        self.evse.plugin(ev)
        self.evse.unplug()
        self.assertIsNone(self.evse.ev)

    def test_unplug_unoccupied(self):
        self.evse.unplug()
        self.assertIsNone(self.evse.ev)

    def test_set_pilot_has_ev_valid_rate(self):
        ev = create_autospec(EV)
        self.evse.plugin(ev)
        self.evse.set_pilot(16, 240, 5)
        self.assertEqual(self.evse.current_pilot, 16)
        self.evse.ev.charge.assert_called_once()

    def test_set_pilot_has_ev_negative_rate(self):
        ev = create_autospec(EV)
        self.evse.plugin(ev)
        with self.assertRaises(InvalidRateError):
            self.evse.set_pilot(-1, 240, 5)

    def test_set_pilot_no_ev_negative_rate(self):
        with self.assertRaises(InvalidRateError):
            self.evse.set_pilot(-1, 240, 5)


class TestDeadbandEVSE(TestEVSE):
    def setUp(self):
        self.evse = DeadbandEVSE('0001', 6, max_rate=8, min_rate=0)

    def test_set_pilot_has_ev_invalid_rate(self):
        ev = create_autospec(EV)
        self.evse.plugin(ev)
        with self.assertRaises(InvalidRateError):
            self.evse.set_pilot(5, 240, 5)

    def test_set_pilot_no_ev_invalid_rate(self):
        with self.assertRaises(InvalidRateError):
            self.evse.set_pilot(5, 240, 5)


class TestFiniteRatesEVSE(TestEVSE):
    def setUp(self):
        self.evse = FiniteRatesEVSE('0001', [0, 8, 16, 24, 32])

    def test_set_pilot_has_ev_invalid_rate(self):
        ev = create_autospec(EV)
        self.evse.plugin(ev)
        with self.assertRaises(InvalidRateError):
            self.evse.set_pilot(17.2, 240, 5)

    def test_set_pilot_no_ev_invalid_rate(self):
        with self.assertRaises(InvalidRateError):
            self.evse.set_pilot(17.2, 240, 5)
