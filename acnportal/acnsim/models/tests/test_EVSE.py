from unittest import TestCase
from unittest.mock import create_autospec

from acnportal.acnsim.models import EV
from acnportal.acnsim.models import EVSE, DeadbandEVSE, FiniteRatesEVSE
from acnportal.acnsim.models import InvalidRateError, StationOccupiedError


class TestEVSE(TestCase):
    def setUp(self):
        self.max_rate = 32
        self.mid_rate = 16
        self.atol = 1e-3
        self.evse = EVSE("0001", max_rate=self.max_rate)

    def test_allowable_pilot_signals_default(self):
        inf_evse = EVSE("0001")
        self.assertEqual(inf_evse.allowable_pilot_signals, [0, float("inf")])

    def test_allowable_pilot_signals(self):
        self.assertEqual(self.evse.allowable_pilot_signals, [0, self.max_rate])

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

    def _set_pilot_has_ev_valid_rate_helper(self, pilot):
        ev = create_autospec(EV)
        self.evse.plugin(ev)
        self.evse.set_pilot(pilot, 240, 5)
        self.assertEqual(self.evse.current_pilot, pilot)
        self.evse.ev.charge.assert_called_once()

    def test_set_pilot_has_ev_valid_rate(self):
        self._set_pilot_has_ev_valid_rate_helper(self.mid_rate)

    def test_set_pilot_has_ev_almost_large_rate(self):
        self._set_pilot_has_ev_valid_rate_helper(
            self.max_rate + self.atol - self.atol / 1e2
        )

    def test_set_pilot_has_ev_almost_small_rate(self):
        self._set_pilot_has_ev_valid_rate_helper(-self.atol + self.atol / 1e2)

    def test_set_pilot_has_ev_negative_rate(self):
        ev = create_autospec(EV)
        self.evse.plugin(ev)
        with self.assertRaises(InvalidRateError):
            self.evse.set_pilot(-1, 240, 5)

    def _set_pilot_no_ev_invalid_rate_helper(self, pilot):
        with self.assertRaises(InvalidRateError):
            self.evse.set_pilot(pilot, 240, 5)

    def test_set_pilot_no_ev_negative_rate(self):
        self._set_pilot_no_ev_invalid_rate_helper(-1)

    def test_set_pilot_no_ev_large_rate(self):
        self._set_pilot_no_ev_invalid_rate_helper(self.max_rate + 1)

    def test_set_pilot_no_ev_barely_large_rate(self):
        self._set_pilot_no_ev_invalid_rate_helper(
            self.max_rate + self.atol + self.atol / 1e2
        )

    def test_set_pilot_no_ev_barely_small_rate(self):
        self._set_pilot_no_ev_invalid_rate_helper(-self.atol - self.atol / 1e2)


class TestDeadbandEVSE(TestEVSE):
    def setUp(self):
        super().setUp()
        self.deadband_end = 6
        self.evse = DeadbandEVSE("0001", self.deadband_end, max_rate=self.max_rate)

    def test_allowable_pilot_signals(self):
        self.assertEqual(
            self.evse.allowable_pilot_signals, [self.deadband_end, self.max_rate]
        )

    def test_set_pilot_has_ev_almost_over_zero_rate(self):
        self._set_pilot_has_ev_valid_rate_helper(self.atol - self.atol / 1e2)

    def test_set_pilot_has_ev_almost_under_deadband_rate(self):
        self._set_pilot_has_ev_valid_rate_helper(
            self.deadband_end - self.atol + self.atol / 1e2
        )

    def test_set_pilot_has_ev_invalid_rate(self):
        ev = create_autospec(EV)
        self.evse.plugin(ev)
        with self.assertRaises(InvalidRateError):
            self.evse.set_pilot(self.deadband_end - 1, 240, 5)

    def test_set_pilot_no_ev_invalid_rate(self):
        self._set_pilot_no_ev_invalid_rate_helper(self.deadband_end - 1)

    def test_set_pilot_no_ev_barely_over_zero_rate(self):
        self._set_pilot_no_ev_invalid_rate_helper(self.atol + self.atol / 1e2)

    def test_set_pilot_has_ev_barely_under_deadband_rate(self):
        self._set_pilot_no_ev_invalid_rate_helper(
            self.deadband_end - self.atol - self.atol / 1e2
        )


class TestFiniteRatesEVSE(TestEVSE):
    def setUp(self):
        super().setUp()
        self.evse = FiniteRatesEVSE("0001", [0, 8, 16, 24, self.max_rate])

    def test_max_rate_finite_rates_evse(self):
        self.assertEqual(self.evse.max_rate, self.max_rate)

    def test_min_rate_finite_rates_evse(self):
        self.assertEqual(self.evse.min_rate, 8)

    def test_allowable_pilot_signals(self):
        self.assertEqual(
            self.evse.allowable_pilot_signals, [0, 8, 16, 24, self.max_rate]
        )

    def test_allowable_pilot_signals_unsorted(self):
        evse2 = FiniteRatesEVSE("0002", [self.max_rate, 16, 8, 24, 0])
        self.assertEqual(evse2.allowable_pilot_signals, [0, 8, 16, 24, self.max_rate])

    def test_allowable_pilot_signals_duplicates(self):
        evse3 = FiniteRatesEVSE("0003", [0, 8, 8, 24, 16, 24, self.max_rate])
        self.assertEqual(evse3.allowable_pilot_signals, [0, 8, 16, 24, self.max_rate])

    def test_allowable_pilot_signals_no_zero(self):
        evse4 = FiniteRatesEVSE("0004", [8, 24, 16, self.max_rate])
        self.assertEqual(evse4.allowable_pilot_signals, [0, 8, 16, 24, self.max_rate])

    def test_set_pilot_has_ev_almost_over_finite_rate(self):
        self._set_pilot_has_ev_valid_rate_helper(
            self.mid_rate + self.atol - self.atol / 1e2
        )

    def test_set_pilot_has_ev_almost_under_finite_rate(self):
        self._set_pilot_has_ev_valid_rate_helper(
            self.mid_rate - self.atol + self.atol / 1e2
        )

    def test_set_pilot_has_ev_invalid_rate(self):
        ev = create_autospec(EV)
        self.evse.plugin(ev)
        with self.assertRaises(InvalidRateError):
            self.evse.set_pilot(17.2, 240, 5)

    def test_set_pilot_no_ev_invalid_rate(self):
        self._set_pilot_no_ev_invalid_rate_helper(17.2)

    def test_set_pilot_no_ev_barely_over_finite_rate(self):
        self._set_pilot_no_ev_invalid_rate_helper(
            self.max_rate + self.atol + self.atol / 1e2
        )

    def test_set_pilot_no_ev_barely_under_finite_rate(self):
        self._set_pilot_no_ev_invalid_rate_helper(
            self.max_rate - self.atol - self.atol / 1e2
        )
