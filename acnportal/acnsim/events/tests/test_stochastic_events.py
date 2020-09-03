import unittest
from unittest.mock import Mock
from acnportal.acnsim.events.stochastic_events import _convert_ev_matrix, \
    StochasticSessionModel, stochastic_events
import numpy as np
from acnportal.acnsim import Battery


class TestConvertEVMatrix(unittest.TestCase):
    def setUp(self):
        self.period = 5
        self.voltage = 208
        self.max_battery_power = 7

    def _default_tests(self, evs):
        with self.subTest("test_length_of_evs"):
            self.assertEqual(3, len(evs))

        with self.subTest("test_arrival_times"):
            self.assertListEqual([ev.arrival for ev in evs],
                                 [78, 99, 120])

        with self.subTest("test_station_ids"):
            self.assertListEqual([ev.station_id for ev in evs],
                                 [f"station_{i}" for i in range(3)])

        with self.subTest("test_session_ids"):
            self.assertListEqual([ev.session_id for ev in evs],
                                 [f"session_{i}" for i in range(3)])

        with self.subTest("test_battery"):
            for ev in evs:
                self.assertIsInstance(ev._battery, Battery)
                self.assertEqual(ev.maximum_charging_power, self.max_battery_power)

    def test_ev_matrix(self):
        ev_matrix = np.array([[6.5, 8, 10], [8.3, 6.05, 3], [10, 3, 6.6]])
        evs = _convert_ev_matrix(ev_matrix, self.period, self.voltage,
                                 self.max_battery_power)

        self._default_tests(evs)

        with self.subTest("test_departure_time"):
            self.assertListEqual([ev.departure for ev in evs],
                                 [174, 172, 156])

        with self.subTest("test_requested_energy"):
            self.assertListEqual([ev.requested_energy for ev in evs],
                                 [10, 3, 6.6])

    def test_ev_matrix_force_feasible(self):
        ev_matrix = np.array([[6.5, 1, 10], [8.3, 6.05, 3], [10, 3, 6.6]])
        evs = _convert_ev_matrix(ev_matrix, self.period, self.voltage,
                                 self.max_battery_power,
                                 force_feasible=True)

        self._default_tests(evs)

        with self.subTest("test_departure_time"):
            self.assertListEqual([ev.departure for ev in evs],
                                 [90, 172, 156])

        with self.subTest("test_requested_energy"):
            self.assertListEqual([ev.requested_energy for ev in evs],
                                 [self.max_battery_power, 3, 6.6])

    def test_ev_matrix_max_len(self):
        ev_matrix = np.array([[6.5, 8, 10], [8.3, 6.05, 3], [10, 3, 6.6]])
        evs = _convert_ev_matrix(ev_matrix, self.period, self.voltage,
                                 self.max_battery_power,
                                 max_len=3)

        self._default_tests(evs)

        with self.subTest("test_departure_time"):
            self.assertListEqual([ev.departure for ev in evs],
                                 [114, 135, 156])

        with self.subTest("test_requested_energy"):
            self.assertListEqual([ev.requested_energy for ev in evs],
                                 [10, 3, 6.6])

    def test_ev_matrix_max_len_force_feasible(self):
        ev_matrix = np.array([[6.5, 8, 10], [8.3, 6.05, 3], [10, 3, 15]])
        evs = _convert_ev_matrix(ev_matrix, self.period, self.voltage,
                                 self.max_battery_power,
                                 max_len=1, force_feasible=True)

        self._default_tests(evs)

        with self.subTest("test_departure_time"):
            self.assertListEqual([ev.departure for ev in evs],
                                 [90, 111, 132])

        with self.subTest("test_requested_energy"):
            self.assertListEqual([ev.requested_energy for ev in evs],
                                 [self.max_battery_power, 3, self.max_battery_power])


class TestStochasticEvents(unittest.TestCase):
    def test_multiple_days(self):
        model = StochasticSessionModel()
        ev_matrix = np.array([[6.5, 8, 10], [8.3, 6.05, 3], [10, 3, 6.6]])
        model.get_sessions = Mock(return_value=ev_matrix)

        events = stochastic_events(model, [3, 3], 5, 208, 7)
        self.assertEqual(6, len(events))


if __name__ == '__main__':
    unittest.main()
