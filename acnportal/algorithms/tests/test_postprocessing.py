# coding=utf-8
""" Tests for Postprocessing algorithms. """
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from acnportal.acnsim.simulator import InvalidScheduleError
from acnportal.algorithms.postprocessing import format_array_schedule


class TestFormatArraySchedule(TestCase):
    def setUp(self) -> None:
        """ Tests for postprocessing.format_array_schedule. """
        self.infrastructure = Mock()
        self.infrastructure.station_ids = ["PS-001", "PS-002", "PS-003", "PS-004"]

    def test_invalid_array_schedule(self) -> None:
        schedule: np.ndarray = np.array([1, 2, 3])
        with self.assertRaises(InvalidScheduleError):
            _ = format_array_schedule(schedule, self.infrastructure)

    def test_scalar_array_schedule(self) -> None:
        schedule: np.ndarray = np.array([1, 2, 3, 4])
        self.assertEqual(
            format_array_schedule(schedule, self.infrastructure),
            {"PS-001": [1.0], "PS-002": [2.0], "PS-003": [3.0], "PS-004": [4.0]},
        )

    def test_array_of_arrays_schedule(self) -> None:
        schedule: np.ndarray = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.assertEqual(
            format_array_schedule(schedule, self.infrastructure),
            {
                "PS-001": [1.0, 2.0],
                "PS-002": [2.0, 3.0],
                "PS-003": [3.0, 4.0],
                "PS-004": [4.0, 5.0],
            },
        )
