""" Unit tests for analysis functions. """
from datetime import datetime
from unittest import TestCase
from unittest.mock import Mock, create_autospec

import numpy as np
import pytz

from acnportal import acnsim
from acnportal.algorithms import BaseAlgorithm


class TestAnalysis(TestCase):
    def setUp(self):
        start = datetime(2018, 12, 31, tzinfo=pytz.timezone("America/Los_Angeles"))
        network = acnsim.ChargingNetwork()
        evse1 = acnsim.EVSE("PS-001", max_rate=32)
        network.register_evse(evse1, 240, 0)
        scheduler = create_autospec(BaseAlgorithm)
        scheduler.max_recompute = None
        self.events = acnsim.EventQueue(events=[acnsim.Event(1)])
        self.simulator = acnsim.Simulator(
            network, scheduler, self.events, start, period=240
        )
        self.simulator._iteration = 10
        self.expected_datetime_array = [
            np.datetime64("2018-12-31T00:00:00.000000"),
            np.datetime64("2018-12-31T04:00:00.000000"),
            np.datetime64("2018-12-31T08:00:00.000000"),
            np.datetime64("2018-12-31T12:00:00.000000"),
            np.datetime64("2018-12-31T16:00:00.000000"),
            np.datetime64("2018-12-31T20:00:00.000000"),
            np.datetime64("2019-01-01T00:00:00.000000"),
            np.datetime64("2019-01-01T04:00:00.000000"),
            np.datetime64("2019-01-01T08:00:00.000000"),
            np.datetime64("2019-01-01T12:00:00.000000"),
        ]

    def test_datetimes_array_warning(self):
        with self.assertWarns(UserWarning):
            datetime_array = acnsim.datetimes_array(self.simulator)
            np.testing.assert_equal(datetime_array, self.expected_datetime_array)

    def test_datetimes_array(self):
        self.events.empty = Mock(self.events.empty)
        self.events.empty = lambda: True
        datetime_array = acnsim.datetimes_array(self.simulator)
        # Check that simulator start is unchanged.
        self.assertIsNotNone(self.simulator.start.tzinfo)
        np.testing.assert_equal(datetime_array, self.expected_datetime_array)
