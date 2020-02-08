""" Unit tests for analysis functions. """
from unittest import TestCase
from unittest.mock import Mock, create_autospec

from acnportal import acnsim
from acnportal.acnsim import Simulator
from acnportal.acnsim.tests.test_simulator import TestSimulator
from acnportal.acnsim import acndata_events
from acnportal.acnsim import sites
from acnportal.algorithms import BaseAlgorithm
from datetime import datetime

import pytz
import numpy as np
import os
import json
from copy import deepcopy


class TestAnalysis(TestCase):
    def test_datetimes_array(self):
        start = datetime(2018, 12, 31)
        network = acnsim.ChargingNetwork()
        evse1 = acnsim.EVSE('PS-001', max_rate=32)
        network.register_evse(evse1, 240, 0)
        scheduler = create_autospec(BaseAlgorithm)
        scheduler.max_recompute = None
        events = acnsim.EventQueue(events=[acnsim.Event(1)])
        simulator = Simulator(
            network, scheduler, events, start, period=240)
        simulator._iteration = 10

        datetime_array = acnsim.datetimes_array(simulator)
        expected_datetime_array = [
            np.datetime64('2018-12-31T00:00:00.000000'),
            np.datetime64('2018-12-31T04:00:00.000000'),
            np.datetime64('2018-12-31T08:00:00.000000'),
            np.datetime64('2018-12-31T12:00:00.000000'),
            np.datetime64('2018-12-31T16:00:00.000000'),
            np.datetime64('2018-12-31T20:00:00.000000'),
            np.datetime64('2019-01-01T00:00:00.000000'),
            np.datetime64('2019-01-01T04:00:00.000000'),
            np.datetime64('2019-01-01T08:00:00.000000'),
            np.datetime64('2019-01-01T12:00:00.000000')
        ]
        np.testing.assert_equal(datetime_array, expected_datetime_array)
