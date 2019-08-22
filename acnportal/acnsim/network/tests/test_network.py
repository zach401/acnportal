from unittest import TestCase
from unittest.mock import Mock, create_autospec

from acnportal.acnsim.models import EV
from acnportal.acnsim.models import EVSE, InvalidRateError
from acnportal.acnsim import ChargingNetwork

import pandas as pd
import numpy as np


class TestChargingNetwork(TestCase):
    def setUp(self):
        self.network = ChargingNetwork()

    def test_register_evse(self):
        evse = EVSE('PS-001')
        self.network.register_evse(evse, 240, 0)
        self.assertIn('PS-001', self.network._EVSEs)
        self.assertIs(self.network._EVSEs['PS-001'], evse)

    # noinspection PyUnresolvedReferences
    def test_plugin_station_exists(self):
        evse = EVSE('PS-001')
        evse.plugin = Mock(evse.plugin)
        self.network.register_evse(evse, 240, 0)
        ev = create_autospec(EV)

        self.network.plugin(ev, 'PS-001')
        evse.plugin.assert_called_once()

    def test_plugin_station_does_not_exist(self):
        ev = create_autospec(EV)
        with self.assertRaises(KeyError):
            self.network.plugin(ev, 'PS-001')

    def test_unplug_station_exists(self):
        evse = EVSE('PS-001')
        evse.unplug = Mock(evse.unplug)
        self.network.register_evse(evse, 240, 0)
        self.network.unplug('PS-001')
        evse.unplug.assert_called_once()

    def test_unplug_station_does_not_exist(self):
        with self.assertRaises(KeyError):
            self.network.unplug('PS-001')

    def test_get_ev_station_exists(self):
        evse = EVSE('PS-001')
        ev = Mock(EV)
        evse.plugin(ev)
        self.network.register_evse(evse, 240, 0)
        self.assertIs(self.network.get_ev('PS-001'), ev)

    def test_get_ev_station_does_not_exist(self):
        with self.assertRaises(KeyError):
            self.network.get_ev('PS-001')

    def test_update_pilots_valid_pilots(self):
        evse1 = EVSE('PS-001')
        self.network.register_evse(evse1, 240, 0)
        evse2 = EVSE('PS-002')
        self.network.register_evse(evse2, 240, 0)
        evse3 = EVSE('PS-003')
        self.network.register_evse(evse3, 240, 0)
        evse1.set_pilot = create_autospec(evse1.set_pilot)
        evse2.set_pilot = create_autospec(evse2.set_pilot)
        evse3.set_pilot = create_autospec(evse3.set_pilot)
        self.network.update_pilots(
            pd.DataFrame.from_dict(
                {'PS-001' : [24, 16], 'PS-002' : [16, 24]}
            ),
            0,
            5
        )
        evse1.set_pilot.assert_any_call(24, 240, 5)
        evse2.set_pilot.assert_any_call(16, 240, 5)
        evse3.set_pilot.assert_any_call(0, 240, 5)