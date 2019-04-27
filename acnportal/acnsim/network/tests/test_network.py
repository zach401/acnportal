from unittest import TestCase
from unittest.mock import Mock, create_autospec

from acnportal.acnsim.models import EV
from acnportal.acnsim.models import EVSE
from acnportal.acnsim import ChargingNetwork


class TestChargingNetwork(TestCase):
    def setUp(self):
        self.network = ChargingNetwork()

    def test_register_evse(self):
        evse = EVSE('PS-001', voltage=240)
        self.network.register_evse(evse)
        self.assertIn('PS-001', self.network._EVSEs)
        self.assertIs(self.network._EVSEs['PS-001'], evse)

    # noinspection PyUnresolvedReferences
    def test_plugin_station_exists(self):
        evse = EVSE('PS-001', voltage=240)
        evse.plugin = Mock(evse.plugin)
        self.network.register_evse(evse)
        ev = create_autospec(EV)

        self.network.plugin(ev, 'PS-001')
        evse.plugin.assert_called_once()

    def test_plugin_station_does_not_exist(self):
        ev = create_autospec(EV)
        with self.assertRaises(KeyError):
            self.network.plugin(ev, 'PS-001')

    def test_unplug_station_exists(self):
        evse = EVSE('PS-001', voltage=240)
        evse.unplug = Mock(evse.unplug)
        self.network.register_evse(evse)
        self.network.unplug('PS-001')
        evse.unplug.assert_called_once()

    def test_unplug_station_does_not_exist(self):
        with self.assertRaises(KeyError):
            self.network.unplug('PS-001')

    def test_get_ev_station_exists(self):
        evse = EVSE('PS-001', voltage=240)
        ev = Mock(EV)
        evse.plugin(ev)
        self.network.register_evse(evse)
        self.assertIs(self.network.get_ev('PS-001'), ev)

    def test_get_ev_station_does_not_exist(self):
        with self.assertRaises(KeyError):
            self.network.get_ev('PS-001')
