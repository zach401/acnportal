from unittest import TestCase
from unittest.mock import Mock, create_autospec
import warnings

from acnportal.acnsim.models import EV
from acnportal.acnsim.models import EVSE
from acnportal.acnsim import ChargingNetwork


class TestChargingNetwork(TestCase):
    def setUp(self):
        self.network = ChargingNetwork()

    def test_register_evse(self):
        evse = EVSE('PS-001')
        self.network.register_evse(evse, 208, 0)
        self.assertIn('PS-001', self.network._EVSEs)
        self.assertIs(self.network._EVSEs['PS-001'], evse)

    # noinspection PyUnresolvedReferences
    def test_arrive_station_exists(self):
        evse = EVSE('PS-001')
        evse.plugin = Mock(evse.plugin)
        self.network.register_evse(evse, 208, 0)
        ev = create_autospec(EV)
        ev.station_id = 'PS-001'
        self.network.arrive(ev)
        evse.plugin.assert_called_once()

    def test_arrive_station_does_not_exist(self):
        ev = create_autospec(EV)
        ev.station_id = 'PS-001'
        with self.assertRaises(KeyError):
            self.network.arrive(ev)

    # noinspection PyUnresolvedReferences
    def test_depart_station_exists_ev_plugged_in(self):
        evse = EVSE('PS-001')
        evse.unplug = Mock(evse.unplug)
        self.network.register_evse(evse, 208, 0)
        ev = Mock(EV)
        ev.station_id = 'PS-001'
        self.network.arrive(ev)
        self.network.depart(ev)
        evse.unplug.assert_called_once()

    # noinspection PyUnresolvedReferences
    def test_depart_station_exists_ev_absent(self):
        evse = EVSE('PS-001')
        evse.unplug = Mock(evse.unplug)
        self.network.register_evse(evse, 208, 0)
        ev = Mock(EV)
        ev.station_id = 'PS-001'

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            self.network.depart(ev)
            evse.unplug.assert_not_called()
            self.assertEqual(len(w), 1)

    def test_depart_station_does_not_exist(self):
        ev = Mock(EV)
        ev.station_id = 'PS-001'
        with self.assertRaises(KeyError):
            self.network.depart(ev)

    def test_get_ev_station_exists(self):
        evse = EVSE('PS-001')
        ev = Mock(EV)
        evse.plugin(ev)
        self.network.register_evse(evse, 208, 0)
        self.assertIs(self.network.get_ev('PS-001'), ev)

    def test_get_ev_station_does_not_exist(self):
        with self.assertRaises(KeyError):
            self.network.get_ev('PS-001')
