from unittest import TestCase
from unittest.mock import create_autospec

from acnportal.acnsim.models import EV
from acnportal.acnsim.models import EVSE
from acnportal.acnsim.network.affinity_charging_network import AffinityChargingNetwork


class TestAffinityChargingNetwork(TestCase):
    def setUp(self):
        self.network = AffinityChargingNetwork()

    def test_arrive_single_station_regular_ev(self):
        evse = EVSE('PS-001')
        self.network.register_evse(evse, 208, 0)
        ev = create_autospec(EV)
        self.network.arrive(ev)
        self.assertEqual(ev.station_id, 'PS-001')
        self.assertEqual(evse.ev, ev)

    def test_arrive_multi_station_regular_ev(self):
        evse_ids = ['01', '02', '03']
        for evse_id in evse_ids:
            self.network.register_evse(EVSE(evse_id), 208, 0)

        ev = create_autospec(EV)
        self.network.arrive(ev)
        self.assertIn(ev.station_id, evse_ids)

    def test_arrive_no_evse_available(self):
        evse = EVSE('PS-001')
        self.network.register_evse(evse, 208, 0)
        ev1 = create_autospec(EV)
        self.network.arrive(ev1)
        self.assertEqual(ev1.station_id, 'PS-001')
        self.assertEqual(evse.ev, ev1)

        ev2 = create_autospec(EV)
        self.network.arrive(ev2)
        self.assertEqual(len(self.network.waiting_queue), 1)
        self.assertEqual(self.network.waiting_queue[0], ev2)

    def test_depart_queue_empty(self):
        evse = EVSE('PS-001')
        self.network.register_evse(evse, 208, 0)
        ev = create_autospec(EV)
        self.network.arrive(ev)
        self.network.depart(ev)
        self.assertIsNone(evse.ev)
        self.assertEqual(len(self.network.waiting_queue), 0)
        # self.assertEqual(len(self.network.notified), 0)

    def test_depart_queue_not_empty(self):
        evse = EVSE('PS-001')
        self.network.register_evse(evse, 208, 0)
        ev1 = create_autospec(EV)
        self.network.arrive(ev1)
        ev2 = create_autospec(EV)
        ev2.station_id = 'PS-002'
        self.network.arrive(ev2)
        self.assertEqual(len(self.network.waiting_queue), 1)

        self.network.depart(ev1)
        self.assertEqual(len(self.network.waiting_queue), 0)
        self.assertEqual(ev2.station_id, 'PS-001')
        # self.assertEqual(len(self.network.notified), 1)

    def test_depart_from_queue(self):
        evse = EVSE('PS-001')
        self.network.register_evse(evse, 208, 0)
        ev1 = create_autospec(EV)
        self.network.arrive(ev1)
        ev2 = create_autospec(EV)
        ev2.station_id = 'PS-001'
        self.network.arrive(ev2)
        self.assertEqual(len(self.network.waiting_queue), 1)

        self.network.depart(ev2)
        self.assertEqual(len(self.network.waiting_queue), 0)
