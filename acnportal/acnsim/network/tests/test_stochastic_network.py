# coding=utf-8
""" Tests for StochasticChargingNetwork functionality. """
from unittest import TestCase
from unittest.mock import Mock, patch

from acnportal.acnsim import StochasticNetwork
from acnportal.acnsim.models import EV, EVSE, Battery


class TestStochasticChargingNetwork(TestCase):
    def setUp(self) -> None:
        """ Set up tests for ChargingNetwork. """
        self.network = StochasticNetwork()
        self.evse_ids = ["PS-001", "PS-002", "PS-003"]
        for evse_id in self.evse_ids:
            self.network.register_evse(EVSE(evse_id), 240, 0)

    def test_available_evses(self):
        result = self.network.available_evses()
        self.assertListEqual(result, self.evse_ids)

    @patch("stochastic_network.random.choice")
    def test_plugin_ev_with_spot_available(self, choice_mock):
        ev = EV(0, 10, 10, "", "Session-01", Battery(100, 0, 6.6))
        choice_mock.return_value = "PS-002"
        self.network.plugin(ev)
        self.assertEqual(ev.station_id, "PS-002")
        self.assertIs(self.network.get_ev("PS-002"), ev)

    def test_plugin_ev_no_spot_available_empty_waiting_queue(self):
        ev = EV(0, 10, 10, "", "Session-01", Battery(100, 0, 6.6))

        # Mock available_evses to behavior like the network is full.
        self.network.available_evses = Mock(return_value=[])

        self.network.plugin(ev)

        # EV should be added to waiting_queue
        self.assertEqual(len(self.network.waiting_queue), 1)
        self.assertIn(ev.session_id, self.network.waiting_queue)
        self.assertIs(self.network.waiting_queue[ev.session_id], ev)

    def test_plugin_ev_no_spot_available_evs_in_waiting_queue(self):
        ev1 = EV(0, 10, 10, "", "Session-01", Battery(100, 0, 6.6))
        ev2 = EV(0, 10, 10, "", "Session-02", Battery(100, 0, 6.6))

        # Mock available_evses to behavior like the network is full.
        self.network.available_evses = Mock(return_value=[])

        # Add ev2 to the waiting queue
        self.network.waiting_queue[ev1.session_id] = ev1
        self.network.waiting_queue.move_to_end(ev1.session_id)

        self.network.plugin(ev2)

        # EV should be added to waiting_queue
        self.assertEqual(len(self.network.waiting_queue), 2)

        # ev1 should be after ev2 in the queue
        _, first_in_queue = self.network.waiting_queue.popitem(last=False)
        _, second_in_queue = self.network.waiting_queue.popitem(last=False)
        self.assertIs(first_in_queue, ev1)
        self.assertIs(second_in_queue, ev2)