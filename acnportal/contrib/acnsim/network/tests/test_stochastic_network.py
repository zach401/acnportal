# coding=utf-8
""" Tests for StochasticChargingNetwork functionality. """
from unittest import TestCase
from unittest.mock import Mock, patch

from acnportal.contrib.acnsim.network import StochasticNetwork
from acnportal.acnsim.models import EV, EVSE, Battery

RANDOM_CHOICE_PATCH_STR = "acnportal.contrib.acnsim.stochastic_network.random.choice"


class TestStochasticChargingNetwork(TestCase):
    def setUp(self) -> None:
        """ Set up tests for ChargingNetwork. """
        self.network = StochasticNetwork()
        self.evse_ids = ["PS-001", "PS-002", "PS-003"]
        for evse_id in self.evse_ids:
            self.network.register_evse(EVSE(evse_id), 240, 0)

    def test_available_evses(self) -> None:
        result = self.network.available_evses()
        self.assertListEqual(result, self.evse_ids)

    @patch(RANDOM_CHOICE_PATCH_STR)
    def test_plugin_ev_with_spot_available(self, choice_mock) -> None:
        ev = EV(0, 10, 10, "", "Session-01", Battery(100, 0, 6.6))
        choice_mock.return_value = "PS-002"
        self.network.plugin(ev)
        self.assertEqual(ev.station_id, "PS-002")
        self.assertIs(self.network.get_ev("PS-002"), ev)

    def test_plugin_ev_no_spot_available_empty_waiting_queue(self) -> None:
        ev = EV(0, 10, 10, "", "Session-01", Battery(100, 0, 6.6))

        # Mock available_evses to behavior like the network is full.
        self.network.available_evses = Mock(return_value=[])

        self.network.plugin(ev)

        # EV should be added to waiting_queue
        self.assertEqual(len(self.network.waiting_queue), 1)
        self.assertIn(ev.session_id, self.network.waiting_queue)
        self.assertIs(self.network.waiting_queue[ev.session_id], ev)

    def test_plugin_ev_no_spot_available_evs_in_waiting_queue(self) -> None:
        ev1 = EV(0, 10, 10, "", "Session-01", Battery(100, 0, 6.6))
        ev2 = EV(0, 10, 10, "", "Session-02", Battery(100, 0, 6.6))

        # Mock available_evses to behavior like the network is full.
        self.network.available_evses = Mock(return_value=[])

        # Add ev1 to the waiting queue
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

    @patch(RANDOM_CHOICE_PATCH_STR)
    def test_unplug_station_exists_session_id_matches_queue_empty(
        self, choice_mock
    ) -> None:
        ev = EV(0, 10, 10, "", "Session-01", Battery(100, 0, 6.6))
        choice_mock.return_value = "PS-002"

        self.network.plugin(ev)

        self.network._EVSEs["PS-002"].unplug = Mock()
        self.network.unplug(ev.station_id, ev.session_id)
        self.network._EVSEs["PS-002"].unplug.assert_called()

    @patch(RANDOM_CHOICE_PATCH_STR)
    def test_unplug_station_exists_session_id_matches_ev_in_queue(
        self, choice_mock
    ) -> None:
        ev1 = EV(0, 10, 10, "", "Session-01", Battery(100, 0, 6.6))
        ev2 = EV(0, 10, 10, "", "Session-02", Battery(100, 0, 6.6))

        # Plug ev1 into space PS-001
        choice_mock.return_value = "PS-002"
        self.network.plugin(ev1)

        # Add ev2 to the waiting queue
        self.network.waiting_queue[ev2.session_id] = ev2
        self.network.waiting_queue.move_to_end(ev2.session_id)

        # Mock the plugin method so we can assert it was called.
        self.network._EVSEs["PS-002"].plugin = Mock()

        # Unplug ev1
        self.network.unplug(ev1.station_id, ev1.session_id)

        # ev2 should be plugged in after ev1 leaves
        self.network._EVSEs["PS-002"].plugin.assert_called_with(ev2)

        # swaps counter should be incremented
        self.assertEqual(self.network.swaps, 1)

    def test_unplug_when_ev_in_waiting_queue(self):
        ev1 = EV(0, 10, 10, "", "Session-01", Battery(100, 0, 6.6))

        # Add ev2 to the waiting queue
        self.network.waiting_queue[ev1.session_id] = ev1
        self.network.waiting_queue.move_to_end(ev1.session_id)

        self.assertEqual(len(self.network.waiting_queue), 1)

        self.network.unplug(ev1.station_id, ev1.session_id)

        # ev1 should be removed from the waiting queue
        self.assertEqual(len(self.network.waiting_queue), 0)

        # never_charged counter should be incremented
        self.assertEqual(self.network.never_charged, 1)

    @patch(RANDOM_CHOICE_PATCH_STR)
    def test_unplug_station_exists_session_id_mismatch(self, choice_mock) -> None:
        ev = EV(0, 10, 10, "", "Session-01", Battery(100, 0, 6.6))
        choice_mock.return_value = "PS-002"
        self.network.plugin(ev)
        self.network._EVSEs["PS-002"].unplug = Mock()
        self.network.unplug(ev.station_id, "Incorrect ID")
        # noinspection PyUnresolvedReferences
        self.network._EVSEs["PS-002"].unplug.assert_not_called()

    def test_unplug_station_does_not_exist(self) -> None:
        with self.assertRaises(KeyError):
            self.network.unplug("PS-005", "Session-01")

    @patch(RANDOM_CHOICE_PATCH_STR)
    def test_post_charging_update_early_departure_ev_in_queue(self, choice_mock):
        self.network.early_departure = True

        ev1 = EV(0, 10, 10, "", "Session-01", Battery(100, 0, 6.6))
        ev2 = EV(0, 10, 10, "", "Session-02", Battery(100, 0, 6.6))

        # Plug ev1 into space PS-001
        choice_mock.return_value = "PS-002"
        self.network.plugin(ev1)

        # Fully charge ev1
        ev1._energy_delivered = 10

        # Add ev2 to the waiting queue
        self.network.waiting_queue[ev2.session_id] = ev2
        self.network.waiting_queue.move_to_end(ev2.session_id)

        # Test that unplug is called during post_charging_update.
        self.network.unplug = Mock()
        self.network.post_charging_update()
        self.network.unplug.assert_called_with(ev1.station_id, ev1.session_id)
        self.assertEqual(self.network.early_unplug, 1)

    @patch(RANDOM_CHOICE_PATCH_STR)
    def test_post_charging_update_early_departure_off_ev_in_queue(self, choice_mock):
        self.network.early_departure = False

        ev1 = EV(0, 10, 10, "", "Session-01", Battery(100, 0, 6.6))
        ev2 = EV(0, 10, 10, "", "Session-02", Battery(100, 0, 6.6))

        # Plug ev1 into space PS-001
        choice_mock.return_value = "PS-002"
        self.network.plugin(ev1)

        # Fully charge ev1
        ev1._energy_delivered = 10

        # Add ev2 to the waiting queue
        self.network.waiting_queue[ev2.session_id] = ev2
        self.network.waiting_queue.move_to_end(ev2.session_id)

        # Test that unplug is called during post_charging_update.
        self.network.unplug = Mock()
        self.network.post_charging_update()
        self.network.unplug.assert_not_called()

    @patch(RANDOM_CHOICE_PATCH_STR)
    def test_post_charging_update_early_departure_ev_not_full(self, choice_mock):
        self.network.early_departure = True

        ev1 = EV(0, 10, 10, "", "Session-01", Battery(100, 0, 6.6))
        ev2 = EV(0, 10, 10, "", "Session-02", Battery(100, 0, 6.6))

        # Plug ev1 into space PS-001
        choice_mock.return_value = "PS-002"
        self.network.plugin(ev1)

        # ev1 not fully charged
        ev1._energy_delivered = 5

        # Add ev2 to the waiting queue
        self.network.waiting_queue[ev2.session_id] = ev2
        self.network.waiting_queue.move_to_end(ev2.session_id)

        # Test that unplug is called during post_charging_update.
        self.network.unplug = Mock()
        self.network.post_charging_update()
        self.network.unplug.assert_not_called()

    @patch(RANDOM_CHOICE_PATCH_STR)
    def test_post_charging_update_early_departure_no_ev_in_queue(self, choice_mock):
        self.network.early_departure = True

        ev1 = EV(0, 10, 10, "", "Session-01", Battery(100, 0, 6.6))

        # Plug ev1 into space PS-001
        choice_mock.return_value = "PS-002"
        self.network.plugin(ev1)

        # Fully charge ev1
        ev1._energy_delivered = 10

        # Test that unplug is called during post_charging_update.
        self.network.unplug = Mock()
        self.network.post_charging_update()
        self.network.unplug.assert_not_called()
