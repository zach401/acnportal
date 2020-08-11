from collections import OrderedDict
from unittest import TestCase
from unittest.mock import Mock, create_autospec

import numpy as np
import pandas as pd

from acnportal.acnsim import ChargingNetwork
from acnportal.acnsim import Current
from acnportal.acnsim.models import EV
from acnportal.acnsim.models import EVSE


class TestChargingNetwork(TestCase):
    def setUp(self):
        self.network = ChargingNetwork()
        self.network_with_tolerance = ChargingNetwork(
            violation_tolerance=1e-3, relative_tolerance=1e-5
        )

    def _test_init_empty(self, network):
        self.assertEqual(network._EVSEs, OrderedDict())
        self.assertEqual(network.constraint_matrix, None)
        self.assertEqual(network.constraint_index, [])
        np.testing.assert_equal(network.magnitudes, np.array([]))
        np.testing.assert_equal(network._voltages, np.array([]))
        np.testing.assert_equal(network._phase_angles, np.array([]))

    def test_init_default_tolerance(self):
        self._test_init_empty(self.network)
        self.assertEqual(self.network.violation_tolerance, 1e-5)
        self.assertEqual(self.network.relative_tolerance, 1e-7)

    def test_init_default_tolerance_set(self):
        self._test_init_empty(self.network_with_tolerance)
        self.assertEqual(self.network_with_tolerance.violation_tolerance, 1e-3)
        self.assertEqual(self.network_with_tolerance.relative_tolerance, 1e-5)

    def test_register_evse(self):
        evse1 = EVSE("PS-001")
        self.network.register_evse(evse1, 240, -30)
        evse2 = EVSE("PS-002")
        evse3 = EVSE("PS-003")
        self.network.register_evse(evse3, 100, 150)
        self.network.register_evse(evse2, 140, 90)
        self.assertIn("PS-001", self.network._EVSEs)
        self.assertIs(self.network._EVSEs["PS-001"], evse1)
        self.assertIn("PS-002", self.network._EVSEs)
        self.assertIs(self.network._EVSEs["PS-002"], evse2)
        self.assertIn("PS-003", self.network._EVSEs)
        self.assertIs(self.network._EVSEs["PS-003"], evse3)
        self.assertEqual(self.network.station_ids, ["PS-001", "PS-003", "PS-002"])
        np.testing.assert_allclose(self.network._phase_angles, np.array([-30, 150, 90]))
        np.testing.assert_allclose(self.network._voltages, np.array([240, 100, 140]))

    # noinspection PyUnresolvedReferences
    def test_plugin_station_exists(self):
        evse = EVSE("PS-001")
        evse.plugin = Mock(evse.plugin)
        self.network.register_evse(evse, 240, 0)
        ev = create_autospec(EV)

        self.network.plugin(ev, "PS-001")
        evse.plugin.assert_called_once()

    def test_plugin_station_does_not_exist(self):
        ev = create_autospec(EV)
        with self.assertRaises(KeyError):
            self.network.plugin(ev, "PS-001")

    # noinspection PyUnresolvedReferences
    def test_unplug_station_exists(self):
        evse = EVSE("PS-001")
        evse.unplug = Mock(evse.unplug)
        self.network.register_evse(evse, 240, 0)
        self.network.unplug("PS-001")
        evse.unplug.assert_called_once()

    def test_unplug_station_does_not_exist(self):
        with self.assertRaises(KeyError):
            self.network.unplug("PS-001")

    def test_get_ev_station_exists(self):
        evse = EVSE("PS-001")
        ev = Mock(EV)
        evse.plugin(ev)
        self.network.register_evse(evse, 240, 0)
        self.assertIs(self.network.get_ev("PS-001"), ev)

    def test_get_ev_station_does_not_exist(self):
        with self.assertRaises(KeyError):
            self.network.get_ev("PS-001")

    def test_update_pilots_valid_pilots(self):
        evse1 = EVSE("PS-001")
        self.network.register_evse(evse1, 240, 0)
        evse2 = EVSE("PS-002")
        self.network.register_evse(evse2, 240, 0)
        evse3 = EVSE("PS-003")
        self.network.register_evse(evse3, 240, 0)
        evse1.set_pilot = create_autospec(evse1.set_pilot)
        evse2.set_pilot = create_autospec(evse2.set_pilot)
        evse3.set_pilot = create_autospec(evse3.set_pilot)
        self.network.update_pilots(np.array([[24, 16], [16, 24], [0, 0]]), 0, 5)
        evse1.set_pilot.assert_any_call(24, 240, 5)
        evse2.set_pilot.assert_any_call(16, 240, 5)
        evse3.set_pilot.assert_any_call(0, 240, 5)

    def test_voltage_accessor(self):
        evse1 = EVSE("PS-001")
        evse2 = EVSE("PS-002")
        evse3 = EVSE("PS-003")
        self.network.register_evse(evse1, 240, -30)
        self.network.register_evse(evse3, 100, 150)
        self.network.register_evse(evse2, 140, 90)
        self.assertEqual(
            self.network.voltages, {"PS-001": 240, "PS-003": 100, "PS-002": 140}
        )

    def test_phase_angles_accessor(self):
        evse1 = EVSE("PS-001")
        evse2 = EVSE("PS-002")
        evse3 = EVSE("PS-003")
        self.network.register_evse(evse1, 240, -30)
        self.network.register_evse(evse3, 100, 150)
        self.network.register_evse(evse2, 140, 90)
        self.assertEqual(
            self.network.phase_angles, {"PS-001": -30, "PS-003": 150, "PS-002": 90}
        )


class TestChargingNetworkConstraints(TestCase):
    def setUp(self):
        self.network = ChargingNetwork()
        self.network.register_evse(EVSE("PS-001"), 240, 0)
        self.network.register_evse(EVSE("PS-004"), 240, 0)
        self.network.register_evse(EVSE("PS-003"), 240, 0)
        self.network.register_evse(EVSE("PS-002"), 240, 0)
        self.network.register_evse(EVSE("PS-006"), 240, 0)
        curr_dict1 = {"PS-001": 0.25, "PS-002": 0.50, "PS-003": -0.25}
        current1 = Current(curr_dict1)
        curr_dict2 = {"PS-006": 0.30, "PS-004": -0.60, "PS-002": 0.50}
        current2 = Current(curr_dict2)
        self.network.add_constraint(current1, 50, name="first_constraint")
        self.network.add_constraint(current2, 10)

    def test_constraints_as_df(self):
        constraint_frame = self.network.constraints_as_df()
        pd.testing.assert_frame_equal(
            constraint_frame,
            pd.DataFrame(
                data=[[0.25, 0.00, -0.25, 0.50, 0.00], [0.00, -0.60, 0.00, 0.50, 0.30]],
                columns=["PS-001", "PS-004", "PS-003", "PS-002", "PS-006"],
                index=["first_constraint", "_const_1"],
            ),
        )

    def test_add_constraint_unregistered_evse(self):
        curr_dict3 = {"PS-001": -0.25, "PS-005": -0.50, "PS-003": 0.25}
        current3 = Current(curr_dict3)
        with self.assertRaises(KeyError):
            self.network.add_constraint(current3, 25, name="_bad_const")

    def test_add_constraint(self):
        np.testing.assert_allclose(self.network.magnitudes, np.array([50, 10]))
        np.testing.assert_allclose(
            self.network.constraint_matrix,
            np.array(
                [[0.25, 0.00, -0.25, 0.50, 0.00], [0.00, -0.60, 0.00, 0.50, 0.30]]
            ),
        )
        self.assertEqual(self.network.constraint_index[0], "first_constraint")
        self.assertEqual(self.network.constraint_index[1], "_const_1")

    def test_add_constraint_warning(self):
        curr_dict3 = {"PS-001": -0.25, "PS-002": -0.50, "PS-003": 0.25}
        current3 = Current(curr_dict3)
        with self.assertWarns(UserWarning):
            self.network.add_constraint(current3, 25, name="_const_1")
        np.testing.assert_allclose(self.network.magnitudes, np.array([50, 10, 25]))
        np.testing.assert_allclose(
            self.network.constraint_matrix,
            np.array(
                [
                    [0.25, 0.00, -0.25, 0.50, 0.00],
                    [0.00, -0.60, 0.00, 0.50, 0.30],
                    [-0.25, 0.00, 0.25, -0.50, 0.00],
                ]
            ),
        )
        self.assertEqual(self.network.constraint_index[0], "first_constraint")
        self.assertEqual(self.network.constraint_index[1], "_const_1")
        self.assertEqual(self.network.constraint_index[2], "_const_1_v2")

    def test_remove_constraint_not_found(self):
        with self.assertRaises(KeyError):
            self.network.remove_constraint("_const_100")

    def test_remove_constraint(self):
        self.network.remove_constraint("first_constraint")
        self.assertEqual(self.network.constraint_index, ["_const_1"])
        np.testing.assert_allclose(self.network.magnitudes, np.array([10]))
        np.testing.assert_allclose(
            self.network.constraint_matrix, np.array([[0.00, -0.60, 0.00, 0.50, 0.30]])
        )

    def test_update_constraint_not_found(self):
        with self.assertRaises(KeyError):
            self.network.update_constraint("_const_100", Current(), 0)

    def test_update_constraint(self):
        curr_dict3 = {"PS-001": -0.25, "PS-002": -0.50, "PS-003": 0.25}
        current3 = Current(curr_dict3)
        self.network.update_constraint("first_constraint", current3, 25)
        self.assertEqual(
            self.network.constraint_index, ["_const_1", "first_constraint"]
        )
        np.testing.assert_allclose(self.network.magnitudes, np.array([10, 25]))
        np.testing.assert_allclose(
            self.network.constraint_matrix,
            np.array(
                [[0.00, -0.60, 0.00, 0.50, 0.30], [-0.25, 0.00, 0.25, -0.50, 0.00]]
            ),
        )

    def test_update_constraint_new_name(self):
        curr_dict3 = {"PS-001": -0.25, "PS-002": -0.50, "PS-003": 0.25}
        current3 = Current(curr_dict3)
        self.network.update_constraint(
            "first_constraint", current3, 25, new_name="negated_first"
        )
        self.assertEqual(self.network.constraint_index, ["_const_1", "negated_first"])
        np.testing.assert_allclose(self.network.magnitudes, np.array([10, 25]))
        np.testing.assert_allclose(
            self.network.constraint_matrix,
            np.array(
                [[0.00, -0.60, 0.00, 0.50, 0.30], [-0.25, 0.00, 0.25, -0.50, 0.00]]
            ),
        )

    def test_is_feasible_good_loads(self):
        good_loads = np.array([[0, 0], [150, 120], [100, 40], [150, 120], [60, 9]])
        self.assertTrue(self.network.is_feasible(good_loads))

    def test_is_feasible_bad_loads(self):
        bad_loads = np.array([[0, 0], [150, 800], [100, 0], [150, 20], [60, 9]])
        self.assertFalse(self.network.is_feasible(bad_loads))

    def test_is_feasible_barely_good_loads(self):
        good_loads = np.array(
            [[160, 200.000035], [0, 0], [0, 0], [20.000015, 0], [0, 0]]
        )
        self.assertTrue(self.network.is_feasible(good_loads))

    def test_is_feasible_barely_bad_loads(self):
        bad_loads = np.array(
            [[160, 200.000045], [0, 0], [0, 0], [20.000015, 0], [0, 0]]
        )
        self.assertFalse(self.network.is_feasible(bad_loads))

    def test_is_feasible_barely_good_loads_set(self):
        good_loads = np.array([[160, 200.0035], [0, 0], [0, 0], [20.0015, 0], [0, 0]])
        self.assertTrue(self.network.is_feasible(good_loads, violation_tolerance=1e-3))

    def test_is_feasible_barely_bad_loads_set(self):
        bad_loads = np.array([[160, 200.0045], [0, 0], [0, 0], [20.0015, 0], [0, 0]])
        self.assertFalse(self.network.is_feasible(bad_loads, violation_tolerance=1e-3))

    def test_is_feasible_barely_good_loads_rel_set(self):
        good_loads = np.array([[160, 200.15], [0, 0], [0, 0], [20.015, 0], [0, 0]])
        self.assertTrue(self.network.is_feasible(good_loads, relative_tolerance=1e-3))

    def test_is_feasible_barely_bad_loads_rel_set(self):
        bad_loads = np.array([[160, 200.21], [0, 0], [0, 0], [20.021, 0], [0, 0]])
        self.assertFalse(self.network.is_feasible(bad_loads, relative_tolerance=1e-3))

    def test_is_feasible_barely_good_loads_set_linear(self):
        good_loads = np.array([[160, 200.0035], [0, 0], [0, 0], [20.0015, 0], [0, 0]])
        self.assertTrue(
            self.network.is_feasible(good_loads, violation_tolerance=1e-3, linear=True)
        )

    def test_is_feasible_barely_bad_loads_set_linear(self):
        bad_loads = np.array([[160, 200.0045], [0, 0], [0, 0], [20.0015, 0], [0, 0]])
        self.assertFalse(
            self.network.is_feasible(bad_loads, violation_tolerance=1e-3, linear=True)
        )

    def test_constraint_current(self):
        loads = np.array([[0, 0], [150, 120], [100, 40], [150, 120], [60, 9]])
        np.testing.assert_allclose(
            self.network.constraint_current(loads),
            np.array([[50 + 0j, 50 + 0j], [3 + 0j, -9.3 + 0j]]),
        )
        np.testing.assert_allclose(
            self.network.constraint_current(
                loads, constraints=["_const_1"], time_indices=[1]
            ),
            np.array([[-9.3 + 0j]]),
        )
        np.testing.assert_allclose(
            self.network.constraint_current(loads, constraints=["first_constraint"]),
            np.array([[50 + 0j, 50 + 0j]]),
        )
        np.testing.assert_allclose(
            self.network.constraint_current(loads, time_indices=[1]),
            np.array([[50 + 0j], [-9.3 + 0j]]),
        )
