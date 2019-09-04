from unittest import TestCase
from unittest.mock import Mock, create_autospec

from acnportal.acnsim.models import EV
from acnportal.acnsim.models import EVSE, InvalidRateError
from acnportal.acnsim import ChargingNetwork, InvalidScheduleError
from acnportal.acnsim import Current

import pandas as pd
import numpy as np


class TestChargingNetwork(TestCase):
    def setUp(self):
        self.network = ChargingNetwork()

    def test_init_empty(self):
        self.assertEqual(self.network._EVSEs, {})
        pd.testing.assert_series_equal(self.network.magnitudes, pd.Series())
        pd.testing.assert_frame_equal(self.network.constraints, pd.DataFrame())
        self.assertEqual(self.network._voltages, {})
        pd.testing.assert_series_equal(self.network._phase_angles, pd.Series())

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

    # noinspection PyUnresolvedReferences
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
        self.network.update_pilots(np.array([[24, 16], [16, 24], [0, 0]]), 0, 5)
        evse1.set_pilot.assert_any_call(24, 240, 5)
        evse2.set_pilot.assert_any_call(16, 240, 5)
        evse3.set_pilot.assert_any_call(0, 240, 5)

    def test_add_constraint(self):
        self.network.register_evse(EVSE('PS-001'), 240, 0)
        self.network.register_evse(EVSE('PS-002'), 240, 0)
        self.network.register_evse(EVSE('PS-003'), 240, 0)
        self.network.register_evse(EVSE('PS-004'), 240, 0)
        self.network.register_evse(EVSE('PS-006'), 240, 0)
        curr_dict1 = {'PS-001' : 0.25, 'PS-002' : 0.50, 'PS-003' : -0.25}
        current1 = Current(curr_dict1)
        curr_dict2 = {'PS-006' : 0.30, 'PS-004' : -0.60, 'PS-002' : 0.50}
        current2 = Current(curr_dict2)
        self.network.add_constraint(current1, 50)
        self.network.add_constraint(current2, 10)
        np.testing.assert_allclose(
            self.network.magnitudes, pd.Series(
                [50, 10], index=['_const_0', '_const_1']))
        pd.testing.assert_frame_equal(
            self.network.constraints, pd.DataFrame(
                np.array([[0.25, 0.50, -0.25, 0.00, 0.00],
                    [0.00, 0.50, 0.00, -0.60, 0.30]]),
                columns=['PS-001', 'PS-002', 'PS-003', 'PS-004', 'PS-006'],
                index=['_const_0', '_const_1']))
        np.testing.assert_allclose(self.network.magnitude_vector, np.array([50, 10]))
        np.testing.assert_allclose(
            self.network.constraint_matrix,
            np.array([[0.25, 0.50, -0.25, 0.00, 0.00],
                [0.00, 0.50, 0.00, -0.60, 0.30]]))
        self.assertEqual(self.network.constraint_index, {'_const_0' : 0, '_const_1' : 1})

    def test_is_feasible_good_loads(self):
        self.network.register_evse(EVSE('PS-001'), 240, 0)
        self.network.register_evse(EVSE('PS-002'), 240, 0)
        self.network.register_evse(EVSE('PS-003'), 240, 0)
        self.network.register_evse(EVSE('PS-004'), 240, 0)
        self.network.register_evse(EVSE('PS-006'), 240, 0)
        curr_dict1 = {'PS-001' : 0.25, 'PS-002' : 0.50, 'PS-003' : -0.25}
        current1 = Current(curr_dict1)
        curr_dict2 = {'PS-006' : 0.30, 'PS-004' : -0.60, 'PS-002' : 0.50}
        current2 = Current(curr_dict2)
        self.network.add_constraint(current1, 50)
        self.network.add_constraint(current2, 10)
        good_loads = {'PS-002' : [150, 120], 'PS-004' : [150, 120], 'PS-006' : [60, 9], 'PS-003' : [100, 40]}
        
        self.assertTrue(self.network.is_feasible(good_loads))

    def test_is_feasible_bad_loads(self):
        self.network.register_evse(EVSE('PS-001'), 240, 0)
        self.network.register_evse(EVSE('PS-002'), 240, 0)
        self.network.register_evse(EVSE('PS-003'), 240, 0)
        self.network.register_evse(EVSE('PS-004'), 240, 0)
        self.network.register_evse(EVSE('PS-006'), 240, 0)
        curr_dict1 = {'PS-001' : 0.25, 'PS-002' : 0.50, 'PS-003' : -0.25}
        current1 = Current(curr_dict1)
        curr_dict2 = {'PS-006' : 0.30, 'PS-004' : -0.60, 'PS-002' : 0.50}
        current2 = Current(curr_dict2)
        self.network.add_constraint(current1, 50)
        self.network.add_constraint(current2, 10)
        bad_loads = {'PS-002' : [150, 800], 'PS-004' : [150, 20], 'PS-006' : [60, 9], 'PS-003' : [100, 0]}
        
        self.assertFalse(self.network.is_feasible(bad_loads))

    def test_is_feasible_unequal_lengths(self):
        self.network.register_evse(EVSE('PS-001'), 240, 0)
        self.network.register_evse(EVSE('PS-002'), 240, 0)
        self.network.register_evse(EVSE('PS-003'), 240, 0)
        self.network.register_evse(EVSE('PS-004'), 240, 0)
        self.network.register_evse(EVSE('PS-006'), 240, 0)
        curr_dict1 = {'PS-001' : 0.25, 'PS-002' : 0.50, 'PS-003' : -0.25}
        current1 = Current(curr_dict1)
        curr_dict2 = {'PS-006' : 0.30, 'PS-004' : -0.60, 'PS-002' : 0.50}
        current2 = Current(curr_dict2)
        self.network.add_constraint(current1, 50)
        self.network.add_constraint(current2, 10)
        unequal_loads = {'PS-002' : [150, 800], 'PS-004' : [150], 'PS-006' : [60, 9], 'PS-003' : [100, 0]}
        with self.assertRaises(InvalidScheduleError):
            self.network.is_feasible(unequal_loads)

    def test_constraint_current(self):
        self.network.register_evse(EVSE('PS-001'), 240, 0)
        self.network.register_evse(EVSE('PS-002'), 240, 0)
        self.network.register_evse(EVSE('PS-003'), 240, 0)
        self.network.register_evse(EVSE('PS-004'), 240, 0)
        self.network.register_evse(EVSE('PS-006'), 240, 0)
        curr_dict1 = {'PS-001' : 0.25, 'PS-002' : 0.50, 'PS-003' : -0.25}
        current1 = Current(curr_dict1)
        curr_dict2 = {'PS-006' : 0.30, 'PS-004' : -0.60, 'PS-002' : 0.50}
        current2 = Current(curr_dict2)
        self.network.add_constraint(current1, 50)
        self.network.add_constraint(current2, 10)
        loads = {'PS-002' : [150, 120], 'PS-004' : [150, 120], 'PS-006' : [60, 9], 'PS-003' : [100, 40]}

        np.testing.assert_allclose(self.network.constraint_current(loads),
            np.array([[50+0j, 50+0j], [3+0j, -9.3+0j]]))
        np.testing.assert_allclose(self.network.constraint_current(loads, constraints=['_const_1'], time_indices=[1]),
            np.array([[-9.3+0j]]))
        np.testing.assert_allclose(self.network.constraint_current(loads, constraints=['_const_0']),
            np.array([[50+0j, 50+0j]]))
        np.testing.assert_allclose(self.network.constraint_current(loads, time_indices=[1]),
            np.array([[50+0j], [-9.3+0j]]))

