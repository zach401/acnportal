from unittest import TestCase
from unittest.mock import Mock, create_autospec

from acnportal.acnsim.models import EV
from acnportal.acnsim.models import EVSE, InvalidRateError
from acnportal.acnsim import ChargingNetwork, Current, ConstraintSet

import pandas as pd
import numpy as np

class TestConstraintSet(TestCase):
    def setUp(self):
        self.constraint_set = ConstraintSet()

    def test_init_empty(self):
        pd.testing.assert_series_equal(self.constraint_set.magnitudes, pd.Series())
        pd.testing.assert_frame_equal(self.constraint_set.constraints, pd.DataFrame())

    # TODO: test init with a constraint matrix; will require new error class

    def test_add_constraint(self):
        curr_dict1 = {'PS-001' : 0.25, 'PS-002' : 0.50, 'PS-003' : -0.25}
        current1 = Current(curr_dict1)
        curr_dict2 = {'PS-006' : 0.30, 'PS-004' : -0.60, 'PS-002' : 0.50}
        current2 = Current(curr_dict2)
        self.constraint_set.add_constraint(current1, 50)
        self.constraint_set.add_constraint(current2, 10)
        pd.testing.assert_series_equal(
            self.constraint_set.magnitudes, pd.Series(
                [50, 10], index=['_const_0', '_const_1']))
        pd.testing.assert_frame_equal(
            self.constraint_set.constraints, pd.DataFrame(
                np.array([[0.25, 0.50, -0.25, 0.00, 0.00],
                    [0.00, 0.50, 0.00, -0.60, 0.30]]),
                columns=['PS-001', 'PS-002', 'PS-003', 'PS-004', 'PS-006'],
                index=['_const_0', '_const_1']))


class TestCurrent(TestCase):
    def setUp(self):
        pass

    def test_init_loads_dict_input(self):
        curr_dict = {'PS-001' : 0.25, 'PS-002' : 0.50, 'PS-003' : -0.25}
        self.current = Current(curr_dict)
        pd.testing.assert_series_equal(
            self.current.loads, pd.Series(
                [0.25, 0.50, -0.25], index=['PS-001', 'PS-002', 'PS-003']))

    def test_init_str_load_input(self):
        curr_str = 'PS-001'
        self.current = Current(curr_str)
        pd.testing.assert_series_equal(
            self.current.loads, pd.Series(
                [1], index=['PS-001']))

    def test_init_none_input(self):
        self.current = Current()
        pd.testing.assert_series_equal(self.current.loads, pd.Series())

    def test_init_str_lst_input(self):
        curr_strs = ['PS-001', 'PS-002', 'PS-003']
        self.current = Current(curr_strs)
        pd.testing.assert_series_equal(
            self.current.loads, pd.Series(
                [1, 1, 1], index=['PS-001', 'PS-002', 'PS-003']))

    def test_add_current_equal_station_ids(self):
        curr_dict1 = {'PS-001' : 0.25, 'PS-002' : 0.50, 'PS-003' : -0.25}
        self.current1 = Current(curr_dict1)
        curr_dict2 = {'PS-001' : 0.30, 'PS-002' : -0.60, 'PS-003' : 0.50}
        self.current2 = Current(curr_dict2)
        self.sum_curr = self.current1 + self.current2
        pd.testing.assert_series_equal(
            self.sum_curr.loads, pd.Series(
                [0.55, -0.10, 0.25], index=['PS-001', 'PS-002', 'PS-003']))

    def test_add_current_unequal_station_ids(self):
        curr_dict1 = {'PS-001' : 0.25, 'PS-002' : 0.50, 'PS-003' : -0.25}
        self.current1 = Current(curr_dict1)
        curr_dict2 = {'PS-006' : 0.30, 'PS-004' : -0.60, 'PS-002' : 0.50}
        self.current2 = Current(curr_dict2)
        self.sum_curr = self.current1 + self.current2
        pd.testing.assert_series_equal(
            self.sum_curr.loads, pd.Series(
                [0.25, 1.00, -0.25, -0.60, 0.30], index=['PS-001', 'PS-002', 'PS-003', 'PS-004', 'PS-006']))

    def test_mul_current(self):
        curr_dict = {'PS-001' : 0.25, 'PS-002' : 0.50, 'PS-003' : -0.25}
        self.current = Current(curr_dict)
        self.current *= 2
        pd.testing.assert_series_equal(
            self.current.loads, pd.Series(
                [0.50, 1.00, -0.5], index=['PS-001', 'PS-002', 'PS-003']))