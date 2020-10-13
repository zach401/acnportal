from unittest import TestCase

import pandas as pd

from acnportal.acnsim import Current


class TestCurrent(TestCase):
    def setUp(self):
        pass

    def test_init_loads_dict_input(self):
        curr_dict = {"PS-001": 0.25, "PS-002": 0.50, "PS-003": -0.25}
        self.current = Current(curr_dict)
        pd.testing.assert_series_equal(
            self.current,
            pd.Series([0.25, 0.50, -0.25], index=["PS-001", "PS-002", "PS-003"]),
            check_series_type=False,
        )

    def test_init_str_load_input(self):
        curr_str = "PS-001"
        self.current = Current(curr_str)
        pd.testing.assert_series_equal(
            self.current, pd.Series([1], index=["PS-001"]), check_series_type=False
        )

    def test_init_none_input(self):
        self.current = Current()
        # The object type is specified explicitly here to address a
        # warning in pandas 1.x.
        pd.testing.assert_series_equal(
            self.current, pd.Series(dtype="float64"), check_series_type=False
        )

    def test_init_str_lst_input(self):
        curr_strs = ["PS-001", "PS-002", "PS-003"]
        self.current = Current(curr_strs)
        pd.testing.assert_series_equal(
            self.current,
            pd.Series([1, 1, 1], index=["PS-001", "PS-002", "PS-003"]),
            check_series_type=False,
        )

    def test_add_current_equal_station_ids(self):
        curr_dict1 = {"PS-001": 0.25, "PS-002": 0.50, "PS-003": -0.25}
        self.current1 = Current(curr_dict1)
        curr_dict2 = {"PS-001": 0.30, "PS-002": -0.60, "PS-003": 0.50}
        self.current2 = Current(curr_dict2)
        self.sum_curr = self.current1 + self.current2
        self.assertIsInstance(self.sum_curr, Current)
        pd.testing.assert_series_equal(
            self.sum_curr,
            pd.Series([0.55, -0.10, 0.25], index=["PS-001", "PS-002", "PS-003"]),
            check_series_type=False,
        )

    def test_add_current_unequal_station_ids(self):
        curr_dict1 = {"PS-001": 0.25, "PS-002": 0.50, "PS-003": -0.25}
        self.current1 = Current(curr_dict1)
        curr_dict2 = {"PS-006": 0.30, "PS-004": -0.60, "PS-002": 0.50}
        self.current2 = Current(curr_dict2)
        self.sum_curr = self.current1 + self.current2
        self.assertIsInstance(self.sum_curr, Current)
        pd.testing.assert_series_equal(
            self.sum_curr,
            pd.Series(
                [0.25, 1.00, -0.25, -0.60, 0.30],
                index=["PS-001", "PS-002", "PS-003", "PS-004", "PS-006"],
            ),
            check_series_type=False,
        )

    def test_sub_current_unequal_station_ids(self):
        curr_dict1 = {"PS-001": 0.25, "PS-002": 0.50, "PS-003": -0.25}
        self.current1 = Current(curr_dict1)
        curr_dict2 = {"PS-006": 0.30, "PS-004": -0.60, "PS-002": 0.50}
        self.current2 = Current(curr_dict2)
        self.diff_curr = self.current1 - self.current2
        self.assertIsInstance(self.diff_curr, Current)
        pd.testing.assert_series_equal(
            self.diff_curr,
            pd.Series(
                [0.25, 0.00, -0.25, 0.60, -0.30],
                index=["PS-001", "PS-002", "PS-003", "PS-004", "PS-006"],
            ),
            check_series_type=False,
        )

    def test_mul_current(self):
        curr_dict = {"PS-001": 0.25, "PS-002": 0.50, "PS-003": -0.25}
        self.current = Current(curr_dict)
        self.current *= 2
        self.assertIsInstance(self.current, Current)
        pd.testing.assert_series_equal(
            self.current,
            pd.Series([0.50, 1.00, -0.5], index=["PS-001", "PS-002", "PS-003"]),
            check_series_type=False,
        )
