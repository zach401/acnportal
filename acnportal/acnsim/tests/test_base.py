import json
import unittest

import numpy as np

from acnportal.acnsim import ChargingNetwork
from acnportal.acnsim.base import ErrorAllWrapper, NpEncoder


class TestErrorAllWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builtin_data = 2
        cls.builtin_wrapper = ErrorAllWrapper(cls.builtin_data)
        cls.network_data = ChargingNetwork()
        cls.network_wrapper = ErrorAllWrapper(cls.network_data)

    def test_correct_on_init(self):
        self.assertEqual(self.builtin_data, self.builtin_wrapper.data)
        self.assertEqual(self.network_data, self.network_wrapper.data)

    def test_builtin_success(self):
        self.assertEqual(self.builtin_wrapper.__dict__, {"_data": 2})

    def test_builtin_error(self):
        with self.assertRaises(TypeError):
            _ = self.builtin_wrapper == self.network_wrapper

    def test_network_attribute_error(self):
        with self.assertRaises(TypeError):
            _ = self.network_wrapper.constraint_matrix

    def test_network_function_error(self):
        with self.assertRaises(TypeError):
            _ = self.network_wrapper.is_feasible(np.array([[0]]))


class TestNpEncoder(unittest.TestCase):
    def test_integer_dump(self):
        test_uint32 = np.uint32(2)
        self.assertEqual(json.dumps(test_uint32, cls=NpEncoder), "2")

    def test_floating_dump(self):
        test_float32 = np.float32(2)
        self.assertEqual(json.dumps(test_float32, cls=NpEncoder), "2.0")

    def test_ndarray_dump(self):
        test_ndarray = np.array([1, 2])
        self.assertEqual(json.dumps(test_ndarray, cls=NpEncoder), "[1, 2]")


if __name__ == "__main__":
    unittest.main()
