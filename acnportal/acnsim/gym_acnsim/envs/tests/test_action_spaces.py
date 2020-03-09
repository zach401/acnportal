# coding=utf-8
""" Tests for SimAction and action space functions. """
import unittest
from importlib.util import find_spec
from typing import Callable, Dict, List, Any
from unittest.mock import create_autospec

import numpy as np

if find_spec("gym") is not None:
    from gym import Space
    from ..action_spaces import SimAction, single_charging_schedule, \
        zero_centered_single_charging_schedule
from ....interface import GymTrainedInterface


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestSimAction(unittest.TestCase):
    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        # The type here is Any as space_function is actually a Mock
        # object, but there's no Mock type in the typing library.
        cls.space_function: Any = create_autospec(lambda interface: Space())
        cls.to_schedule: Callable[[GymTrainedInterface, np.ndarray],
                                  Dict[str, List[float]]] = \
            lambda interface, array: {'a': [0]}
        cls.name: str = "stub_action"
        cls.sim_action: SimAction = SimAction(
            cls.space_function, cls.to_schedule, cls.name)
        cls.interface: GymTrainedInterface = create_autospec(
            GymTrainedInterface)

    def test_correct_on_init_sim_action_name(self) -> None:
        self.assertEqual(self.sim_action.name, self.name)

    def test_get_space(self) -> None:
        self.sim_action.get_space(self.interface)
        self.space_function.assert_called_once()

    def test_get_schedule(self) -> None:
        array: np.ndarray = np.array([[1, 0], [0, 1]])
        self.assertEqual(self.sim_action.get_schedule(self.interface, array),
                         {'a': [0]})


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestSingleChargingSchedule(unittest.TestCase):
    # Some class variables are defined outside of setUpClass so that
    # the code inspector knows that inherited classes have these
    # attributes.
    max_rate: float = 16.
    min_rate: float = 0.
    negative_rate: float = -4.
    deadband_rate: float = 6.

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim_action: SimAction = single_charging_schedule()
        cls.station_ids: List[str] = ['T1', 'T2']
        cls.offset: float = 0.5

        def _interface_builder(interface: Any, min_rate: float) -> Any:
            interface.station_ids = cls.station_ids
            interface.max_pilot_signal = lambda station_id: cls.max_rate
            interface.min_pilot_signal = lambda station_id: (
                min_rate
                if station_id == cls.station_ids[1] else cls.min_rate)
            return interface

        cls.interface: Any = _interface_builder(
            create_autospec(GymTrainedInterface), cls.min_rate)
        cls.interface_negative_min: Any = _interface_builder(
            create_autospec(GymTrainedInterface), cls.negative_rate)
        cls.interface_deadband_min: Any = _interface_builder(
            create_autospec(GymTrainedInterface), cls.deadband_rate)

    def test_correct_on_init_single_name(self) -> None:
        self.assertEqual(self.sim_action.name, 'single schedule')

    def _test_space_function_helper(self,
                                    interface: GymTrainedInterface,
                                    min_rate: float,
                                    max_rate: float) -> None:
        out_space: Space = self.sim_action.get_space(interface)
        self.assertEqual(out_space.shape, (len(self.station_ids),))
        np.testing.assert_equal(out_space.low, 2 * [min_rate])
        np.testing.assert_equal(out_space.high, 2 * [max_rate])
        self.assertEqual(out_space.dtype, 'float')

    def test_single_space_function(self) -> None:
        self._test_space_function_helper(self.interface,
                                         self.min_rate,
                                         self.max_rate)

    def test_single_space_function_negative_min(self) -> None:
        self._test_space_function_helper(self.interface_negative_min,
                                         self.negative_rate,
                                         self.max_rate)

    def test_single_space_function_deadband_min(self) -> None:
        self._test_space_function_helper(self.interface_deadband_min,
                                         self.min_rate,
                                         self.max_rate)

    def test_single_to_schedule(self) -> None:
        good_schedule: Dict[str, List[float]] = self.sim_action.get_schedule(
            self.interface,
            np.array([self.min_rate + self.offset,
                      (self.max_rate - self.min_rate) / 2])
        )
        self.assertEqual(good_schedule, {
            self.station_ids[0]: [self.min_rate + self.offset],
            self.station_ids[1]: [(self.max_rate - self.min_rate) / 2]
        })

    def test_single_to_bad_schedule(self) -> None:
        # The get_schedule function does not test if the input schedule
        # array is within the action space.
        bad_schedule: Dict[str, List[float]] = self.sim_action.get_schedule(
            self.interface,
            np.array([self.min_rate - self.offset,
                      self.max_rate + self.offset])
        )
        self.assertEqual(bad_schedule, {
            self.station_ids[0]: [self.min_rate - self.offset],
            self.station_ids[1]: [self.max_rate + self.offset]
        })

    def test_single_error_schedule(self) -> None:
        with self.assertRaises(TypeError):
            _ = self.sim_action.get_schedule(
                self.interface,
                np.array([[self.min_rate - self.offset],
                          [self.max_rate + self.offset]])
            )


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestZeroCenteredSingleChargingSchedule(TestSingleChargingSchedule):
    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.sim_action: SimAction = zero_centered_single_charging_schedule()
        cls.shifted_max = cls.max_rate - (cls.max_rate
                                          + cls.min_rate) / 2
        cls.shifted_minimums = [
            cls.min_rate - (cls.max_rate + cls.min_rate) / 2,
            cls.negative_rate - (cls.max_rate + cls.negative_rate) / 2,
            cls.min_rate - (cls.max_rate + cls.deadband_rate) / 2
        ]
        cls.negative_max_shift = cls.max_rate - (cls.max_rate
                                                 + cls.negative_rate) / 2

    def test_correct_on_init_single_name(self) -> None:
        self.assertEqual(self.sim_action.name, 'zero-centered single schedule')

    def test_single_space_function(self) -> None:
        self._test_space_function_helper(self.interface,
                                         self.shifted_minimums[0],
                                         self.shifted_max)

    def test_single_space_function_negative_min(self) -> None:
        self._test_space_function_helper(self.interface_negative_min,
                                         self.shifted_minimums[1],
                                         self.negative_max_shift)

    def test_single_space_function_deadband_min(self) -> None:
        self._test_space_function_helper(self.interface_deadband_min,
                                         self.shifted_minimums[2],
                                         self.shifted_max)

    def test_single_to_bad_schedule(self) -> None:
        # The get_schedule function does not test if the input schedule
        # array is within the action space.
        bad_schedule: Dict[str, List[float]] = self.sim_action.get_schedule(
            self.interface,
            np.array([self.min_rate - self.offset,
                      self.max_rate + self.offset])
        )
        self.assertEqual(bad_schedule, {
            self.station_ids[0]: [self.min_rate - self.offset
                                  + (self.max_rate + self.min_rate) / 2],
            self.station_ids[1]: [self.max_rate + self.offset
                                  + (self.max_rate + self.min_rate) / 2]
        })

    def test_single_to_schedule(self) -> None:
        good_schedule: Dict[str, List[float]] = self.sim_action.get_schedule(
            self.interface,
            np.array([self.min_rate - (self.max_rate + self.min_rate) / 2,
                      self.max_rate - (self.max_rate + self.min_rate) / 2])
        )
        self.assertEqual(good_schedule, {
            self.station_ids[0]: [self.min_rate],
            self.station_ids[1]: [self.max_rate]
        })


if __name__ == '__main__':
    unittest.main()
