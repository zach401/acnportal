# coding=utf-8
""" Tests for custom ACN-Sim gym environment. """
import unittest
from importlib.util import find_spec
from typing import Callable
from unittest.mock import create_autospec, Mock

import numpy as np

if find_spec("gym") is not None:
    from .test_base_env import TestBaseSimEnv
    from gym import Space
    from ..action_spaces import SimAction
    from ..observation import SimObservation
    from .. import BaseSimEnv, CustomSimEnv, RebuildingEnv


# def sim_action_stub() -> SimAction:
#     """ Generates a SimAction instance that wraps functions to handle
#     actions taking the form of a vector of pilot signals. For this
#     action type, a single entry represents the pilot signal sent to
#     single EVSE at the current timestep. The space bounds
#     pilot signals above the maximum allowable rate over all EVSEs and
#     below the minimum allowable rate over all EVSEs.
#
#     As a 0 min rate is assumed to be allowed, the action space lower
#     bound is set to 0 if the station min rates are all greater than 0.
#     """
#     def space_function(interface: GymTrainedInterface) -> Space:


class TestCustomSimEnv(TestBaseSimEnv):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()

        self.action_object: SimAction = create_autospec(SimAction)
        self.action_object.get_space = Mock()
        self.action_object.get_space.return_value = Space(shape=(7, 13))
        self.action_object.get_schedule = Mock()
        self.action_object.get_schedule.return_value = {'a': 1, 'b': 2}

        self.observation_object1: SimObservation = create_autospec(
            SimObservation)
        self.observation_object1.name = 'dummy_obs_1'
        self.observation_object1.get_space = Mock()
        self.observation_object1.get_space.return_value = Space(shape=(13, 17))
        self.observation_object1.get_obs = Mock()
        self.observation_object1.get_obs.return_value = np.eye(3)

        self.observation_object2: SimObservation = create_autospec(
            SimObservation)
        self.observation_object2.name = 'dummy_obs_2'
        self.observation_object2.get_space = Mock()
        self.observation_object2.get_space.return_value = Space(shape=(17, 19))
        self.observation_object2.get_obs = Mock()
        self.observation_object2.get_obs.return_value = np.eye(4)

        self.reward_function1: Callable[[BaseSimEnv], float] = lambda env: 42
        self.reward_function2: Callable[[BaseSimEnv], float] = lambda env: 1337

        self.env: CustomSimEnv = CustomSimEnv(
            self.training_interface,
            [self.observation_object1, self.observation_object2],
            self.action_object,
            [self.reward_function1, self.reward_function2]
        )

    def test_correct_on_init(self) -> None:
        super().test_correct_on_init()
        self.assertEqual(self.env.observation_objects,
                         [self.observation_object1, self.observation_object2])

        spaces_dict = self.env.observation_space.spaces
        self.assertEqual(spaces_dict['dummy_obs_1'].shape, (13, 17))
        self.assertEqual(spaces_dict['dummy_obs_2'].shape, (17, 19))
        self.assertIsNone(spaces_dict['dummy_obs_1'].dtype)
        self.assertIsNone(spaces_dict['dummy_obs_2'].dtype)

        self.assertEqual(self.env.action_object, self.action_object)
        self.assertEqual(self.env.action_space.shape, (7, 13))
        self.assertIsNone(self.env.action_space.dtype)

        self.assertEqual(self.env.reward_functions,
                         [self.reward_function1, self.reward_function2])

    def test_action_to_schedule(self) -> None:
        self.env.action = np.eye(2)
        self.out_schedule = self.env.action_to_schedule()
        interface, action = self.action_object.get_schedule.call_args[0]
        self.assertEqual(interface, self.training_interface)
        np.testing.assert_equal(action, np.eye(2))

    def test_observation_from_state(self) -> None:
        observation = self.env.observation_from_state()
        np.testing.assert_equal(observation[self.observation_object1.name],
                                np.eye(3))
        np.testing.assert_equal(observation[self.observation_object2.name],
                                np.eye(4))

    def test_reward_from_state(self) -> None:
        self.assertEqual(self.env.reward_from_state(), 42 + 1337)


class TestRebuildingEnvNoGenFunc(TestCustomSimEnv):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()

        self.env: RebuildingEnv = RebuildingEnv(
            self.training_interface,
            [self.observation_object1, self.observation_object2],
            self.action_object,
            [self.reward_function1, self.reward_function2]
        )

    def test_double_none_error(self) -> None:
        with self.assertRaises(TypeError):
            self.env: RebuildingEnv = RebuildingEnv(
                None,
                [self.observation_object1, self.observation_object2],
                self.action_object,
                [self.reward_function1, self.reward_function2]
            )


class TestRebuildingEnv(TestCustomSimEnv):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()

        self.interface_generating_function = Mock()
        self.interface_generating_function.return_value = \
            self.training_interface

        self.env: RebuildingEnv = RebuildingEnv(
            None,
            [self.observation_object1, self.observation_object2],
            self.action_object,
            [self.reward_function1, self.reward_function2],
            self.interface_generating_function
        )

    def test_reset(self) -> None:
        self.interface_generating_function.reset_mock()

        # Assign tracking values to each interface so we can keep track
        # of which interface we're looking at without forcing the memory
        # addresses to be the same.
        self.env.interface.tracking_value = 1
        self.env.prev_interface.tracking_value = 2
        self.env._init_snapshot.tracking_value = 3
        self.training_interface.tracking_value = 4

        self.env.observation_from_state = lambda: np.eye(2)

        observation = self.env.reset()

        self.interface_generating_function.assert_called_once()
        self.assertEqual(self.env.interface.tracking_value,
                         self.training_interface.tracking_value)
        self.assertEqual(self.env.prev_interface.tracking_value,
                         self.training_interface.tracking_value)
        self.assertEqual(self.env._init_snapshot.tracking_value,
                         self.training_interface.tracking_value)
        np.testing.assert_equal(observation, np.eye(2))


if __name__ == '__main__':
    unittest.main()
