# coding=utf-8
""" Tests for the base ACN-Sim gym environment. """
import unittest
from importlib.util import find_spec
from typing import Dict, Callable
from unittest.mock import create_autospec, Mock

import numpy as np

from .... import Simulator

if find_spec("gym") is not None:
    from gym import Space
    from ..action_spaces import SimAction
    from ..observation import SimObservation
    from ....interface import GymTrainedInterface, GymTrainingInterface
    from .. import BaseSimEnv, CustomSimEnv, RebuildingEnv


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestBaseSimEnv(unittest.TestCase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        self.mocked_simulator: Simulator = create_autospec(Simulator)
        self.mocked_simulator.__deepcopy__ = lambda x: x
        self.training_interface: GymTrainingInterface = \
            GymTrainingInterface(self.mocked_simulator)
        self.env: BaseSimEnv = BaseSimEnv(self.training_interface)

    def test_correct_on_init(self) -> None:
        self.assertEqual(self.env.interface, self.training_interface)
        self.assertNotEqual(self.env.prev_interface,
                            self.training_interface)
        self.assertNotEqual(self.env._init_snapshot,
                            self.training_interface)

        for attr in ['action', 'observation', 'reward', 'done', 'info']:
            self.assertIsNone(getattr(self.env, attr))

        self.assertEqual(self.env.schedule, {})

    def test_update_state(self) -> None:
        self.env.observation_from_state = create_autospec(
            self.env.observation_from_state)
        self.env.reward_from_state = create_autospec(
            self.env.reward_from_state)
        self.env.done_from_state = create_autospec(self.env.done_from_state)
        self.env.info_from_state = create_autospec(self.env.info_from_state)

        self.env.update_state()

        self.env.observation_from_state.assert_called_once()
        self.env.reward_from_state.assert_called_once()
        self.env.done_from_state.assert_called_once()
        self.env.info_from_state.assert_called_once()

    def test_store_previous_state(self) -> None:
        self.env.store_previous_state()
        self.assertEqual(self.env.interface, self.env.prev_interface)

    def test_step_with_training_interface(self) -> None:
        training_base_env: BaseSimEnv = BaseSimEnv(
            GymTrainedInterface(self.mocked_simulator))
        with self.assertRaises(TypeError):
            training_base_env.step(np.array([1, 2]))

    def test_step(self) -> None:
        dummy_schedule: Dict[str, float] = {'a': 1, 'b': 2}
        self.env.observation = np.eye(2)
        self.env.reward = 100.0
        self.env.done = False
        self.env.info = {'info': None}
        self.env.action_to_schedule = lambda: dummy_schedule

        self.env.store_previous_state = create_autospec(
            self.env.store_previous_state)
        self.env.interface.step = create_autospec(self.env.interface.step)
        self.env.update_state = create_autospec(self.env.update_state)

        observation, reward, done, info = self.env.step(np.array([1, 2]))

        np.testing.assert_equal(observation, np.eye(2))
        self.assertEqual(reward, 100.0)
        self.assertEqual(done, False)
        self.assertEqual(info, {'info': None})
        np.testing.assert_equal(self.env.action, np.array([1, 2]))
        self.assertEqual(self.env.schedule, dummy_schedule)

        self.env.store_previous_state.assert_called_once()
        self.training_interface.step.assert_called_with(dummy_schedule)
        self.env.update_state.assert_called_once()

    def test_reset(self) -> None:
        self.env.interface = GymTrainingInterface(self.mocked_simulator)
        self.env.prev_interface = GymTrainingInterface(self.mocked_simulator)
        self.env._init_snapshot = GymTrainingInterface(self.mocked_simulator)

        # Assign tracking values to each interface so we can keep track
        # of which interface we're looking at without forcing the memory
        # addresses to be the same.
        self.env.interface.tracking_value = 1
        self.env.prev_interface.tracking_value = 2
        self.env._init_snapshot.tracking_value = 3

        self.env.observation_from_state = lambda: np.eye(2)

        observation = self.env.reset()

        self.assertEqual(self.env.interface.tracking_value,
                         self.env._init_snapshot.tracking_value)
        self.assertEqual(self.env.prev_interface.tracking_value,
                         self.env._init_snapshot.tracking_value)
        np.testing.assert_equal(observation, np.eye(2))


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
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


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
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


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
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
