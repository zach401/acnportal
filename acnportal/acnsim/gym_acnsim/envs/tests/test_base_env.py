# coding=utf-8
""" Tests for the base ACN-Sim gym environment. """
import unittest
from importlib.util import find_spec
from typing import Dict
from unittest.mock import create_autospec, Mock

import numpy as np

from .... import Simulator

if find_spec("gym") is not None:
    from ....interface import GymTrainedInterface, GymTrainingInterface
    from .. import BaseSimEnv


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestBaseSimEnv(unittest.TestCase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        self.mocked_simulator: Simulator = create_autospec(Simulator)
        self.mocked_simulator.__deepcopy__ = lambda x: x
        self.training_interface: GymTrainingInterface = \
            GymTrainingInterface(self.mocked_simulator)
        self.base_env: BaseSimEnv = BaseSimEnv(self.training_interface)

    def test_correct_on_init(self) -> None:
        self.assertEqual(self.base_env.interface, self.training_interface)
        self.assertNotEqual(self.base_env.prev_interface,
                            self.training_interface)
        self.assertNotEqual(self.base_env._init_snapshot,
                            self.training_interface)
        for attr in ['action', 'observation', 'reward', 'done', 'info']:
            self.assertIsNone(getattr(self.base_env, attr))
        self.assertEqual(self.base_env.schedule, {})

    def test_update_state(self) -> None:
        self.base_env.observation_from_state = Mock()
        self.base_env.reward_from_state = Mock()
        self.base_env.done_from_state = Mock()
        self.base_env.info_from_state = Mock()
        self.base_env.update_state()
        self.base_env.observation_from_state.assert_called_once()
        self.base_env.reward_from_state.assert_called_once()
        self.base_env.done_from_state.assert_called_once()
        self.base_env.info_from_state.assert_called_once()

    def test_store_previous_state(self) -> None:
        self.base_env.store_previous_state()
        self.assertEqual(self.base_env.interface, self.base_env.prev_interface)

    def test_step_with_training_interface(self) -> None:
        training_base_env: BaseSimEnv = BaseSimEnv(
            GymTrainedInterface(self.mocked_simulator))
        with self.assertRaises(TypeError):
            training_base_env.step(np.array([1, 2]))

    def test_step(self) -> None:
        dummy_schedule: Dict[str, float] = {'a': 1, 'b': 2}
        self.base_env.observation = np.eye(2)
        self.base_env.reward = 100.0
        self.base_env.done = False
        self.base_env.info = {'info': None}
        self.base_env.action_to_schedule = lambda: dummy_schedule
        self.base_env.store_previous_state = Mock()
        self.base_env.interface.step = Mock()
        self.base_env.update_state = Mock()
        observation, reward, done, info = self.base_env.step(np.array([1, 2]))
        np.testing.assert_equal(observation, np.eye(2))
        self.assertEqual(reward, 100.0)
        self.assertEqual(done, False)
        self.assertEqual(info, {'info': None})
        np.testing.assert_equal(self.base_env.action, np.array([1, 2]))
        self.assertEqual(self.base_env.schedule, dummy_schedule)
        self.base_env.store_previous_state.assert_called_once()
        self.training_interface.step.assert_called_with(dummy_schedule)
        self.base_env.update_state.assert_called_once()

    def test_reset(self) -> None:
        self.base_env.interface = GymTrainingInterface(self.mocked_simulator)
        self.base_env.prev_interface = GymTrainingInterface(self.mocked_simulator)
        self.base_env._init_snapshot = GymTrainingInterface(self.mocked_simulator)
        self.base_env.interface.tracking_value = 1
        self.base_env.prev_interface.tracking_value = 2
        self.base_env._init_snapshot.tracking_value = 3
        self.base_env.observation_from_state = lambda: np.eye(2)
        observation = self.base_env.reset()
        self.assertEqual(self.base_env.interface.tracking_value,
                         self.base_env._init_snapshot.tracking_value)
        self.assertEqual(self.base_env.prev_interface.tracking_value,
                         self.base_env._init_snapshot.tracking_value)
        np.testing.assert_equal(observation, np.eye(2))


if __name__ == '__main__':
    unittest.main()
