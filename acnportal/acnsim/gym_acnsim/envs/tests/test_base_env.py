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
        self.env.observation_from_state = Mock()
        self.env.reward_from_state = Mock()
        self.env.done_from_state = Mock()
        self.env.info_from_state = Mock()

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

        self.env.store_previous_state = Mock()
        self.env.interface.step = Mock()
        self.env.update_state = Mock()

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


if __name__ == '__main__':
    unittest.main()
