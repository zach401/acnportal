# coding=utf-8
""" Tests for the ACN-Sim gym algorithm and model wrapper. """
import unittest
from importlib.util import find_spec
from unittest.mock import create_autospec, Mock, call

import numpy as np


if find_spec("gym") is not None:
    from ...acnsim import Interface, GymTrainedInterface, \
        GymTrainingInterface, Simulator
    from ...acnsim.gym_acnsim.envs import BaseSimEnv
    from .. import SimRLModelWrapper, GymBaseAlgorithm, GymTrainedAlgorithm


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestSimRLModelWrapper(unittest.TestCase):
    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = object()
        cls.model_wrapper = SimRLModelWrapper(cls.model)

    def test_correct_on_init(self) -> None:
        self.assertEqual(self.model_wrapper.model, self.model)

    def test_predict_not_implemented_error(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.model_wrapper.predict(*4*[None])


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestGymBaseAlgorithm(unittest.TestCase):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        self.algorithm = GymBaseAlgorithm()
        self.env = create_autospec(BaseSimEnv)

    def test_correct_on_init(self) -> None:
        self.assertEqual(self.algorithm.max_recompute, 1)
        with self.assertRaises(ValueError):
            _ = self.algorithm.env

    def test_schedule_error(self) -> None:
        with self.assertRaises(NotImplementedError):
            _ = self.algorithm.schedule(None)

    def test_register_env(self) -> None:
        self.algorithm.register_env(self.env)
        self.assertEqual(self.algorithm.env, self.env)

    def test_register_interface_with_env(self) -> None:
        interface = Interface(create_autospec(Simulator))
        self.algorithm.register_env(self.env)
        self.algorithm.register_interface(interface)
        self.assertNotEqual(self.algorithm.interface, interface)
        self.assertIsInstance(self.algorithm.interface, GymTrainedInterface)
        self.assertEqual(self.env.interface, interface)

    def test_register_gym_interface(self) -> None:
        gym_interface = GymTrainedInterface(create_autospec(Simulator))
        self.algorithm.register_interface(gym_interface)
        self.assertEqual(self.algorithm.interface, gym_interface)

    def test_register_training_interface_error(self) -> None:
        training_interface = GymTrainingInterface(create_autospec(Simulator))
        with self.assertRaises(TypeError):
            self.algorithm.register_interface(training_interface)


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestGymTrainedAlgorithm(TestGymBaseAlgorithm):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.algorithm = GymTrainedAlgorithm()
        self.model = create_autospec(SimRLModelWrapper)
        self.interface = create_autospec(GymTrainedInterface)

    def test_schedule_error(self) -> None:
        # schedule() is implemented in GymTrainedAlgorithm.
        pass

    def test_correct_on_init(self) -> None:
        super().test_correct_on_init()
        with self.assertRaises(ValueError):
            _ = self.algorithm.model

    def test_model(self) -> None:
        self.model = create_autospec(SimRLModelWrapper)
        self.algorithm.register_model(self.model)
        self.assertEqual(self.algorithm.model, self.model)

    def test_schedule_error_no_model(self) -> None:
        self.algorithm.register_interface(self.interface)
        self.algorithm.register_env(self.env)
        with self.assertRaises(TypeError):
            self.algorithm.schedule(None)

    def test_schedule_error_no_env(self) -> None:
        self.algorithm.register_interface(self.interface)
        self.algorithm.register_model(self.model)
        with self.assertRaises(TypeError):
            self.algorithm.schedule(None)

    def test_schedule_error_no_interface(self):
        self.algorithm.register_interface(self.interface)
        self.algorithm.register_env(self.env)
        self.algorithm.register_model(self.model)
        self.env.interface = None
        with self.assertRaises(TypeError):
            self.algorithm.schedule(None)

    def test_schedule(self) -> None:
        self.env.interface = None
        self.algorithm.register_env(self.env)
        self.algorithm.register_interface(self.interface)
        self.algorithm.register_model(self.model)

        self.env.observation = np.eye(2)
        self.env.reward = 3
        self.env.done = True
        self.env.info = {'info': 'info'}

        self.env.update_state = Mock()
        self.env.store_previous_state = Mock()
        self.model.predict = Mock(return_value=np.ones((2,)))
        self.env.action_to_schedule = Mock(return_value={'PS-000': [0]})

        # As order of calls is important here, we wrap all mocks in a
        # parent.
        mock_parent = Mock()
        mock_parent.attach_mock(self.env.update_state, 'update_state')
        mock_parent.attach_mock(self.env.store_previous_state,
                                'store_previous_state')
        mock_parent.attach_mock(self.model.predict, 'predict')
        mock_parent.attach_mock(self.env.action_to_schedule,
                                'action_to_schedule')

        return_schedule = self.algorithm.schedule([])

        mock_parent.assert_has_calls([
            call.update_state(),
            call.store_previous_state(),
            call.predict(self.env.observation,
                         self.env.reward,
                         self.env.done,
                         self.env.info),
            call.action_to_schedule()
        ])

        np.testing.assert_equal(self.env.action, np.ones((2,)))
        self.assertEqual(self.env.schedule, {'PS-000': [0]})
        self.assertEqual(return_schedule, {'PS-000': [0]})


if __name__ == '__main__':
    unittest.main()
