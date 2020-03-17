# coding=utf-8
""" Tests for reward functions. """
import unittest
from typing import Callable, Dict, List
from unittest.mock import create_autospec
from importlib.util import find_spec

import numpy as np

from .... import Simulator, ChargingNetwork, EVSE

if find_spec("gym") is not None:
    from gym import spaces
    from ..action_spaces import SimAction
    from ....interface import GymTrainingInterface, GymTrainedInterface
    from .. import reward_functions as rf, CustomSimEnv


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestRewardFunction(unittest.TestCase):
    simulator: Simulator

    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        self.simulator: Simulator = create_autospec(Simulator)
        self.interface: GymTrainingInterface = GymTrainingInterface(
            self.simulator)
        # Set mock's deepcopy to return input (a dict of already copied
        # attributes).
        self.simulator.__deepcopy__ = lambda x: x
        self.sim_action_space_function: \
            Callable[[GymTrainedInterface], spaces.Space] = \
            lambda x: spaces.Space()
        self.sim_action_function: \
            Callable[[GymTrainedInterface, np.ndarray],
                     Dict[str, List[float]]] = \
            lambda x, y: {'a': [0]}
        self.env: CustomSimEnv = CustomSimEnv(
            self.interface,
            [],
            SimAction(self.sim_action_space_function,
                      self.sim_action_function,
                      'stub_action'),
            []
        )
        self.simulator.network = create_autospec(ChargingNetwork)
        self.simulator.network.station_ids = ['TS-001', 'TS-002', 'TS-003']


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestEVSEViolation(TestRewardFunction):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()

    def test_evse_violation_key_error(self) -> None:
        self.env.schedule = {'TS-001': [0], 'TS-002': [0]}
        self.simulator.network.station_ids = ['TS-001']
        with self.assertRaises(KeyError):
            _ = rf.evse_violation(self.env)

    def _continuous_evse_helper(self) -> None:
        self.evse1 = create_autospec(EVSE)
        self.evse1.is_continuous = True
        self.evse1.allowable_pilot_signals = [0, 32]

        self.evse2 = create_autospec(EVSE)
        self.evse2.is_continuous = True
        self.evse2.allowable_pilot_signals = [6, 16]

        self.evse3 = create_autospec(EVSE)
        self.evse3.is_continuous = True
        self.evse3.allowable_pilot_signals = [6, 32]

        self.simulator.network._EVSEs = {'TS-001': self.evse1,
                                         'TS-002': self.evse2,
                                         'TS-003': self.evse3}

    def test_evse_violation_continuous_violation(self) -> None:
        self._continuous_evse_helper()
        self.env.schedule = {'TS-001': [34, 31],
                             'TS-002': [4, 5],
                             'TS-003': [0, 0]}
        self.assertEqual(rf.evse_violation(self.env), -5)

    def test_evse_violation_continuous_no_violation(self) -> None:
        self._continuous_evse_helper()
        self.env.schedule = {'TS-001': [31, 16],
                             'TS-002': [7, 16],
                             'TS-003': [0, 0]}
        self.assertEqual(rf.evse_violation(self.env), 0)

    def _discrete_evse_helper(self) -> None:
        self.evse1 = create_autospec(EVSE)
        self.evse1.is_continuous = False
        self.evse1.allowable_pilot_signals = [8, 16, 24, 32]

        self.evse2 = create_autospec(EVSE)
        self.evse2.is_continuous = False
        self.evse2.allowable_pilot_signals = [6, 16]

        self.evse3 = create_autospec(EVSE)
        self.evse3.is_continuous = False
        self.evse3.allowable_pilot_signals = list(range(1, 32))

        self.simulator.network._EVSEs = {'TS-001': self.evse1,
                                         'TS-002': self.evse2,
                                         'TS-003': self.evse3}

    def test_evse_violation_non_continuous_violation(self) -> None:
        self._discrete_evse_helper()
        self.env.schedule = {'TS-001': [4, 19],
                             'TS-002': [8, 18],
                             'TS-003': [0, 0]}
        self.assertEqual(rf.evse_violation(self.env), -11)

    def test_evse_violation_non_continuous_no_violation(self) -> None:
        self._discrete_evse_helper()
        self.env.schedule = {'TS-001': [8, 24],
                             'TS-002': [6, 16],
                             'TS-003': [0, 0]}
        self.assertEqual(rf.evse_violation(self.env), 0)


class TestUnpluggedEVViolation(TestRewardFunction):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.env.schedule = {'TS-001': [8, 24],
                             'TS-002': [6, 16]}

    def test_unplugged_ev_violation_empty_schedules(self) -> None:
        self.env.schedule = {'TS-001': [], 'TS-002': []}
        self.assertEqual(rf.unplugged_ev_violation(self.env), 0)

    def test_unplugged_ev_violation_all_unplugged(self) -> None:
        self.simulator.network.active_station_ids = ['TS-003']
        self.assertEqual(rf.unplugged_ev_violation(self.env), -14)

    def test_unplugged_ev_violation_some_unplugged(self) -> None:
        self.simulator.network.active_station_ids = ['TS-002', 'TS-003']
        self.assertEqual(rf.unplugged_ev_violation(self.env), -8)

    def test_unplugged_ev_violation_none_unplugged(self) -> None:
        self.simulator.network.active_station_ids = [
            'TS-001', 'TS-002', 'TS-003']
        self.assertEqual(rf.unplugged_ev_violation(self.env), 0)


class TestConstraintViolation(TestRewardFunction):
    # noinspection PyMissingOrEmptyDocstring
    def setUp(self) -> None:
        super().setUp()
        self.simulator.network.constraint_matrix = np.array([[0, 1, 1],
                                                             [1, 0, 1],
                                                             [1, 1, 0]])
        self.simulator.network.magnitudes = np.array([32, 32, 32])
        self.simulator.network.constraint_index = ['C1', 'C2', 'C3']

    def test_constraint_violation_with_violation(self) -> None:
        pass

    def test_constraint_violation_no_violation(self) -> None:
        pass


if __name__ == '__main__':
    unittest.main()
