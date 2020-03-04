import unittest
from typing import Callable, Dict, List
from unittest.mock import Mock, create_autospec

import numpy
from importlib.util import find_spec
if find_spec("gym") is not None:
    from gym import Space
    from ..action_spaces import SimAction
from ....interface import GymTrainedInterface


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestSimAction(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.space_function = create_autospec(lambda interface: Space())
        cls.to_schedule: Callable[[GymTrainedInterface, numpy.ndarray],
                                  Dict[str, List[float]]] = \
            lambda interface, array: {'a': [0]}
        cls.name = "stub_action"
        cls.sim_action = SimAction(
            cls.space_function, cls.to_schedule, cls.name)
        cls.interface = Mock(GymTrainedInterface)

    def test_correct_on_init_sim_action(self):
        self.assertEqual(self.sim_action.name, self.name)

    def test_get_space(self):
        self.sim_action.get_space(self.interface)
        self.space_function.assert_called_once()

    def test_get_schedule(self):
        array = numpy.array([[1, 0], [0, 1]])
        self.assertEqual(self.sim_action.get_schedule(self.interface, array),
                         {'a': [0]})


if __name__ == '__main__':
    unittest.main()
