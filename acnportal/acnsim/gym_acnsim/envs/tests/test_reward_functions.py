# coding=utf-8
""" Tests for reward functions. """
import unittest
from unittest.mock import create_autospec
from importlib.util import find_spec

from ....interface import GymTrainingInterface
from .. import reward_functions as rf, CustomSimEnv


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestEVSEViolation(unittest.TestCase):
    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        cls.env: CustomSimEnv = create_autospec(CustomSimEnv)
        cls.env.interface = create_autospec(GymTrainingInterface)

    def test_evse_violation(self) -> None:
        self.evse_violation: float = rf.evse_violation(self.env)


if __name__ == '__main__':
    unittest.main()
