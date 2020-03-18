# coding=utf-8
""" Tests for custom ACN-Sim gym environment. """
import unittest
from importlib.util import find_spec
from typing import Dict
from unittest.mock import create_autospec, Mock

import numpy as np

from .... import Simulator

if find_spec("gym") is not None:
    from ....interface import GymTrainedInterface, GymTrainingInterface
    from .. import BaseSimEnv


class TestCustomSimEnv(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
