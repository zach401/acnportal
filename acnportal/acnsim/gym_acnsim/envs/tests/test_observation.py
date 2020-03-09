# coding=utf-8
""" Tests for SimObservation and observation generating functions. """
import unittest
from collections import namedtuple
from importlib.util import find_spec
from typing import Any, Callable, Optional
from unittest.mock import create_autospec

import numpy as np

from acnportal.acnsim import EV

if find_spec("gym") is not None:
    from gym import Space
    from .. import observation as obs
from ....interface import GymTrainedInterface


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestSimObservation(unittest.TestCase):
    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        # The type here is Any as space_function is actually a Mock
        # object, but there's no Mock type in the typing library.
        cls.space_function: Any = create_autospec(lambda interface: Space())
        cls.obs_function: Callable[[GymTrainedInterface], np.ndarray] = \
            lambda interface: np.array([0, 0])
        cls.name: str = "stub_observation"
        cls.sim_observation: obs.SimObservation = obs.SimObservation(
            cls.space_function, cls.obs_function, cls.name)
        cls.interface: GymTrainedInterface = create_autospec(
            GymTrainedInterface)

    def test_correct_on_init_sim_observation_name(self) -> None:
        self.assertEqual(self.sim_observation.name, self.name)

    def test_get_space(self) -> None:
        self.sim_observation.get_space(self.interface)
        self.space_function.assert_called_once()

    def test_get_schedule(self) -> None:
        np.testing.assert_equal(self.sim_observation.get_obs(self.interface),
                                np.array([0, 0]))


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestEVObservationClass(unittest.TestCase):
    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        # The type here is Any as space_function is actually a Mock
        # object, but there's no Mock type in the typing library.
        cls.ev1: Any = create_autospec(EV)
        cls.ev1.arrival = 0
        cls.ev1.departure = 10
        cls.ev1.station_id = "T1"
        cls.remaining_amp_periods1 = 20
        cls.ev2: Any = create_autospec(EV)
        cls.ev2.arrival = 5
        cls.ev2.departure = 25
        cls.ev2.station_id = "T3"
        cls.remaining_amp_periods2 = 10
        cls.interface: Any = create_autospec(GymTrainedInterface)
        cls.interface.active_evs = [cls.ev1, cls.ev2]

        cls.interface.remaining_amp_periods = (
            lambda ev: (cls.remaining_amp_periods1
                        if ev.station_id == cls.ev1.station_id
                        else cls.remaining_amp_periods2))

        cls.station_ids = [cls.ev1.station_id, "T2", cls.ev2.station_id]
        cls.num_stations = len(cls.station_ids)
        cls.interface.station_ids = cls.station_ids
        cls.sim_observation: Optional[obs.SimObservation] = None
        cls.obs_name: Optional[str] = None

    def test_space_function(self) -> None:
        if self.sim_observation is None:
            return
        out_space: Space = self.sim_observation.get_space(self.interface)
        self.assertEqual(out_space.shape, (self.num_stations,))
        np.testing.assert_equal(out_space.low, self.num_stations * [0])
        np.testing.assert_equal(out_space.high, self.num_stations * [np.inf])
        self.assertEqual(out_space.dtype, 'float')

    def test_correct_on_init_name(self) -> None:
        if self.obs_name is None:
            return
        self.assertEqual(self.sim_observation.name, self.obs_name)


class TestArrivalObservation(TestEVObservationClass):
    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.sim_observation = obs.arrival_observation()
        cls.obs_name = 'arrivals'

    def test_arrival_observation(self) -> None:
        np.testing.assert_equal(self.sim_observation.get_obs(self.interface),
                                np.array([self.ev1.arrival + 1,
                                          0,
                                          self.ev2.arrival + 1]))


class TestDepartureObservation(TestEVObservationClass):
    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.sim_observation = obs.departure_observation()
        cls.obs_name = 'departures'

    def test_departure_observation(self) -> None:
        np.testing.assert_equal(self.sim_observation.get_obs(self.interface),
                                np.array([self.ev1.departure + 1,
                                          0,
                                          self.ev2.departure + 1]))


class TestDemandObservation(TestEVObservationClass):
    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.sim_observation = obs.remaining_demand_observation()
        cls.obs_name = 'demands'

    def test_departure_observation(self) -> None:
        np.testing.assert_equal(self.sim_observation.get_obs(self.interface),
                                np.array([self.remaining_amp_periods1 + 1,
                                          0,
                                          self.remaining_amp_periods2 + 1]))


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestConstraintObservation(unittest.TestCase):
    # Some class variables are defined outside of setUpClass so that
    # the code inspector knows that inherited classes have these
    # attributes.
    # The type here is Any as space_function is actually a Mock
    # object, but there's no Mock type in the typing library.
    interface: Any = create_autospec(GymTrainedInterface)

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim_observation: Optional[obs.SimObservation] = None
        cls.attribute_name: Optional[str] = None
        cls.obs_name: Optional[str] = None
        cls.obs_shape = None

    def test_space_function(self) -> None:
        if self.sim_observation is None:
            return
        out_space: Space = self.sim_observation.get_space(self.interface)
        self.assertEqual(out_space.shape, self.obs_shape)
        np.testing.assert_equal(out_space.low,
                                -np.inf * np.ones(self.obs_shape))
        np.testing.assert_equal(out_space.high,
                                np.inf * np.ones(self.obs_shape))
        self.assertEqual(out_space.dtype, 'float')

    def test_correct_on_init_name(self) -> None:
        if self.obs_name is None:
            return
        self.assertEqual(self.sim_observation.name, self.obs_name)


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestConstraintMatrixObservation(TestConstraintObservation):
    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.sim_observation = obs.constraint_matrix_observation()
        cls.constraint_matrix = np.array([[1, 0], [0, 1]])
        cls.obs_name = 'constraint matrix'
        cls.attribute_name = 'constraint_matrix'
        cls.obs_shape = cls.constraint_matrix.shape
        cls.interface.get_constraints = lambda: namedtuple(
            'Constraint', [cls.attribute_name])(cls.constraint_matrix)

    def test_constraint_matrix_observation(self) -> None:
        np.testing.assert_equal(self.sim_observation.get_obs(self.interface),
                                self.constraint_matrix)


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestMagnitudesObservation(TestConstraintObservation):
    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.sim_observation = obs.magnitudes_observation()
        cls.magnitudes = np.array([1, 1])
        cls.obs_name = 'magnitudes'
        cls.attribute_name = 'magnitudes'
        cls.obs_shape = cls.magnitudes.shape
        cls.interface.get_constraints = lambda: namedtuple(
            'Constraint', [cls.attribute_name])(cls.magnitudes)

    def test_constraint_matrix_observation(self) -> None:
        np.testing.assert_equal(self.sim_observation.get_obs(self.interface),
                                self.magnitudes)


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestTimestepObservation(unittest.TestCase):
    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        # The type here is Any as space_function is actually a Mock
        # object, but there's no Mock type in the typing library.
        cls.interface = create_autospec(GymTrainedInterface)
        cls.interface.current_time = 100
        cls.sim_observation: obs.SimObservation = obs.timestep_observation()
        cls.obs_name: str = 'timestep'

    def test_space_function(self) -> None:
        out_space: Space = self.sim_observation.get_space(self.interface)
        self.assertEqual(out_space.shape, (1,))
        np.testing.assert_equal(out_space.low, [0])
        np.testing.assert_equal(out_space.high, [np.inf])
        self.assertEqual(out_space.dtype, 'float')

    def test_correct_on_init_name(self) -> None:
        self.assertEqual(self.sim_observation.name, self.obs_name)

    def test_timestep_observation(self) -> None:
        np.testing.assert_equal(self.sim_observation.get_obs(self.interface),
                                self.interface.current_time + 1)


if __name__ == '__main__':
    unittest.main()
