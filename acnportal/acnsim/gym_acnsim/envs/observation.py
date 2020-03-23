# coding=utf-8
"""
Module containing definition of a gym_acnsim observation and factory
functions for different builtin observations.

See the SimObservation docstring for more information on the
SimObservation class.

Each factory function takes no arguments and returns an instance of type
SimObservation. Each factory function defines a space_function and and
an obs_function with the following signatures:

space_function: Callable[[GymInterface], spaces.Space]
obs_function: Callable[[GymInterface], np.ndarray]

The space_function gives a gym space for a given observation type.
The obs_function gives a gym observation for a given observation type.
The observation returned by obs_function is a point in the space
returned by space_function.
"""
from typing import Callable

import numpy as np
from gym import spaces

from acnportal.acnsim import EV
from acnportal.acnsim.interface import GymTrainedInterface


class SimObservation:
    """
    Class representing an OpenAI Gym observation of an ACN-Sim
    simulation.

    An instance of SimObservation contains a space_function, which
    generates a gym space from an input Interface using attributes and
    functions of the input Interface, and an obs_function, which
    generates a gym observation from an input Interface using attributes
    and functions of the input Interface. Each instance also requires a
    name (given as a string).

    This class enables Simulation environments with customizable
    observations, as a SimObservation object with user-defined or built
    in space and obs functions can be input to a BaseSimEnv-like object
    to enable a new observation without creating a new environment.

    Each type of observation is the same type of object, but the details
    of the space and obs functions are different. This was done because
    space and obs functions are static, as observations of a specific
    type do not have any attributes. However, each observation type
    requires both a space and observation generating function, so a
    wrapping data structure is required.

    Attributes:
        _space_function (Callable[[GymInterface], spaces.Space]):
            Function that accepts a GymInterface and generates a gym
            space in which all observations for this instance exist.
        _obs_function (Callable[[GymInterface], np.ndarray]): Function
            that accepts a GymInterface and generates a gym observation
            based on the input interface.
        name (str): Name of this observation. This attribute allows an
            environment to distinguish between different types of
            observation.
    """
    _space_function: Callable[[GymTrainedInterface], spaces.Space]
    _obs_function: Callable[[GymTrainedInterface], np.ndarray]
    name: str

    def __init__(self,
                 space_function: Callable[[GymTrainedInterface], spaces.Space],
                 obs_function: Callable[[GymTrainedInterface], np.ndarray],
                 name: str) -> None:
        """
        Args:
            space_function (Callable[[GymInterface], spaces.Space]):
                Function that accepts a GymInterface and generates a
                gym space in which all observations for this instance
                exist.
            obs_function (Callable[[GymInterface], np.ndarray]):
                Function that accepts a GymInterface and generates a
                gym observation based on the input interface.
            name (str): Name of this observation. This attribute allows
                an environment to distinguish between different types of
                observation.

        Returns:
            None.
        """
        self._space_function = space_function
        self._obs_function = obs_function
        self.name = name

    def get_space(self, interface: GymTrainedInterface) -> spaces.Space:
        """
        Returns the gym space in which all observations for this
        observation type exist. The characteristics of the interface
        (for example, number of EVSEs if station demands are observed)
        may change the dimensions of the returned space, so this method
        requires a GymInterface as input.

        Args:
            interface (GymTrainedInterface): Interface to an ACN-Sim Simulation
                that contains details of and functions to generate
                details about the current Simulation.

        Returns:
            spaces.Space: A gym space in which all observations for this
                observation type exist.
        """
        return self._space_function(interface)

    def get_obs(self, interface: GymTrainedInterface) -> np.ndarray:
        """
        Returns a gym observation for the state of the simulation given
        by interface. The exact observation depends on both the input
        interface and the observation generating function obs_func with
        which this object was initialized.

        Args:
            interface (GymTrainedInterface): Interface to an ACN-Sim Simulation
                that contains details of and functions to generate
                details about the current Simulation.

        Returns:
            np.ndarray: A gym observation generated by _obs_function
                with this interface.
        """
        return self._obs_function(interface)


# Per active EV observation factory functions. Note that all EV data
# is shifted up by 1, as 0's indicate no EV is plugged in.
def _ev_observation(
        attribute_function: Callable[[GymTrainedInterface, EV], float],
        name: str
) -> SimObservation:
    # noinspection PyMissingOrEmptyDocstring
    def space_function(interface: GymTrainedInterface) -> spaces.Space:
        return spaces.Box(
            low=0, high=np.inf,
            shape=(len(interface.station_ids),),
            dtype='float'
        )

    # noinspection PyMissingOrEmptyDocstring
    def obs_function(interface: GymTrainedInterface) -> np.ndarray:
        attribute_values: dict = {station_id: 0
                                  for station_id in interface.station_ids}
        for ev in interface.active_evs:
            attribute_values[ev.station_id] = attribute_function(
                interface, ev) + 1
        return np.array(list(attribute_values.values()))
    return SimObservation(space_function, obs_function, name=name)


def arrival_observation() -> SimObservation:
    """ Generates a SimObservation instance that wraps functions to
    observe active EV arrivals.

    Zeros in the output observation array indicate no EV is plugged in;
    as such, all observations are shifted up by 1.
    """
    return _ev_observation(lambda _, ev: ev.arrival, 'arrivals')


def departure_observation() -> SimObservation:
    """ Generates a SimObservation instance that wraps functions to
    observe active EV departures.

    Zeros in the output observation array indicate no EV is plugged in;
    as such, all observations are shifted up by 1.
    """
    return _ev_observation(lambda _, ev: ev.departure, 'departures')


def remaining_demand_observation() -> SimObservation:
    """ Generates a SimObservation instance that wraps functions to
    observe active EV remaining energy demands in amp periods.

    Zeros in the output observation array indicate no EV is plugged in;
    as such, all observations are shifted up by 1.
    """
    return _ev_observation(
        lambda interface, ev: interface.remaining_amp_periods(ev), 'demands')


# Network-wide observation factory functions.
def _constraints_observation(attribute: str, name: str) -> SimObservation:
    # noinspection PyMissingOrEmptyDocstring
    def space_function(interface: GymTrainedInterface) -> spaces.Space:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=getattr(interface.get_constraints(), attribute).shape,
            dtype='float'
        )

    # noinspection PyMissingOrEmptyDocstring
    def obs_function(interface: GymTrainedInterface) -> np.ndarray:
        return getattr(interface.get_constraints(), attribute)
    return SimObservation(space_function, obs_function, name=name)


def constraint_matrix_observation() -> SimObservation:
    """ Generates a SimObservation instance that wraps functions to
    observe the network constraint matrix.
    """
    return _constraints_observation('constraint_matrix', 'constraint matrix')


def magnitudes_observation() -> SimObservation:
    """ Generates a SimObservation instance that wraps functions to
    observe the network limiting current magnitudes in amps.
    """
    return _constraints_observation('magnitudes', 'magnitudes')


def timestep_observation() -> SimObservation:
    """ Generates a SimObservation instance that wraps functions to
    observe the current timestep of the simulation, in periods.

    To comply with the timesteps returned by arrival and departure
    observations, the observed timestep is one greater than than that
    returned by the simulation. Simulations thus start at timestep 1
    from an RL agent's perspective.
    """
    # noinspection PyUnusedLocal
    # noinspection PyMissingOrEmptyDocstring
    def space_function(interface: GymTrainedInterface) -> spaces.Space:
        return spaces.Box(low=0, high=np.inf, shape=(1,), dtype='float')

    # noinspection PyMissingOrEmptyDocstring
    def obs_function(interface: GymTrainedInterface) -> np.ndarray:
        return np.array(interface.current_time + 1)
    return SimObservation(space_function, obs_function, name='timestep')
