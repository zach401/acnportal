# coding=utf-8
"""
Module containing definition of a gym_acnsim action and factory
functions for different builtin actions.

See the SimAction docstring for more information on the
SimAction class.

Each factory function takes no arguments and returns an instance of type
SimAction. Each factory function defines a space_function and and
an to_schedule function with the following signatures:

space_function:
    Callable[[GymInterface], Space]
to_schedule:
    Callable[[GymInterface, np.ndarray], Dict[str, List[float]]]

The space_function gives a gym space for a given action type.
to_schedule gives an ACN-Sim schedule for a given action. The
to_schedule method does not enforce action space constraints, as some
learning algorithms treat action space constraints as loose rather than
strict.
"""
from typing import Callable, Dict, List

import numpy as np
from gym import Space
from gym.spaces import Box

from acnportal.acnsim.interface import GymTrainedInterface


class SimAction:
    """
    Class representing an OpenAI Gym action for an ACN-Sim Simulation.

    An instance of SimAction contains a space_function, which
    generates a gym space from an input interface using attributes
    and functions of the input Interface, and a to_schedule function,
    which generates a ACN-Sim schedule from an input interface and
    action. Gym observations can have many types, including numpy
    ndarray, but ACN-Sim schedules must be of type Dict[str,
    List[float]], namely a list of floats for each station id in the
    network. Each instance also requires name (given as a string).

    This class enables Simulation environments with customizable
    actions, as a SimAction object with user-defined or built in
    space and to_schedule functions can be input to a BaseSimEnv-like
    object to enable a new action type without creating a new
    environment.

    Each type of action is the same type of object, but the details
    of the space and to_schedule functions are different. This was
    done because space and to_schedule functions are static,
    as actions of a specific type do not have any attributes.
    However, each action type requires both a space and schedule
    generating function, so a wrapping data structure is required.

    Attributes:
        _space_function (Callable[[GymInterface], Space]):
            Function that accepts a GymInterface and generates a gym
            space in which all actions for this instance exist.
        _to_schedule (Callable[[GymInterface, np.ndarray],
                      Dict[str, List[float]]]):
            Function that accepts an interface to a simulation and an
            action and generates a schedule that can be submitted to
            an ACN-Sim Simulation.
        name (str): Name of this action. This attribute allows an
            environment to distinguish between different types of
            actions.
    """
    _space_function: Callable[[GymTrainedInterface], Space]
    _to_schedule: Callable[[GymTrainedInterface, np.ndarray],
                           Dict[str, List[float]]]
    name: str

    def __init__(self,
                 space_function: Callable[[GymTrainedInterface], Space],
                 to_schedule: Callable[[GymTrainedInterface, np.ndarray],
                                       Dict[str, List[float]]],
                 name: str) -> None:
        """
        Args:
            space_function (Callable[[GymInterface], Space]):
                Function that accepts a GymInterface and generates a
                gym space in which all observations for this instance
                exist.
            to_schedule (Callable[[GymInterface, np.ndarray],
                         Dict[str, List[float]]]):
                Function that accepts a GymInterface and an action and
                generates an ACN Sim schedule based on the action.
            name (str): Name of this observation. This attribute allows
                an environment to distinguish between different types
                of observation.
        Returns:
            None.

        """
        self._space_function = space_function
        self._to_schedule = to_schedule
        self.name = name

    def get_space(self, interface: GymTrainedInterface) -> Space:
        """
        Returns the gym space in which all actions for this action
        type exist. The characteristics of the interface (for
        example, number of EVSEs if charging schedules are given) may
        change the dimensions of the returned space, so this method
        requires a GymInterface as input.

        Args:
            interface (GymTrainedInterface): Interface to an ACN-Sim
                Simulation that contains details of and functions to
                generate details about the current Simulation.

        Returns:
            Space: A gym space in which all actions for this
                action type exist.
        """
        return self._space_function(interface)

    def get_schedule(self, interface: GymTrainedInterface,
                     action: np.ndarray) -> Dict[str, List[float]]:
        """
        Returns an ACN-Sim schedule given an input action.

        Args:
            interface (GymTrainedInterface): Interface to a simulation.
            action (np.ndarray): Action to be converted into an ACN-Sim
                schedule.

        Returns:
            Dict[str, List[float]]: A charging schedule that can be
                submitted to an ACN-Sim Simulator.
        """
        return self._to_schedule(interface, action)


# Action factory functions.
def single_charging_schedule() -> SimAction:
    """ Generates a SimAction instance that wraps functions to handle
    actions taking the form of a vector of pilot signals. For this
    action type, a single entry represents the pilot signal sent to
    single EVSE at the current timestep. The space bounds
    pilot signals above the maximum allowable rate over all EVSEs and
    below the minimum allowable rate over all EVSEs.

    As a 0 min rate is assumed to be allowed, the action space lower
    bound is set to 0 if the station min rates are all greater than 0.
    """

    # noinspection PyMissingOrEmptyDocstring
    def space_function(interface: GymTrainedInterface) -> Box:
        num_evses: int = len(interface.station_ids)
        max_rate: float = max([interface.max_pilot_signal(station_id)
                               for station_id in interface.station_ids])
        min_rate: float = min(0.0, min(
            [interface.min_pilot_signal(station_id)
             for station_id in interface.station_ids]
        ))
        return Box(low=min_rate, high=max_rate,
                   shape=(num_evses,), dtype='float')

    # noinspection PyMissingOrEmptyDocstring
    def to_schedule(interface: GymTrainedInterface,
                    action: np.ndarray) -> Dict[str, List[float]]:
        if len(action.shape) > 1:
            raise TypeError(
                f"Single schedule action type only accepts schedules "
                f"of length <= 1 in a 1-D numpy array. Got shape = "
                f"{len(action.shape)}."
            )
        return {interface.station_ids[i]: [action[i]]
                for i in range(len(action))}
    return SimAction(space_function, to_schedule, 'single schedule')


def zero_centered_single_charging_schedule() -> SimAction:
    """ Generates a SimAction instance that wraps functions to handle
    actions taking the form of a vector of pilot signals. For this
    action type, actions are assumed to be centered about 0, in that
    an action of 0 corresponds to a pilot signal of max_rate/2. So,
    to convert to a schedule, actions need to be shifted by a certain
    amount and converted to a dictionary.

    As a 0 min rate is assumed to be allowed, the action space lower
    bound is set to -rate_offset_array if the station min rates are all
    greater than 0.
    """

    # noinspection PyMissingOrEmptyDocstring
    def space_function(interface: GymTrainedInterface) -> Box:
        num_evses: int = len(interface.station_ids)
        max_rates: np.ndarray = np.array(
            [interface.max_pilot_signal(station_id)
             for station_id in interface.station_ids]
        )
        min_rates: np.ndarray = np.array(
            [interface.min_pilot_signal(station_id)
             for station_id in interface.station_ids]
        )
        rate_offset_array: np.ndarray = (max_rates + min_rates) / 2
        return Box(low=min(min(-rate_offset_array),
                           min(min_rates - rate_offset_array)),
                   high=max(max_rates - rate_offset_array),
                   shape=(num_evses,), dtype='float')

    # noinspection PyMissingOrEmptyDocstring
    def to_schedule(interface: GymTrainedInterface,
                    action: np.ndarray) -> Dict[str, List[float]]:
        if len(action.shape) > 1:
            raise TypeError(
                f"Single schedule action type only accepts schedules "
                f"of length <= 1 in a 1-D numpy array. Got shape = "
                f"{len(action.shape)}."
            )
        max_rates: np.ndarray = np.array(
            [interface.max_pilot_signal(station_id)
             for station_id in interface.station_ids]
        )
        min_rates: np.ndarray = np.array(
            [interface.min_pilot_signal(station_id)
             for station_id in interface.station_ids]
        )
        rate_offset_array: np.ndarray = (max_rates + min_rates) / 2
        offset_action: np.ndarray = action + rate_offset_array
        return {interface.station_ids[i]: [offset_action[i]]
                for i in range(len(offset_action))}
    return SimAction(space_function, to_schedule,
                     'zero-centered single schedule')
