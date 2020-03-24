# coding=utf-8
"""
This package contains customizable gym environments that wrap
simulations.
"""
from copy import deepcopy
from typing import Optional, Dict, List, Callable, Any

import numpy as np
from gym import spaces

from .base_env import BaseSimEnv
from . import observation as obs, reward_functions as rf
from .action_spaces import SimAction, zero_centered_single_charging_schedule
from .observation import SimObservation
from ...interface import GymTrainedInterface


class CustomSimEnv(BaseSimEnv):
    """ A simulator environment with customizable observations, action
    spaces, and rewards.

    Observations are specified as objects, where each object specifies a
    function to generate a space from a simulation interface and a
    function to generate an observation from a simulation interface.

    Action spaces are specified as functions that generate a space from
    a simulation interface.

    Rewards are specified as functions that generate a number (reward)
    from an environment.

    Users may define their own objects/functions to input to this
    environment, use the objects/functions defined in the gym_acnsim
    package, or use an environment factory function defined in the
    sim_prototype_env module.
    """
    observation_objects: List[SimObservation]
    observation_space: spaces.Dict
    action_object: SimAction
    action_space: spaces.Space
    reward_functions: List[Callable[[BaseSimEnv], float]]

    def __init__(
            self,
            interface: Optional[GymTrainedInterface],
            observation_objects: List[SimObservation],
            action_object: SimAction,
            reward_functions: List[Callable[[BaseSimEnv], float]]
    ) -> None:
        """ Initialize this environment. Every CustomSimEnv needs a list
        of SimObservation objects, action space functions, and reward
        functions.

        Args:
            interface (acnsim.GymInterface): See BaseSimEnv.__init__.
            observation_objects (List[SimObservation]): List of objects
                that specify how to calculate observation spaces and
                observations from an interface to a simulation.
            action_object (SimAction): List of functions that
                specify how to calculate action spaces from an interface
                to a simulation.
            reward_functions (List[Callable[[BaseSimEnv], float]]): List
                of functions which take as input a BaseSimEnv instance
                and return a number.
        """
        super().__init__(interface)

        self.observation_objects = observation_objects
        self.action_object = action_object
        self.reward_functions = reward_functions
        if interface is None:
            return
        self.observation_space = spaces.Dict({
            observation_object.name: observation_object.get_space(
                self.interface)
            for observation_object in observation_objects
        })

        self.action_space = action_object.get_space(interface)

    @property
    def interface(self) -> GymTrainedInterface:
        return self._interface

    @interface.setter
    def interface(self, new_interface: GymTrainedInterface) -> None:
        if self._interface is None:
            self._init_snapshot = deepcopy(new_interface)
            self._prev_interface = deepcopy(new_interface)
        self._interface = new_interface
        self.observation_space = spaces.Dict({
            observation_object.name: observation_object.get_space(
                new_interface)
            for observation_object in self.observation_objects
        })

        self.action_space = self.action_object.get_space(new_interface)

    def render(self, mode='human'):
        """ Renders the environment. Implements gym.Env.render(). """
        raise NotImplementedError

    def action_to_schedule(self) -> Dict[str, List[float]]:
        """ Convert an agent action to a schedule to be input to the
        simulator.

        Returns:
            schedule (Dict[str, List[float]]): Dictionary mapping
                station ids to a schedule of pilot signals.
        """
        return self.action_object.get_schedule(self.interface, self.action)

    def observation_from_state(self) -> Dict[str, np.ndarray]:
        """ Construct an environment observation from the state of the
        simulator using the environment's observation construction
        functions.

        Returns:
            observation (Dict[str, np.ndarray]): An environment
                observation generated from the simulation state
        """
        return {
            observation_object.name: observation_object.get_obs(
                self.interface)
            for observation_object in self.observation_objects
        }

    def reward_from_state(self) -> float:
        """ Calculate a reward from the state of the simulator

        Returns:
            reward (float): a reward generated from the simulation
                state
        """
        return sum(np.array([
            reward_func(self) for reward_func in self.reward_functions]))

    def done_from_state(self) -> bool:
        """ Determine if the simulation is done from the state of the
        simulator

        Returns:
            done (bool): True if the simulation is done, False if not
        """
        return self.interface.is_done

    def info_from_state(self) -> Dict[Any, Any]:
        """ Give information about the environment using the state of
        the simulator. In this case, all the info about the simulator
        is given by returning a dict containing the simulator's
        interface.

        Returns:
            info (Dict[str, GymTrainedInterface]): The interface between
                the environment and Simulator.
        """
        return {'interface': self.interface}


# Default observation objects, action object, and reward functions list
# for use with make_default_sim_env and make_rebuilding_default_sim_env.
default_observation_objects: List[SimObservation] = [
    obs.arrival_observation(),
    obs.departure_observation(),
    obs.remaining_demand_observation(),
    obs.constraint_matrix_observation(),
    obs.magnitudes_observation(),
    obs.timestep_observation()
]
default_action_object: SimAction = zero_centered_single_charging_schedule()
default_reward_functions: List[Callable[[BaseSimEnv], float]] = [
    rf.evse_violation,
    rf.unplugged_ev_violation,
    rf.current_constraint_violation,
    rf.hard_charging_reward
]


def make_default_sim_env(
        interface: Optional[GymTrainedInterface] = None) -> CustomSimEnv:
    """ A simulator environment with the following characteristics:

    The action and observation spaces are continuous.

    An action in this environment is a pilot signal for each EVSE,
    within the minimum and maximum EVSE rates.

    An observation is a dict consisting of fields (times are 1-indexed
    in the observations):
        arrivals: arrival time of the EV at each EVSE (or 0 if there's
             no EV plugged in)
        departures: departure time of the EV at each EVSE (or 0 if
            there's no EV plugged in)
        demand: energy demand of the EV at each EVSE (unoccupied
            EVSEs have demand 0)
        constraint_matrix: matrix of aggregate current coefficients
        magnitudes: magnitude vector constraining aggregate currents
        timestep: timestep of the simulation

    The reward is calculated as follows:
        If no constraints (on the network or on the EVSEs) were
            violated by the action,
        a reward equal to the total charge delivered (in A) is
            returned
        If any constraint violation occurred, a negative reward equal
            to the magnitude of the violation is returned.
        Network constraint violations are scaled by the number of EVs
        Finally, a user-input reward function is added to the total
            reward.

    The simulation is considered done if the event queue is empty.
    """
    return CustomSimEnv(
        interface,
        default_observation_objects,
        default_action_object,
        default_reward_functions
    )


class RebuildingEnv(CustomSimEnv):
    """ A simulator environment that subclasses CustomSimEnv, with
    the extra property that the entire simulation is rebuilt within 
    the environment when __init__ or reset are called

    This is especially useful if the network or event queue have 
    stochastic elements.
    """
    def __init__(
            self,
            interface: Optional[GymTrainedInterface],
            observation_objects: List[SimObservation],
            action_object: SimAction,
            reward_functions: List[Callable[[BaseSimEnv], float]],
            interface_generating_function: Optional[
                Callable[[], GymTrainedInterface]] = None
    ) -> None:
        """ Initialize this environment. Every CustomSimEnv needs a list
        of SimObservation objects, action space functions, and reward
        functions. At least one of either interface or
        interface_generating_function must not be None.

        Args:
            interface (acnsim.GymTrainedInterface): An interface with
                which the environment is initialized if no interface
                generating function is provided. If an interface
                generating function is provided, this argument is
                unused.
            observation_objects (List[SimObservation]): List of objects
                that specify how to calculate observation spaces and
                observations from an interface to a simulation.
            action_object (SimAction): List of functions that
                specify how to calculate action spaces from an interface
                to a simulation.
            reward_functions (List[Callable[[BaseSimEnv], float]]): List
                of functions which take as input a BaseSimEnv instance
                and return a number.
            interface_generating_function (Optional[Callable[[],
                                                    GymInterface]]):
                Function which returns a GymInterface to a generated
                simulator.
        """
        if interface_generating_function is None and interface is None:
            raise TypeError("At least one of either interface or "
                            "interface_generating_function must not be "
                            "None")

        if interface_generating_function is None:
            self._init_snapshot = deepcopy(interface)

            # noinspection PyMissingOrEmptyDocstring
            def interface_generating_function() -> GymTrainedInterface:
                return self._init_snapshot
        else:
            interface: GymTrainedInterface = \
                interface_generating_function()

        self.interface_generating_function = interface_generating_function

        super().__init__(interface,
                         observation_objects,
                         action_object,
                         reward_functions)

    @classmethod
    def from_custom_sim_env(
            cls,
            env: CustomSimEnv,
            interface_generating_function: Optional[
                Callable[[], GymTrainedInterface]] = None
    ) -> 'RebuildingEnv':
        return cls(env.interface,
                   env.observation_objects,
                   env.action_object,
                   env.reward_functions,
                   interface_generating_function=interface_generating_function)

    def reset(self) -> Dict[str, np.ndarray]:
        """ Resets the state of the simulation and returns an initial 
        observation. Resetting is done by setting the interface to 
        the simulation to an interface to the simulation in its 
        initial state.
        
        Returns:
            observation (np.ndarray): the initial observation.
        """
        temp_interface = self.interface_generating_function()
        self.interface = deepcopy(temp_interface)
        self._prev_interface = deepcopy(temp_interface)
        self._init_snapshot = deepcopy(temp_interface)
        return self.observation_from_state()

    def render(self, mode='human'):
        """ Renders the environment. Implements gym.Env.render(). """
        raise NotImplementedError


def make_rebuilding_default_sim_env(
        interface_generating_function: Optional[
            Callable[[], GymTrainedInterface]]
) -> RebuildingEnv:
    """ A simulator environment with the same characteristics as the
    environment returned by make_default_sim_env except on every reset,
    the simulation is completely rebuilt using interface_generating_function.

    See make_default_sim_env for more info.
    """
    interface = interface_generating_function()
    return RebuildingEnv.from_custom_sim_env(
        make_default_sim_env(interface),
        interface_generating_function=interface_generating_function
    )
