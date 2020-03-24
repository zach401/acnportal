# coding=utf-8
"""
This module contains an abstract gym environment that wraps an ACN-Sim
Simulation.
"""
from copy import deepcopy
from typing import Optional, Dict, List, Any, Tuple

import gym
import numpy as np

from ...interface import GymTrainedInterface, GymTrainingInterface


class BaseSimEnv(gym.Env):
    """ Abstract base class meant to be inherited from to implement
    new ACN-Sim Environments.

    Subclasses must implement the following methods:
        action_to_schedule
        observation_from_state
        reward_from_state
        done_from_state

    Subclasses must also specify observation_space and action_space,
    either as class or instance variables.

    Optionally, subclasses may implement info_from_state, which here
    returns an empty dict.

    Subclasses may override __init__, step, and reset functions.

    Currently, no render function is implemented, though this function
    is not required for internal functionality.

    Attributes:
        _interface (GymTrainedInterface): An interface to a simulation to be
            stepped by this environment, or None. If None, an interface must
            be set later.
        _init_snapshot (GymTrainedInterface): A deep copy of the initial
            interface, used for environment resets.
        _prev_interface (GymTrainedInterface): A deep copy of the interface
            at the previous time step; used for calculating action
            rewards.
        _action (object): The action taken by the agent in this
            agent-environment loop iteration.
        _schedule (Dict[str, List[number]]): Dictionary mapping
            station ids to a schedule of pilot signals.
        _observation (np.ndarray): The observation given to the agent in
            this agent-environment loop iteration.
        _done (object): An object representing whether or not the
            execution of the environment is complete.
        _info (object): An object that gives info about the environment.
    """
    _interface: Optional[GymTrainedInterface]
    _init_snapshot: GymTrainedInterface
    _prev_interface: GymTrainedInterface
    _action: Optional[np.ndarray]
    _schedule: Dict[str, List[float]]
    _observation: Optional[np.ndarray]
    _reward: Optional[float]
    _done: Optional[bool]
    _info: Optional[Dict[Any, Any]]

    def __init__(self, interface: Optional[GymTrainedInterface]) -> None:
        self._interface = interface
        self._init_snapshot = deepcopy(interface)
        self._prev_interface = deepcopy(interface)
        self._action = None
        self._schedule = {}
        self._observation = None
        self._reward = None
        self._done = None
        self._info = None

    @property
    def interface(self) -> GymTrainedInterface:
        return self._interface

    @interface.setter
    def interface(self, new_interface: GymTrainedInterface) -> None:
        if self._interface is None:
            self._init_snapshot = deepcopy(new_interface)
            self._prev_interface = deepcopy(new_interface)
        self._interface = new_interface

    @property
    def prev_interface(self) -> GymTrainedInterface:
        return self._prev_interface

    @prev_interface.setter
    def prev_interface(self, new_prev_interface: GymTrainedInterface) -> None:
        self._prev_interface = new_prev_interface

    @property
    def action(self) -> np.ndarray:
        return deepcopy(self._action)

    @action.setter
    def action(self, new_action: np.ndarray) -> None:
        self._action = new_action

    @property
    def schedule(self) -> Dict[str, List[float]]:
        return deepcopy(self._schedule)

    @schedule.setter
    def schedule(self, new_schedule: Dict[str, List[float]]) -> None:
        self._schedule = new_schedule

    @property
    def observation(self) -> np.ndarray:
        return deepcopy(self._observation)

    @observation.setter
    def observation(self, new_observation: np.ndarray) -> None:
        self._observation = new_observation

    @property
    def reward(self) -> float:
        return deepcopy(self._reward)

    @reward.setter
    def reward(self, new_reward: float) -> None:
        self._reward = new_reward

    @property
    def done(self) -> bool:
        return deepcopy(self._done)

    @done.setter
    def done(self, new_done: bool) -> None:
        self._done = new_done

    @property
    def info(self) -> Dict[Any, Any]:
        return deepcopy(self._info)

    @info.setter
    def info(self, new_info: Dict[Any, Any]) -> None:
        self._info = new_info

    def update_state(self) -> None:
        """ Update the state of the environment. Namely, the
        observation, reward, done, and info attributes of the
        environment.

        Returns:
            None.
        """
        if not isinstance(self._interface, GymTrainedInterface):
            raise TypeError(
                "Environment interface must be of type "
                "GymTrainedInterface to update state. Either "
                "use sim.run() to progress the environment or set a "
                "new interface."
            )
        self.observation = self.observation_from_state()
        self.reward = self.reward_from_state()
        self.done = self.done_from_state()
        self.info = self.info_from_state()

    def store_previous_state(self) -> None:
        """ Store the previous state of the simulation in the
        _prev_interface environment attribute.

        Returns:
            None.
        """
        self._prev_interface = self.interface

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """ Step the simulation one timestep with an agent's action.

        Accepts an action and returns a tuple (observation, reward,
        done, info).

        Implements gym.Env.step()

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (np.ndarray): agent's observation of the current
                environment
            reward (float) : amount of reward returned after previous
                action
            done (bool): whether the episode has ended, in which case
                further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information
                (helpful for debugging, and sometimes learning)
        """
        if not isinstance(self._interface, GymTrainingInterface):
            raise TypeError(
                "Environment interface must be of type "
                "GymTrainingInterface to call step function. Either "
                "use sim.run() to progress the environment or set a "
                "new interface."
            )
        self.action = action
        self.schedule = self.action_to_schedule()

        self.store_previous_state()
        self._interface.step(self.schedule)

        self.update_state()

        return self.observation, self.reward, self.done, self.info

    def reset(self) -> Dict[str, np.ndarray]:
        """ Resets the state of the simulation and returns an initial
        observation. Resetting is done by setting the interface to the
        simulation to an interface to the simulation in its initial
        state.

        Implements gym.Env.reset()

        Returns:
            observation (np.ndarray): the initial observation.
        """
        self.interface = deepcopy(self._init_snapshot)
        self._prev_interface = deepcopy(self._init_snapshot)
        return self.observation_from_state()

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
        raise NotImplementedError

    def observation_from_state(self) -> Dict[str, np.ndarray]:
        """ Construct an environment observation from the state of the
        simulator

        Returns:
            observation (Dict[str, np.ndarray]): an environment
                observation generated from the simulation state
        """
        raise NotImplementedError

    def reward_from_state(self) -> float:
        """ Calculate a reward from the state of the simulator

        Returns:
            reward (float): a reward generated from the simulation
            state
        """
        raise NotImplementedError

    def done_from_state(self) -> bool:
        """ Determine if the simulation is done from the state of the
        simulator

        Returns:
            done (bool): True if the simulation is done, False if not
        """
        raise NotImplementedError

    def info_from_state(self) -> Dict[Any, Any]:
        """ Give information about the environment using the state of
        the simulator

        Returns:
            info (dict): dict of environment information
        """
        raise NotImplementedError
