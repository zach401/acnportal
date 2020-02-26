import gym
import numpy as np
from gym import spaces
from copy import deepcopy
from .. import reward_functions as rf
from ..action_spaces import SimAction, zero_centered_single_charging_schedule
from ..observation import SimObservation
from .. import observation as obs
from ...interface import GymInterface, GymTrainingInterface
from typing import Optional, Dict, List, Any, Tuple, Callable


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
        _interface (GymInterface): An interface to a simulation to be
            stepped by this environment.
        _init_snapshot (GymInterface): A deep copy of the initial
            interface, used for environment resets.
        _prev_interface (GymInterface): A deep copy of the interface
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
    _interface: GymInterface
    _init_snapshot: GymInterface
    _prev_interface: GymInterface
    _action: Optional[np.ndarray]
    _schedule: Dict[str, List[float]]
    _observation: Optional[np.ndarray]
    _reward: Optional[float]
    _done: Optional[bool]
    _info: Optional[Dict[Any, Any]]

    def __init__(self, interface: GymInterface) -> None:
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
    def interface(self) -> GymInterface:
        return deepcopy(self._interface)

    @interface.setter
    def interface(self, new_interface: GymInterface) -> None:
        self._interface = new_interface

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
        self.observation = self.observation_from_state()
        self.reward = self.reward_from_state()
        self.done = self.done_from_state()
        self.info = self.info_from_state()

    def store_previous_state(self) -> None:
        """ Store the previous state of the simulation in the
        prev_interface environment attribute.

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

    @staticmethod
    def info_from_state():
        """ Give information about the environment using the state of
        the simulator

        Returns:
            info (dict): dict of environment information
        """
        return {}


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
    observation_space: spaces.Space
    action_object: SimAction
    action_space: spaces.Space
    reward_functions: List[Callable[[BaseSimEnv], float]]

    def __init__(
            self,
            interface: GymInterface,
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
        self.observation_space = spaces.Dict({
            observation_object.name: observation_object.get_space(
                self.interface)
            for observation_object in observation_objects
        })

        self.action_object = action_object
        self.action_space = action_object.get_space(interface)

        self.reward_functions = reward_functions

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
            observation_object.name: observation_object.get_space(
                self.interface)
            for observation_object in self.observation_objects
        }

    def reward_from_state(self) -> float:
        """ Calculate a reward from the state of the simulator

        Returns:
            reward (float): a reward generated from the simulation
                state
        """
        return sum([
            reward_func(self) for reward_func in self.reward_functions])

    def done_from_state(self) -> bool:
        """ Determine if the simulation is done from the state of the
        simulator

        Returns:
            done (bool): True if the simulation is done, False if not
        """
        return self.interface.is_done


def make_default_sim_env(interface: GymInterface) -> CustomSimEnv:
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
    observation_objects: List[SimObservation] = [
        obs.arrival_observation(),
        obs.departure_observation(),
        obs.remaining_demand_observation(),
        obs.constraint_matrix_observation(),
        obs.magnitudes_observation(),
        obs.timestep_observation()
    ]
    action_object: SimAction = zero_centered_single_charging_schedule()
    reward_functions: List[Callable[[BaseSimEnv], float]] = [
        rf.evse_violation,
        rf.unplugged_ev_violation,
        rf.constraint_violation,
        rf.hard_charging_reward
    ]
    return CustomSimEnv(
        interface, observation_objects, action_object, reward_functions)


class RebuildingEnv(CustomSimEnv):
    """ A simulator environment that subclasses CustomSimEnv, with
    the extra property that the entire simulation is rebuilt within 
    the environment when __init__ or reset are called

    This is especially useful if the network or event queue have 
    stochastic elements.
    """
    def __init__(
            self,
            interface: GymInterface,
            observation_objects: List[SimObservation],
            action_object: SimAction,
            reward_functions: List[Callable[[BaseSimEnv], float]],
            sim_gen_function: Optional[Callable[[], GymInterface]] = None
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
            sim_gen_function (Optional[Callable[[], GymInterface]]):
                Function which returns a GymInterface to a generated
                simulator.
        """
        if sim_gen_function is None:
            def sim_gen_function():
                return self.init_snapshot
        self.sim_gen_function = sim_gen_function

        super().__init__(interface,
                         observation_objects,
                         action_object,
                         reward_functions)

        temp_interface = self.sim_gen_function()
        self.interface = deepcopy(temp_interface)
        self.prev_interface = deepcopy(temp_interface)
        self.init_snapshot = deepcopy(temp_interface)

    def reset(self) -> Dict[str, np.ndarray]:
        """ Resets the state of the simulation and returns an initial 
        observation. Resetting is done by setting the interface to 
        the simulation to an interface to the simulation in its 
        initial state.
        
        Returns:
            observation (np.ndarray): the initial observation.
        """
        temp_interface = self.sim_gen_function()
        self.interface = deepcopy(temp_interface)
        self.prev_interface = deepcopy(temp_interface)
        self.init_snapshot = deepcopy(temp_interface)
        return self.observation_from_state()

    def render(self, mode='human'):
        """ Renders the environment. Implements gym.Env.render(). """
        raise NotImplementedError
