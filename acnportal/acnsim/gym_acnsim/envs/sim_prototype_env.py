import gym
import math
import numpy as np
from gym import spaces
import copy

from copy import deepcopy
import random

from acnportal import acnsim
from acnportal import algorithms
from acnportal.acnsim import events
from acnportal.acnsim import models
from acnportal.acnsim import gym_acnsim

from gym.wrappers import FlattenDictWrapper
from gym.wrappers import ClipAction

class BaseSimEnv(gym.Env):
    """ Abstract base class meant to be inherited from to implement new ACN-Sim Environments.

    Subclasses must implement the following methods:
        _action_to_schedule
        _observation_from_state
        _reward_from_state
        _done_from_state

    Subclasses must also specify observation_space and action_space, either as class
    or instance variables.

    Optionally, subclasses may implement _info_from_state, which here returns an empty dict.
    
    Subclasses may override __init__, step, and reset functions.

    Currently, no render function is implemented, though this function is not required
    for internal functionality.

    Attributes:
        interface (GymInterface): an interface to a simulation to be stepped by this environment.
        init_snapshot (GymInterface): a deep copy of the initial interface, used for environment resets.
        prev_interface (GymInterface): a deep copy of the interface at the previous time step; used for calculating action rewards.
        action (object): the action taken by the agent in this agent-environment loop iteration
    """

    def __init__(self, interface):
        self.interface = interface
        self.init_snapshot = copy.deepcopy(interface)
        # TODO: having prev_interface functionality slows down stepping as each step makes a copy of the entire simulation
        self.prev_interface = copy.deepcopy(interface)
        self.action = None

    def step(self, action):
        """ Step the simulation one timestep with an agent's action.

        Accepts an action and returns a tuple (observation, reward, done, info).
        
        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # TODO: We can remove action as an input ot action_to_schedule and use instance var isntead
        self.action = action
        schedule = self._action_to_schedule(action)
        
        self.prev_interface = copy.deepcopy(self.interface)
        self.interface.step(schedule)
        
        observation = self._observation_from_state()
        reward = self._reward_from_state()
        done = self._done_from_state()
        info = self._info_from_state()

        return observation, reward, done, info

    def reset(self):
        """Resets the state of the simulation and returns an initial observation.
        Resetting is done by setting the interface to the simulation to an interface
        to the simulation in its initial state.
        
        Returns:
            observation (object): the initial observation.
        """
        self.interface = copy.deepcopy(self.init_snapshot)
        self.prev_interface = copy.deepcopy(self.init_snapshot)
        return self._observation_from_state()

    def _action_to_schedule(self, action):
        """ Convert an agent action to a schedule to be input to the simulator.

        Args:
            action (object): an action provided by the agent.

        Returns:
            schedule (Dict[str, List[number]]): Dictionary mappding station ids to a schedule of pilot signals.
        """
        raise NotImplementedError

    def _observation_from_state(self):
        """ Construct an environment observation from the state of the simulator

        Returns:
            observation (object): an environment observation generated from the simulation state
        """
        # TODO: should always copy over previous and current interfaces
        raise NotImplementedError

    def _reward_from_state(self):
        """ Calculate a reward from the state of the simulator

        Returns:
            reward (float): a reward generated from the simulation state
        """
        raise NotImplementedError

    def _done_from_state(self):
        """ Determine if the simulation is done from the state of the simulator

        Returns:
            done (bool): True if the simulation is done, False if not
        """
        raise NotImplementedError

    def _info_from_state(self):
        """ Give information about the environment using the state of the simulator

        Returns:
            info (dict): dict of environment information
        """
        return {}

class DefaultSimEnv(BaseSimEnv):
    """ A simulator environment with the following characteristics:

    The action and observation spaces are continuous.

    An action in this environment is a pilot signal for each EVSE, within the minimum and maximum
    EVSE rates.

    An observation is a dict consisting of fields (times are 1-indexed in
    the observations):
        arrivals: arrival time of the EV at each EVSE (or 0 if there's no EV plugged in)
        departures: departure time of the EV at each EVSE (or 0 if there's no EV plugged in)
        demand: energy demand of the EV at each EVSE (unoccupied EVSEs have demand 0)
        constraint_matrix: matrix of aggregate current coefficients
        magnitudes: magnitude vector constraining aggregate currents
        timestep: timestep of the simulation

    The reward is calculated as follows:
        If no constraints (on the network or on the EVSEs) were violated by the action,
        a reward equal to the total charge delivered (in A) is returned
        If any constraint violation occurred, a negative reward equal to the magnitude of the violation is returned.
        Network constraint violations are scaled by the number of EVs
        Finally, a user-input reward function is added to the total reward.

    The simulation is considered done if the event queue is empty.
    """
    def __init__(self, interface, reward_func=None):
        """ Initialize this environment. Every Sim environment needs an interface to a simulator which
        runs an iteration every time the environment is stepped.

        Args:
            interface (acnsim.GymInterface): OpenAI Interface with ACN simulation for this environment to use.
            reward_func (acnsim.GymInterface -> number): A function which takes no arguments and returns a number.
        """
        super().__init__(interface)

        # Get parameters that constrain the action/observation spaces
        self.num_evses = self.interface.num_evses
        self.min_rates = np.array([evse.min_rate for evse in self.interface.evse_list])
        self.max_rates = np.array([evse.max_rate for evse in self.interface.evse_list])
        constraint_matrix, magnitudes = self.interface.network_constraints

        # Some baselines require zero-centering; subtract this offset from actions to do this
        # TODO: this would be better as an action wrapper
        self.rate_offset_array = (self.max_rates + self.min_rates) / 2
        
        if reward_func is None:
            def reward_func(self): return 0
        self.reward_func = reward_func

        # Action space is the set of possible schedules (0 - 32 A for each EVSE)
        # Recentered about 0
        self.action_space = spaces.Box(low=self.min_rates-self.rate_offset_array, high=self.max_rates-self.rate_offset_array, dtype='float32')

        # Observation space contains vectors with the following info
        # arrival time
        arrival_space = spaces.Box(low=0, high=np.inf, shape=(self.num_evses,), dtype='float32')
        # departure time
        departure_space = spaces.Box(low=0, high=np.inf, shape=(self.num_evses,), dtype='float32')
        # remaining amp-period demand
        remaining_demand_space = spaces.Box(low=0, high=np.inf, shape=(self.num_evses,), dtype='float32')
        # constraint matrix (coefficients in aggregate currents)
        constraint_matrix_space = spaces.Box(low=-1*np.inf, high=np.inf, shape=constraint_matrix.shape, dtype='float32')
        # magnitude vector (upper limits on aggregate currents)
        magnitudes_space = spaces.Box(low=-1*np.inf, high=np.inf, shape=magnitudes.shape, dtype='float32')
        # current sim timestep
        timestep_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype='float32')

        # Total observation space is a Dict space of the subspaces
        # TODO: plurals for keys
        self.observation_dict = {
            'arrivals': arrival_space,
            'departures': departure_space,
            'demand': remaining_demand_space,
            'constraint_matrix': constraint_matrix_space,
            'magnitudes': magnitudes_space,
            'timestep': timestep_space
        }
        self.observation_space = spaces.Dict(self.observation_dict)

        # Portion of the observation that is independent of agent action
        self.static_obs = {'constraint_matrix': constraint_matrix, 'magnitudes': magnitudes}

    def _action_to_schedule(self, action):
        """ Convert an agent action to a schedule to be input to the simulator.

        Args:
            action (object): an action provided by the agent.

        Returns:
            schedule (Dict[str, List[number]]): Dictionary mappding station ids to a schedule of pilot signals.
        """
        action = action + self.rate_offset_array
        new_schedule = {self.interface.station_ids[i]: [action[i]] for i in range(len(action))}
        return new_schedule

    def _observation_from_state(self):
        """ Construct an environment observation from the state of the simulator

        Returns:
            observation (object): an environment observation generated from the simulation state
        """
        # Note constraint_matrix and magnitudes don't change in time, so are
        # stored in self.static_obs
        curr_obs = self.static_obs

        # Time-like observations are 1 indexed as 0 means no EV is plugged in.
        curr_obs['arrivals'] = np.array([evse.ev.arrival + 1 if evse.ev is not None else 0 for evse in self.interface.evse_list])
        curr_obs['departures'] = np.array([evse.ev.departure + 1 if evse.ev is not None else 0 for evse in self.interface.evse_list])
        curr_obs['demand'] = np.array([self.interface.remaining_amp_periods(evse.ev) if evse.ev is not None else 0 for evse in self.interface.evse_list])
        curr_obs['timestep'] = self.interface.current_time + 1

        return curr_obs

    def _reward_from_state(self):
        """ Calculate a reward from the state of the simulator

        Returns:
            reward (float): a reward generated from the simulation state
        """
        action = self.action + self.rate_offset_array
        # EVSE violation is the (negative) sum of violations of individual EVSE constraints,
        # e.g. min/max charging rates.
        evse_violation = 0

        # Unplugged EV violation is the (negative) sum of charge delivered to empty EVSEs
        unplugged_ev_violation = 0

        for i in range(len(self.interface.evse_list)):
            # If a single EVSE constraint is violated, a negative reward equal to the
            # magnitude of the violation is added to the total reward
            curr_evse = self.interface.evse_list[i]
            if action[i] < curr_evse.min_rate:
                evse_violation -= curr_evse.min_rate - action[i]
            if action[i] > curr_evse.max_rate:
                evse_violation -= action[i] - curr_evse.max_rate

            # If charge is attempted to be delivered to an evse with no ev,
            # this rate is subtracted from the reward.
            if self.interface.evse_list[i].ev is None:
                unplugged_ev_violation -= abs(action[i])

        # If a network constraint is violated, a negative reward equal to the abs
        # of the constraint violation, times the number of EVSEs, is added
        _, magnitudes = self.interface.network_constraints
        # Calculate aggregate currents for this charging schedule
        outvec = abs(self.interface.constraint_currents(
            np.array([[action[i]] for i in range(len(action))])))
        # Calculate violation of each individual constraint violation
        diffvec = np.array(
            [0 if outvec[i] <= magnitudes[i] 
                else outvec[i] - magnitudes[i] 
                    for i in range(len(outvec))])
        # Calculate total constraint violation, scaled by number of EVSEs
        constraint_violation = np.linalg.norm(diffvec) * self.num_evses * (-1)

        # If no violation penalties are incurred, reward for charge delivered
        charging_reward = 0
        if evse_violation == 0 and constraint_violation == 0:
            # TODO: currently only takes last actual charging rates, should take
            # all charging rates caused by this schedule
            # TODO: function for this that doesn't require private variable access
            charging_reward = np.sum(self.interface._simulator.charging_rates[:, self.interface.current_time-1])
            # TODO: there seems to be a problem with plugging in 2 evs at the same timestep having inaccurate len active evs
        # TODO: add options to toggle which rewards are included in the sum
        reward = charging_reward + evse_violation + constraint_violation  + unplugged_ev_violation + self.reward_func(self.interface)
        return reward

    def _done_from_state(self):
        """ Determine if the simulation is done from the state of the simulator

        Returns:
            done (bool): True if the simulation is done, False if not
        """
        return self.interface.is_done

class RebuildingEnv(DefaultSimEnv):
    """ A simulator environment that subclasses DefaultSimEnv, with the extra property
    that the entire simulation is rebuilt within the environment when __init__ or reset are called

    This is especially useful if the network or event queue have stochastic elements.
    """
    # TODO: reward_func or sim_gen_function, choose a convention
    def __init__(self, interface, reward_func=None, sim_gen_func=None):
        """ Initialize this environment. Every Sim environment needs an interface to a simulator which
        runs an iteration every time the environment is stepped.

        Args:
            interface (acnsim.GymInterface): OpenAI Gym Interface with ACN simulation for this environment to use.
            reward_func (-> number): A function which takes no arguments and returns a number.
            sim_gen_function (-> acnsim.GymInterface): function which returns a GymInterface to a generated simulator.
        """
        if sim_gen_func is None:
            def sim_gen_func(self): return self.init_snapshot
        else:
            self.sim_gen_func = sim_gen_func
        
        super().__init__(interface, reward_func=reward_func)

        temp_interface = self.sim_gen_func()
        self.interface = copy.deepcopy(temp_interface)
        self.prev_interface = copy.deepcopy(temp_interface)
        self.init_snapshot = copy.deepcopy(temp_interface)

    def reset(self):
        """ Resets the state of the simulation and returns an initial observation.
        Resetting is done by setting the interface to the simulation to an interface
        to the simulation in its initial state.
        
        Returns:
            observation (object): the initial observation.
        """
        temp_interface = self.sim_gen_func()
        self.interface = copy.deepcopy(temp_interface)
        self.prev_interface = copy.deepcopy(temp_interface)
        self.init_snapshot = copy.deepcopy(temp_interface)
        return self._observation_from_state()
