import gym
import numpy as np
from gym import spaces
import copy
from .. import reward_functions as rf


class BaseSimEnv(gym.Env):
    """ Abstract base class meant to be inherited from to implement
    new ACN-Sim Environments.

    Subclasses must implement the following methods:
        _action_to_schedule
        _observation_from_state
        _reward_from_state
        _done_from_state

    Subclasses must also specify observation_space and action_space,
    either as class or instance variables.

    Optionally, subclasses may implement _info_from_state, which here
    returns an empty dict.
    
    Subclasses may override __init__, step, and reset functions.

    Currently, no render function is implemented, though this function
    is not required for internal functionality.

    Attributes:
        interface (GymInterface): An interface to a simulation to be
            stepped by this environment.
        init_snapshot (GymInterface): A deep copy of the initial
            interface, used for environment resets.
        prev_interface (GymInterface): A deep copy of the interface
            at the previous time step; used for calculating action 
            rewards.
        action (object): The action taken by the agent in this
            agent-environment loop iteration.
        schedule (Dict[str, List[number]]): Dictionary mapping
            station ids to a schedule of pilot signals.
        observation (object): The observation given to the agent in
            this agent-environment loop iteration.
        done (object): An object representing whether or not the
            execution of the environment is complete.
        info (object): An object that gives info about the environment.
    """
    def __init__(self, interface):
        self.interface = interface
        self.init_snapshot = copy.deepcopy(interface)
        self.prev_interface = copy.deepcopy(interface)
        self.action = None
        self.schedule = {}
        self.observation = None
        self.done = None
        self.info = None

    def step(self, action):
        """ Step the simulation one timestep with an agent's action.

        Accepts an action and returns a tuple (observation, reward,
        done, info).

        Implements gym.Env.step()

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current 
                environment
            reward (float) : amount of reward returned after previous 
                action
            done (bool): whether the episode has ended, in which case 
                further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information 
                (helpful for debugging, and sometimes learning)
        """
        self.action = action
        self.schedule = self._action_to_schedule()
        
        self.prev_interface = copy.deepcopy(self.interface)
        self.interface.step(self.schedule)
        
        observation = self._observation_from_state()
        reward = self._reward_from_state()
        done = self._done_from_state()
        info = self._info_from_state()

        return observation, reward, done, info

    def reset(self):
        """ Resets the state of the simulation and returns an initial
        observation. Resetting is done by setting the interface to the
        simulation to an interface to the simulation in its initial 
        state.

        Implements gym.Env.reset()

        Returns:
            observation (object): the initial observation.
        """
        self.interface = copy.deepcopy(self.init_snapshot)
        self.prev_interface = copy.deepcopy(self.init_snapshot)
        return self._observation_from_state()

    def render(self, mode='human'):
        """ Renders the environment. Implements gym.Env.render(). """
        raise NotImplementedError

    def _action_to_schedule(self):
        """ Convert an agent action to a schedule to be input to the
        simulator.

        Returns:
            schedule (Dict[str, List[number]]): Dictionary mappding 
                station ids to a schedule of pilot signals.
        """
        raise NotImplementedError

    def _observation_from_state(self):
        """ Construct an environment observation from the state of the
        simulator

        Returns:
            observation (object): an environment observation 
                generated from the simulation state
        """
        raise NotImplementedError

    def _reward_from_state(self):
        """ Calculate a reward from the state of the simulator

        Returns:
            reward (float): a reward generated from the simulation 
            state
        """
        raise NotImplementedError

    def _done_from_state(self):
        """ Determine if the simulation is done from the state of the
        simulator

        Returns:
            done (bool): True if the simulation is done, False if not
        """
        raise NotImplementedError

    def _info_from_state(self):
        """ Give information about the environment using the state of
        the simulator

        Returns:
            info (dict): dict of environment information
        """
        return {}


class DefaultSimEnv(BaseSimEnv):
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
    def __init__(self, interface, reward_funcs=None):
        """ Initialize this environment. Every Sim environment needs 
        an interface to a simulator which runs an iteration every 
        time the environment is stepped.

        Args:
            interface (acnsim.GymInterface): OpenAI Interface with 
                ACN simulation for this environment to use.
            reward_funcs (List[BaseSimEnv -> number]): 
                List of functions which take as input a BaseSimEnv
                instance and return a number.
        """
        super().__init__(interface)

        # Get parameters that constrain the action/observation spaces
        self.num_evses = len(self.interface.station_ids)
        self.min_rates = np.array(
            [evse.min_rate for evse in self.interface.evse_list])
        self.max_rates = np.array(
            [evse.max_rate for evse in self.interface.evse_list])
        constraint_obj = self.interface.get_constraints()
        constraint_matrix, magnitudes = \
            constraint_obj.constraint_matrix, constraint_obj.magnitudes

        # Some baselines require zero-centering; subtract this offset 
        # from actions to do this
        self.rate_offset_array = (self.max_rates + self.min_rates) / 2
        
        if reward_funcs is None:
            self.reward_funcs = [
                rf.evse_violation,
                rf.unplugged_ev_violation,
                rf.constraint_violation,
                rf.hard_charging_reward
            ]
        self.reward_funcs = reward_funcs

        # Action space is the set of possible schedules (0 - 32 A for 
        # each EVSE)
        # Re-centered about 0
        self.action_space = spaces.Box(
            low=self.min_rates-self.rate_offset_array, 
            high=self.max_rates-self.rate_offset_array, 
            dtype='float32')

        # Observation space contains vectors with the following info
        # arrival time
        arrival_space = spaces.Box(
            low=0, high=np.inf, shape=(self.num_evses,), 
            dtype='float32')
        # departure time
        departure_space = spaces.Box(
            low=0, high=np.inf, shape=(self.num_evses,), dtype='float32')
        # remaining amp-period demand
        remaining_demand_space = spaces.Box(
            low=0, high=np.inf, shape=(self.num_evses,), dtype='float32')
        # constraint matrix (coefficients in aggregate currents)
        constraint_matrix_space = spaces.Box(
            low=-1*np.inf, high=np.inf,
            shape=constraint_matrix.shape, dtype='float32'
        )
        # magnitude vector (upper limits on aggregate currents)
        magnitudes_space = spaces.Box(
            low=-1*np.inf, high=np.inf,
            shape=magnitudes.shape, dtype='float32'
        )
        # current sim timestep
        timestep_space = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype='float32')

        # Total observation space is a Dict space of the subspaces
        self.observation_dict = {
            'arrivals': arrival_space,
            'departures': departure_space,
            'demands': remaining_demand_space,
            'constraint_matrix': constraint_matrix_space,
            'magnitudes': magnitudes_space,
            'timestep': timestep_space
        }
        self.observation_space = spaces.Dict(self.observation_dict)

        # Portion of the observation that is independent of agent 
        # action
        self.static_obs = {'constraint_matrix': constraint_matrix,
                           'magnitudes': magnitudes}

    def render(self, mode='human'):
        """ Renders the environment. Implements gym.Env.render(). """
        raise NotImplementedError

    def _action_to_schedule(self):
        """ Convert an agent action to a schedule to be input to the 
        simulator.

        Returns:
            schedule (Dict[str, List[number]]): Dictionary mappding 
                station ids to a schedule of pilot signals.
        """
        offset_action = self.action + self.rate_offset_array
        new_schedule = {self.interface.station_ids[i]: [offset_action[i]]
                        for i in range(len(offset_action))}
        return new_schedule

    def _observation_from_state(self):
        """ Construct an environment observation from the state of 
        the simulator

        Returns:
            observation (object): an environment observation 
                generated from the simulation state
        """
        # Note constraint_matrix and magnitudes don't change in time, 
        # stored in self.static_obs
        curr_obs = self.static_obs

        # Time-like observations are 1 indexed as 0 means no EV is 
        # plugged in.
        curr_obs['arrivals'] = np.array([
            evse.ev.arrival + 1
            if evse.ev is not None
            else 0
            for evse in self.interface.evse_list
        ])

        curr_obs['departures'] = np.array([
            evse.ev.departure + 1
            if evse.ev is not None
            else 0
            for evse in self.interface.evse_list
        ])

        curr_obs['demand'] = np.array([
            self.interface.remaining_amp_periods(evse.ev)
            if evse.ev is not None
            else 0
            for evse in self.interface.evse_list
        ])

        curr_obs['timestep'] = self.interface.current_time + 1

        return curr_obs

    def _reward_from_state(self):
        """ Calculate a reward from the state of the simulator

        Returns:
            reward (float): a reward generated from the simulation 
                state
        """
        total_reward = 0
        for reward_func in self.reward_funcs:
            total_reward += reward_func(self)

        return total_reward

    def _done_from_state(self):
        """ Determine if the simulation is done from the state of the 
        simulator

        Returns:
            done (bool): True if the simulation is done, False if not
        """
        return self.interface.is_done


class RebuildingEnv(DefaultSimEnv):
    """ A simulator environment that subclasses DefaultSimEnv, with 
    the extra property that the entire simulation is rebuilt within 
    the environment when __init__ or reset are called

    This is especially useful if the network or event queue have 
    stochastic elements.
    """
    def __init__(self, interface, reward_funcs=None,
                 sim_gen_func=None, event_lst=None):
        """ Initialize this environment. Every Sim environment needs 
        an interface to a simulator which runs an iteration every 
        time the environment is stepped.

        Args:
            interface (acnsim.GymInterface): OpenAI Gym Interface 
                with ACN simulation for this environment to use.
            reward_funcs (List[-> number]): A list of functions which
                take no arguments and return a number.
            sim_gen_func (-> acnsim.GymInterface): function which
                returns a GymInterface to a generated simulator.
        """
        if sim_gen_func is None:
            def sim_gen_func(self, *args, **kwargs): 
                return self.init_snapshot
        else:
            self.sim_gen_func = sim_gen_func
            self.event_lst = event_lst
        
        super().__init__(interface, reward_funcs=reward_funcs)

        if self.event_lst is None:
            temp_interface = self.sim_gen_func()
        else:
            temp_interface = self.sim_gen_func(self.event_lst)
        self.interface = copy.deepcopy(temp_interface)
        self.prev_interface = copy.deepcopy(temp_interface)
        self.init_snapshot = copy.deepcopy(temp_interface)

    def reset(self):
        """ Resets the state of the simulation and returns an initial 
        observation. Resetting is done by setting the interface to 
        the simulation to an interface to the simulation in its 
        initial state.
        
        Returns:
            observation (object): the initial observation.
        """
        if self.event_lst is None:
            temp_interface = self.sim_gen_func()
        else:
            temp_interface = self.sim_gen_func(self.event_lst)
        self.interface = copy.deepcopy(temp_interface)
        self.prev_interface = copy.deepcopy(temp_interface)
        self.init_snapshot = copy.deepcopy(temp_interface)
        return self._observation_from_state()

    def render(self, mode='human'):
        """ Renders the environment. Implements gym.Env.render(). """
        raise NotImplementedError
