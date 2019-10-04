import gym
import math
import numpy as np
from gym import spaces
import copy

class SimpleSimEnv(gym.Env):
    # Action space: set of possible schedules passed to the network
    # These do not have the constraints applied to them
    # For now, assume single-timestep algorithmic horizon
    # Set of observations of the network. These observations include the following info:
    # remaining demand
    # Action space: set of possible schedules passed to the network (continuous)
    # These do not have the constraints applied to them
    # For now, assume single-timestep algorithmic horizon
    # Set of observations of the network. These observations include the following info:
    # Current arrival, departure, and energy demand remaining for all EVs plugged in

    def __init__(self, interface):
        """ Initialize this environment. Every Sim environment needs an interface to a simulator which
        runs an iteration every time the environment is stepped.

        Args:
            interface (acnsim.OpenAIInterface): OpenAI Interface with ACN simulation for this environment to use.
        """
        self.interface = interface
        self.init_snapshot = copy.deepcopy(interface)
        self.total_time = self.interface.last_predicted_timestamp
        num_evses = self.interface.num_evses
        min_rates = np.array([evse.min_rate for evse in self.interface.evse_list])
        max_rates = np.array([evse.max_rate for evse in self.interface.evse_list])

        # Action space is the set of possible schedules (0 - 32 A for each EVSE)
        self.action_space = spaces.Box(low=min_rates, high=max_rates, dtype=np.float32)

        self.observation_space = spaces.Box(low=np.zeros((num_evses,)), high=max_rates*self.total_time, dtype='int32')

        # There's no initial observation
        self.obs = self._state_to_obs()

    def step(self, action):
        new_schedule = {self.interface.station_ids[i]: [action[i]] for i in range(len(action))}
        done = self.interface.step(new_schedule)
        observation = self._state_to_obs()
        reward = self._reward_from_state(action)
        info = {}
        return observation, reward, done, info


    def _state_to_obs(self):
        return np.array([self.interface.remaining_amp_periods(evse.ev) if evse.ev is not None else -1 for evse in self.interface.evse_list])
        
    def _reward_from_state(self, action):
        evse_violation = 0
        unplugged_ev_violation = 0
        for i in range(len(self.interface.evse_list)):
            # If a single EVSE constraint is violated, a negative reward equal to the
            # abs of the violation is added to the total reward
            curr_evse = self.interface.evse_list[i]
            if action[i] < curr_evse.min_rate:
                evse_violation -= curr_evse.min_rate - action[i]
            if action[i] > curr_evse.max_rate:
                evse_violation -= action[i] - curr_evse.max_rate

            # If rate is attempted to be delivered to an evse with no ev,
            # this rate is subtracted from the reward.
            if self.interface.evse_list[i].ev is None:
                unplugged_ev_violation -= abs(action[i])

        # If a network constraint is violated, a negative reward equal to the abs
        # of the constraint violation, times the number of EVSEs, is added
        _, magnitudes = self.interface.network_constraints
        outvec = abs(self.interface.constraint_currents(
            np.array([[action[i]] for i in range(len(action))])))
        diffvec = np.array(
            [0 if outvec[i] <= magnitudes[i] 
                else outvec[i] - magnitudes[i] 
                    for i in range(len(outvec))])
        constraint_violation = np.linalg.norm(diffvec) * self.interface.num_evses * (-1)

        # If no violation penalties are incurred, reward for charge delivered
        # Check if the schedule history was updated with this schedule
        # (meaning schedule was feasible)
        charging_reward = 0
        lap = np.array([self.interface.last_applied_pilot_signals[station_id] if station_id in self.interface.last_applied_pilot_signals else action[self.interface.station_ids.index(station_id)] for station_id in self.interface.station_ids])
        lap_keys = list(self.interface.last_applied_pilot_signals.keys())
        relevant_actions = np.array([[action[i]] if self.interface.station_ids[i] in lap_keys else 0])
        if np.allclose(lap, relevant_actions):
            assert evse_violation == 0
            assert constraint_violation == 0
            # TODO: currently only takes last actual charging rates, should take
            # all charging rates caused by this schedule
            charging_reward = sum(list(self.interface.last_actual_charging_rate.values()))

        return evse_violation + constraint_violation + charging_reward + unplugged_ev_violation

# TODO: random event generator within environment

    def reset(self):
        self.interface = copy.deepcopy(self.init_snapshot)
        observation = self._state_to_obs()
        return self._state_to_obs()

    def render(self):
        # TODO: render env
        pass

class SimPrototypeEnv(gym.Env):
    # Action space: set of possible schedules passed to the network
    # These do not have the constraints applied to them
    # For now, assume single-timestep algorithmic horizon
    # Set of observations of the network. These observations include the following info:
    # Current date and time
    # Current arrival, departure, and energy demand remaining for all EVs plugged in

    def __init__(self, interface):
        """ Initialize this environment. Every Sim environment needs an interface to a simulator which
        runs an iteration every time the environment is stepped.

        Args:
            interface (acnsim.OpenAIInterface): OpenAI Interface with ACN simulation for this environment to use.
        """
        self.interface = interface
        self.init_snapshot = copy.deepcopy(interface)
        self.total_time = self.interface.last_predicted_timestamp
        num_evses = self.interface.num_evses

        # Action space is the set of possible schedules (0 - 32 A for each EVSE)
        self.action_space = spaces.MultiDiscrete([32] * num_evses)

        # Observation space contains vectors with the following info
        # arrival time
        arrival_space = spaces.Box(low=0, high=self.total_time, shape=(32,), dtype='int32')
        # departure time
        departure_space = spaces.Box(low=0, high=self.total_time, shape=(32,), dtype='int32')
        # remaining amp-period demand
        remaining_demand_space = spaces.Box(low=0, high=self.total_time*32, shape=(32,), dtype='int32')
        # current sim timestep
        timestep_space = spaces.Discrete(self.total_time+1)
        
        # Total observation space is a Dict space of the subspaces
        self.observation_space = spaces.Dict({
            'arrivals': arrival_space,
            'departures': departure_space,
            'demand': remaining_demand_space,
            'timestep': timestep_space
        })

        # There's no initial observation
        self.obs = None

    def step(self, action):
        new_schedule = {self.interface.station_ids[i]: [action[i]] for i in range(len(action))}
        done = self.interface.step(new_schedule)
        observation = self._state_to_obs()
        reward = self._reward_from_state()
        info = {}
        return observation, reward, done, info


    def _state_to_obs(self):
        curr_obs = {}
        curr_obs['arrivals'] = np.array([evse.ev.arrival if evse.ev is not None else -1 for evse in self.interface.evse_list])
        curr_obs['departures'] = np.array([evse.ev.departure if evse.ev is not None else -1 for evse in self.interface.evse_list])
        curr_obs['demand'] = np.array([math.ceil(self.interface.remaining_amp_periods(evse.ev)) if evse.ev is not None else -1 for evse in self.interface.evse_list])
        curr_obs['timestep'] = self.interface.current_time

    def _reward_from_state(self):
        return self.interface.last_energy_delivered()

    def reset(self):
        pass

    def render(self):
        pass

# TODO: For new environments, can initialize env with input function
class ContSimPrototypeEnv(gym.Env):
    # Action space: set of possible schedules passed to the network (continuous)
    # These do not have the constraints applied to them
    # For now, assume single-timestep algorithmic horizon
    # Set of observations of the network. These observations include the following info:
    # Current arrival, departure, and energy demand remaining for all EVs plugged in

    def __init__(self, interface):
        """ Initialize this environment. Every Sim environment needs an interface to a simulator which
        runs an iteration every time the environment is stepped.

        Args:
            interface (acnsim.OpenAIInterface): OpenAI Interface with ACN simulation for this environment to use.
        """
        self.interface = interface
        self.init_snapshot = copy.deepcopy(interface)
        self.total_time = self.interface.last_predicted_timestamp
        num_evses = self.interface.num_evses
        min_rates = np.array([evse.min_rate for evse in self.interface.evse_list])
        max_rates = np.array([evse.max_rate for evse in self.interface.evse_list])

        # Action space is the set of possible schedules (0 - 32 A for each EVSE)
        self.action_space = spaces.Box(low=min_rates, high=max_rates, dtype=np.float32)

        # Observation space contains vectors with the following info
        # arrival time
        arrival_space = spaces.Box(low=0, high=self.total_time, shape=(num_evses,), dtype='int32')
        # departure time
        departure_space = spaces.Box(low=0, high=self.total_time, shape=(num_evses,), dtype='int32')
        # remaining amp-period demand
        # TODO: This assumes infeasible demands won't be input. Filter this out?
        remaining_demand_space = spaces.Box(low=np.zeros((num_evses,)), high=max_rates*self.total_time, dtype='int32')
        # current sim timestep
        timestep_space = spaces.Box(low=0, high=self.total_time+1, shape=(1,), dtype='int32')
        
        # Total observation space is a Dict space of the subspaces
        self.observation_space = spaces.Dict({
            'arrivals': arrival_space,
            'departures': departure_space,
            'demand': remaining_demand_space,
            'timestep': timestep_space
        })

        # There's no initial observation
        self.obs = None

    def step(self, action):
        new_schedule = {self.interface.station_ids[i]: [action[i]] for i in range(len(action))}
        done = self.interface.step(new_schedule)
        observation = self._state_to_obs()
        reward = self._reward_from_state(action)
        info = {}
        print(action)
        return observation, reward, done, info


    def _state_to_obs(self):
        curr_obs = {}
        curr_obs['arrivals'] = np.array([evse.ev.arrival if evse.ev is not None else -1 for evse in self.interface.evse_list])
        curr_obs['departures'] = np.array([evse.ev.departure if evse.ev is not None else -1 for evse in self.interface.evse_list])
        curr_obs['demand'] = np.array([math.ceil(self.interface.remaining_amp_periods(evse.ev)) if evse.ev is not None else -1 for evse in self.interface.evse_list])
        curr_obs['timestep'] = self.interface.current_time
        return curr_obs
        
    def _reward_from_state(self, action):
        evse_violation = 0
        unplugged_ev_violation = 0
        for i in range(len(self.interface.evse_list)):
            # If a single EVSE constraint is violated, a negative reward equal to the
            # abs of the violation is added to the total reward
            curr_evse = self.interface.evse_list[i]
            if action[i] < curr_evse.min_rate:
                evse_violation -= curr_evse.min_rate - action[i]
            if action[i] > curr_evse.max_rate:
                evse_violation -= action[i] - curr_evse.max_rate

            # If rate is attempted to be delivered to an evse with no ev,
            # this rate is subtracted from the reward.
            if self.interface.evse_list[i] not in self.interface.active_evs:
                unplugged_ev_violation -= abs(action[i])

        # If a network constraint is violated, a negative reward equal to the abs
        # of the constraint violation, times the number of EVSEs, is added
        _, magnitudes = self.interface.network_constraints
        outvec = abs(self.interface.constraint_currents(
            np.array([[action[i]] for i in range(len(action))])))
        diffvec = np.array(
            [0 if outvec[i] <= magnitudes[i] 
                else outvec[i] - magnitudes[i] 
                    for i in range(len(outvec))])
        constraint_violation = np.linalg.norm(diffvec) * self.interface.num_evses * (-1)

        # If no violation penalties are incurred, reward for charge delivered
        # Check if the schedule history was updated with this schedule
        # (meaning schedule was feasible)
        charging_reward = 0
        lap = np.array([self.interface.last_applied_pilot_signals[station_id] if station_id in self.interface.last_applied_pilot_signals else action[self.interface.station_ids.index(station_id)] for station_id in self.interface.station_ids])
        lap_keys = list(self.interface.last_applied_pilot_signals.keys())
        relevant_actions = np.array([[action[i]] if self.interface.station_ids[i] in lap_keys else 0])
        if np.allclose(lap, relevant_actions):
            assert evse_violation == 0
            assert constraint_violation == 0
            # TODO: currently only takes last actual charging rates, should take
            # all charging rates caused by this schedule
            charging_reward = sum(list(self.interface.last_actual_charging_rate.values()))

        return evse_violation + constraint_violation + charging_reward

# TODO: random event generator within environment

    def reset(self):
        self.interface = self.init_snapshot
        observation = self._state_to_obs()
        assert isinstance(observation, dict)
        return self._state_to_obs()

    def render(self):
        pass