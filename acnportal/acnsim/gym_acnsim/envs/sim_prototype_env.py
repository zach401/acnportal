import gym
import math
from gym import spaces
from acnsim.sites import caltech_acn
import copy

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
		self.total_time = self.interface.get_last_predicted_timestamp()
		num_evses = self.interface.get_num_evses()

		# Action space is the set of possible schedules (0 - 32 A for each EVSE)
		self.action_space = spaces.MultiDiscrete([32] * len(num_evses))

		# Observation space contains vectors with the following info
		# arrival time
		arrival_space = spaces.Box(low=0, high=self.total_time, shape=(32,), dtype=int32)
		# departure time
		departure_space = spaces.Box(low=0, high=self.total_time, shape=(32,), dtype=int32)
		# remaining amp-period demand
		remaining_demand_space = spaces.Box(low=0, high=self.total_time*32, shape=(32,), dtype=int32)
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
		new_schedule = {self.interface.get_evse_list[i]: action[i].to_list() for i in range(len(action))}
        done = self.interface.step(new_schedule)
        observation = self._state_to_obs()
        reward = self._reward_from_state()
        info = {}
        return observation, reward, done, info


    def _state_to_obs(self):
    	curr_obs = {}
    	curr_obs['arrivals'] = np.array([evse.ev.arrival if evse.ev is not None else -1 for evse in self.interface.get_evse_list()])
    	curr_obs['departures'] = np.array([evse.ev.departure if evse.ev is not None else -1 for evse in self.interface.get_evse_list()])
    	curr_obs['demand'] = np.array([math.ceil(self.interface.remaining_amp_periods(evse.ev)) if evse.ev is not None else -1 for evse in self.interface.get_evse_list()])
    	curr_obs['timestep'] = self.interface.current_time

    def _reward_from_state(self):
    	return self.interface.last_energy_delivered()

	def reset(self):
		pass

	def render(self):
		pass