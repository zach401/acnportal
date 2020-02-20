# -- Run Simulation ----------------------------------------------------------------------------------------------------
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import numpy as np

from acnportal import acnsim
from acnportal import algorithms
from acnportal.acnsim import events
from acnportal.acnsim import models
from acnportal.acnsim import gym_acnsim
import gym
from gym.wrappers import FlattenDictWrapper
from gym.wrappers import ClipAction

from spinup import sac, vpg
from stable_baselines.common.policies import MlpPolicy as CommonMlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
from stable_baselines.td3.policies import MlpPolicy as TD3MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

from stable_baselines import SAC, PPO2, DDPG, TD3
from datetime import datetime
import pytz
import matplotlib.pyplot as plt

def random_plugin(num, time_limit, evse):
    """ Returns a list of num random plugin events occuring anytime from time 0 to time_limit
    Each plugin has a random arrival and departure under the time limit, and a satisfiable
    requested energy assuming no other cars plugged in.
    The plugins occur for a single EVSE
    """
    out_event_lst = [None] * num
    times = []
    i = 0
    while i < 2*num:
        rnum = random.randint(0, time_limit)
        if rnum not in times:
            times.append(rnum)
            i += 1
    times = sorted(times)
    battery = models.Battery(100, 0, 100)
    for i in range(num):
        arrival_time = times[2*i]
        departure_time = times[2*i+1]
        requested_energy = (departure_time - arrival_time) / 60 * 32 * 220 / 2
        ev = models.EV(arrival_time, departure_time, requested_energy, evse, '', battery)
        out_event_lst[i] = events.PluginEvent(arrival_time, ev)
    return out_event_lst

def sim_gen_func():
    """
    Initializes a simulation with random events on a 4 EVSE, 1 phase, 1 constraint ACN
    """
    timezone = pytz.timezone('America/Los_Angeles')
    start = timezone.localize(datetime(2018, 9, 5))
    voltage = 220
    period = 1

    # Make random event queue
    cn = acnsim.sites.simple_acn(["SA-001", "SA-002", "SA-003", "SA-004"], aggregate_cap=80*208/1000)
    event_list = []
    for station_id in cn.station_ids:
        event_list.extend(random_plugin(10, 100, station_id))
    event_queue = events.EventQueue(event_list)

    # Placeholder algorithm
    schrl = algorithms.GymAlgorithm()

    # Simulation to be wrapped
    simrl = acnsim.Simulator(deepcopy(cn), schrl, deepcopy(event_queue), start, period=period, verbose=False)
    return schrl.interface

def reward_func(interface):
    """
    A reward function that downshifts charging reward (acting roughly like a unused capacity penalty)
    """
    min_rates = np.array([evse.min_rate for evse in interface.evse_list])
    max_rates = np.array([evse.max_rate for evse in interface.evse_list])
    rate_offset_array = (max_rates + min_rates) / 2
    penalty = 0
    for evse in interface.active_evs:
        penalty -= np.mean(rate_offset_array)
    return penalty


def env_func(interface, reward_func, sim_gen_func):
    """
    Generate an environment compatible with simple baselines (i.e. flatten dict obs space)
    """
    return FlattenDictWrapper(gym.make(id='rebuilding-acnsim-v0',
        interface=interface, reward_func=reward_func, sim_gen_func=sim_gen_func),
        ['arrivals', 'departures', 'demand', 'constraint_matrix', 'magnitudes', 'timestep'])

n_cpu = 1
env = DummyVecEnv([lambda: env_func(sim_gen_func(), reward_func, sim_gen_func) for i in range(n_cpu)])

model = PPO2(CommonMlpPolicy, env, verbose=2)
model.learn(total_timesteps=10000, log_interval=10)
model.save("PPO_test_big_rc"+str(datetime.now()))
# ^ run model = PPO2.load("PPO_test_big_rc"+[DATE SAVED]) in the directory where the model is stored


# See how the model behaves
obs = env.reset()
dones = [False]
while not dones[0]:
    action, _states = model.predict(obs)
    print("obs:", obs)
    obs, rewards, dones, info = env.step(action)
    print("action:", action)
    print("rewards:", rewards)
    print()

# # the noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# param_noise = None
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

# # model = DDPG(DDPGMlpPolicy, env, verbose=2, action_noise=action_noise)