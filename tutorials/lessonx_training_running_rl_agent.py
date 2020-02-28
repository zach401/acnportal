"""
ACN-Sim Tutorial: Lesson X
Training and Running a Reinforcement Learning Agent on ACN-Sim
by Sunash Sharma
Last updated: 02/27/2020

It is strongly suggested that this tutorial is run in its own
environment (e.g. conda or pyenv), as it will require dependencies
not required by the rest of ACN-Portal.

In this lesson we will learn how to train a reinforcement learning (
RL) agent and run it using OpenAI Gym environments that wrap ACN-Sim.
For this example we will be using the stable-baselines proximal
policy optimization (PPO2) algorithm. As such, running this tutorial
requires the stable-baselines package.

"""

# If running in a new environment, such as Google Colab, run this first.

# !git clone https://github.com/zach401/acnportal.git
# !pip install acnportal/.[gym]
# !pip install stable-baselines

import random
from copy import deepcopy
from datetime import datetime
from typing import List

import gym
import pytz
from gym.wrappers import FlattenObservation
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

from acnportal import acnsim
from acnportal import algorithms
from acnportal.acnsim import events
from acnportal.acnsim import models


# For this lesson, we will use a simple example. Imagine we have a
# single charger, with EV's plugging in over a set duration. Each EV
# has a random arrival and departure time, requesting an amount of
# energy that equates to a laxity of $d/2$, where $d$ is staying
# duration. (i.e. we may charge at half the maximum rate for the entire
# staying time and deliver all the energy requested). First, let's make
# some functions to generate Simulation instances that simulate this
# scenario. We'll start by defining a function which generates random
# plugins for a single EVSE.
def random_plugin(num, time_limit, evse, laxity_ratio=1 / 2,
                  max_rate=32, voltage=208, period=1):
    """ Returns a list of num random plugin events occurring anytime
    from time 0 to time_limit. Each plugin has a random arrival and
    departure under the time limit, and a satisfiable requested
    energy assuming no other cars plugged in. Each EV has initial
    laxity equal to half the staying duration unless otherwise
    specified.
    
    The plugins occur for a single EVSE, whose maximal rate and
    voltage are assumed to be 32 A and  208 V, respectively, unless
    otherwise specified.

    Args:
        num (int): Number of random plugin
        time_limit (int):
        evse (str):
        laxity_ratio (float):
        max_rate (float):
        voltage (float):
        period (int):
    """
    out_event_lst: List[events.Event] = []
    times = []
    i = 0
    while i < 2 * num:
        random_timestep = random.randint(0, time_limit)
        if random_timestep not in times:
            times.append(random_timestep)
            i += 1
    times = sorted(times)
    battery = models.Battery(100, 0, 100)
    for i in range(num):
        arrival_time = times[2 * i]
        departure_time = times[2 * i + 1]
        requested_energy = (
                (departure_time - arrival_time) / (60 / period)
                * max_rate * voltage / (1 / laxity_ratio)
        )
        ev = models.EV(arrival_time, departure_time, requested_energy,
                       evse, f'rs-{evse}-{i}', battery)
        out_event_lst.append(events.PluginEvent(arrival_time, ev))
    return out_event_lst


# Since the above event generation is stochastic, we'll want to
# completely rebuild the simulation each time the environment is
# reset, so that the next simulation has a new event queue. As such,
# we will define a simulation generating function.
def sim_gen_function():
    """
    Initializes a simulation with random events on a 1 phase, 1
    constraint ACN (simple_acn), with 1 EVSE
    """
    timezone = pytz.timezone('America/Los_Angeles')
    start = timezone.localize(datetime(2018, 9, 5))
    period = 1

    # Make random event queue
    cn = acnsim.sites.simple_acn(['EVSE-001'],
                                 aggregate_cap=20 * 208 / 1000)
    event_list = []
    for station_id in cn.station_ids:
        event_list.extend(random_plugin(10, 100, station_id))
    event_queue = events.EventQueue(event_list)

    # For training, this algorithm isn't run. So, we need not provide
    # any arguments.
    schedule_rl = algorithms.GymAlgorithm()

    # Simulation to be wrapped
    _ = acnsim.Simulator(deepcopy(cn), schedule_rl,
                         deepcopy(event_queue), start, period=period,
                         verbose=False)
    return schedule_rl.interface


# ACN-Sim gym environments wrap an interface to an ACN-Sim
# Simulation. These environments allow for customizable observations,
# reward functions, and actions through the CustomSimEnv class,
# and for rebuilding through the RebuildingSimEnv class (the
# RebuildingSimEnv class extends the CustomSimEnv class, and so has
# all the customization features of the latter). As an example,
# let's make a rebuilding simulation environment with the following
# characteristics:
# 
# - Observations:
#     - Arrival times of all currently plugged-in EVs.
#     - Departure times of all currently plugged-in EVs.
#     - Remaining demand of all currently plugged-in EVs.
#     - Constraint matrix of the network.
#     - Limiting constraint magnitudes of the network.
#     - Current timestep of the simulation
# - Action:
#     - A zero-centered array of pilot signals. A 0 entry in the array
#       corresponds to a charging rate of 16 A.
# - Rewards:
#     - A negative reward for each amp of violation of individual EVSE
#       constraints.
#     - A negative reward for each amp of pilot signal delivered to an
#       EVSE with no EV plugged in.
#     - A negative reward for each amp of network constraint violation.
#     - A positive charging reward for each amp of charge delivered if
#       the above penalties are all 0.
# 
# The observations, actions, and rewards listed here are all already
# encoded in the `gym_acnsim` package; see the package documentation
# for more details. Broadly, each observation object has space and
# observation generating functions. Each action is an object with
# space and schedule generating functions. Each reward is a function
# of the environment, outputting a number. The environment described
# here is generated by the make_rebuilding_default_env function from
# the gym_acnsim object; see the code there for more details. The
# `gym_acnsim` package provides `'default-rebuilding-acnsim-v0'`,
# a registered gym environment that provides this functionality. To
# make this environment, we need to input as a `kwarg` the
# `sim_gen_func` we defined earlier.
env = DummyVecEnv([lambda: FlattenObservation(
    gym.make('default-rebuilding-acnsim-v0',
             sim_gen_function=sim_gen_function))])
model = PPO2('MlpPolicy', env, verbose=2).learn(10000)
