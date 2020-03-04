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

import os
import random
from copy import deepcopy
from datetime import datetime
from typing import List, Callable, Optional, Dict

import numpy as np
import gym
import pytz
from gym.wrappers import FlattenObservation
from matplotlib import pyplot as plt
from stable_baselines import PPO2
from stable_baselines.common import BaseRLModel
from stable_baselines.common.vec_env import DummyVecEnv

from acnportal import acnsim
from acnportal import algorithms
from acnportal.acnsim import events, models, Simulator
from acnportal.acnsim.gym_acnsim.envs.action_spaces import SimAction
from acnportal.acnsim.gym_acnsim.envs import BaseSimEnv, \
    reward_functions, CustomSimEnv, default_action_object, \
    default_observation_objects
from acnportal.acnsim.gym_acnsim.envs.observation import SimObservation
from acnportal.acnsim.interface import GymTrainedInterface, Interface
from acnportal.algorithms import SimRLModelWrapper, BaseAlgorithm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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
def _random_sim_builder(algorithm: BaseAlgorithm) -> Simulator:
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

    # Simulation to be wrapped
    return acnsim.Simulator(deepcopy(cn), algorithm,
                            deepcopy(event_queue), start, period=period,
                            verbose=False)


def interface_generating_function() -> BaseAlgorithm:
    """
    Initializes a simulation with random events on a 1 phase, 1
    constraint ACN (simple_acn), with 1 EVSE
    """
    # For training, this algorithm isn't run. So, we need not provide
    # any arguments.
    schedule_rl = algorithms.GymTrainingAlgorithm()

    # Simulation to be wrapped
    _ = _random_sim_builder(schedule_rl)
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
vec_env = DummyVecEnv([lambda: FlattenObservation(
    gym.make('default-rebuilding-acnsim-v0',
             interface_generating_function=interface_generating_function)
)])
model = PPO2('MlpPolicy', vec_env, verbose=2)
num_iterations: int = int(1e3)
model_name: str = f"PPO2_{num_iterations}_test_{'today'}.zip"
# model.learn(num_iterations)
# model.save(model_name)

# We've trained the above model for 10000 iterations. Packaged with this
# library is the same model trained for 1000000 iterations, which we
# will now load
model.load(model_name)


# This is a stable_baselines PPO2 model. PPO2 requires vectorized
# environments to run, so the model wrapper should convert between
# vectorized and non-vectorized environments.
class StableBaselinesRLModel(SimRLModelWrapper):
    """ An RL model wrapper that wraps stable_baselines style models.
    """
    model: BaseRLModel

    def predict(self, observation, reward, done, info, **kwargs) -> np.ndarray:
        return self.model.predict(observation, **kwargs)


class GymTrainedAlgorithmVectorized(algorithms.BaseAlgorithm):
    """ Abstract algorithm class for Simulations using a
    reinforcement learning agent that operates in an Open AI Gym
    environment that is vectorized via stable-baselines VecEnv style
    constructions.

    Implements abstract class BaseAlgorithm.

    Vectorized environments in stable-baselines do not inherit from
    gym Env, so we must define a new algorithm class that handles
    models that use these environments.

    Args:
        max_recompute (int): See BaseAlgorithm.
    """

    _env: Optional[DummyVecEnv]
    max_recompute: Optional[int]
    _model: Optional[SimRLModelWrapper]

    def __init__(self, max_recompute: int = 1) -> None:
        super().__init__()
        self._env = None
        self.max_recompute = max_recompute
        self._model = None

    def __deepcopy__(self, memodict: Optional[Dict] = None
                     ) -> "GymTrainedAlgorithmVectorized":
        return type(self)(max_recompute=self.max_recompute)

    def register_interface(self, interface: Interface) -> None:
        """ NOTE: Registering an interface sets the environment's
        interface to GymTrainedInterface.
        """
        if not isinstance(interface, GymTrainedInterface):
            gym_interface: GymTrainedInterface = \
                GymTrainedInterface.from_interface(interface)
        else:
            gym_interface: GymTrainedInterface = interface
        super().register_interface(gym_interface)
        if self._env is not None:
            self.env.interface = interface

    @property
    def env(self) -> DummyVecEnv:
        """ Return the algorithm's gym environment.

        Returns:
            DummyVecEnv: A gym environment that wraps a simulation.

        Raises:
            ValueError: Exception raised if vec_env is accessed prior to
                an vec_env being registered.
        """
        if self._env is not None:
            return self._env
        else:
            raise ValueError(
                'No vec_env has been registered yet. Please call '
                'register_env with an appropriate environment before '
                'attempting to call vec_env or schedule.'
            )

    def register_env(self, env: DummyVecEnv) -> None:
        """ Register a model that outputs schedules for the simulation.

        Args:
            env (DummyVecEnv): An vec_env wrapping a simulation.

        Returns:
            None
        """
        self._env = env

    @property
    def model(self) -> SimRLModelWrapper:
        """ Return the algorithm's predictive model.

        Returns:
            SimRLModelWrapper: A predictive model that returns an array
                of actions given an environment wrapping a simulation.

        Raises:
            ValueError: Exception raised if model is accessed prior to
                a model being registered.
        """
        if self._model is not None:
            return self._model
        else:
            raise ValueError(
                'No model has been registered yet. Please call '
                'register_model with an appropriate model before '
                'attempting to call model or schedule.'
            )

    def register_model(self, new_model: SimRLModelWrapper) -> None:
        """ Register a model that outputs schedules for the simulation.

        Args:
            new_model (SimRLModelWrapper): A model that can be used for
                predictions in ACN-Sim.

        Returns:
            None
        """
        self._model = new_model

    def schedule(self, active_evs) -> Dict[str, List[float]]:
        """ Creates a schedule of charging rates for each EVSE in the
        network. This only works if a model and environment have been
        registered.

        Overrides BaseAlgorithm.schedule().

        The environment is assumed to be vectorized.
        """
        if self._model is None or self._env is None:
            raise TypeError(
                f"A model and environment must be set to call the "
                f"schedule function for GymAlgorithm."
            )
        env: BaseSimEnv = self._env.envs[0].env
        if not isinstance(env.interface, GymTrainedInterface):
            raise TypeError(
                "GymAlgorithm environment must have an interface of "
                "type GymTrainedInterface to call schedule(). "
            )
        env.update_state()
        env.store_previous_state()
        env.action = self.model.predict(
            self._env.env_method("observation", env.observation)[0],
            env.reward, env.done, env.info
        )[0]
        env.schedule = env.action_to_schedule()
        return env.schedule


evaluation_algorithm = GymTrainedAlgorithmVectorized()
evaluation_simulation = _random_sim_builder(evaluation_algorithm)

# Make a new, single-use environment with only charging rewards.
observation_objects: List[SimObservation] = default_observation_objects
action_object: SimAction = default_action_object
reward_functions: List[Callable[[BaseSimEnv], float]] = [
    reward_functions.hard_charging_reward]
eval_env: DummyVecEnv = DummyVecEnv([lambda: FlattenObservation(CustomSimEnv(evaluation_algorithm.interface,
                                      observation_objects,
                                      action_object,
                                      reward_functions))])
evaluation_algorithm.register_env(eval_env)
evaluation_algorithm.register_model(StableBaselinesRLModel(model))

evaluation_simulation.run()

plt.plot(acnsim.aggregate_current(evaluation_simulation))
plt.show()
