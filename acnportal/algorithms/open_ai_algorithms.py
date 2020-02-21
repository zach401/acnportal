from .base_algorithm import BaseAlgorithm
from ..acnsim.interface import GymInterface
from ..acnsim.gym_acnsim.envs import BaseSimEnv


class SimRLModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, observation, reward, done, info):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


class StableBaselinesModel(SimRLModelWrapper):
    def predict(self, observation, reward, done, info):
        return self.model.predict(observation)

    def learn(self):
        return self.model.learn()

    def save(self):
        return self.model.save()


class GymAlgorithm(BaseAlgorithm):
    """
    Algorithm class for Simulations using a reinforcement learning
    agent that operates in an Open AI Gym environment.

    Implements abstract class BaseAlgorithm.

    Simulations that run GymAlgorithm-style schedulers have two entry
    points for simulation control. First, the Simulator may call
    scheduler.run(), causing this GymAlgorithm to run model.predict().
    Alternatively, one may call model.learn(env), which instead will
    step through the simulation

    Args:
        model (SimRLModelWrapper): The reinforcement learning model that
            will give actions (schedules) to the simulator.
        env (BaseSimEnv): The Gym environment that represents this
            simulation.
        max_recompute (int): See BaseAlgorithm.
    """
    model: SimRLModelWrapper
    env: BaseSimEnv

    def __init__(self, model, env, max_recompute=1):
        super().__init__()
        self.model = model
        self.env = env
        self.max_recompute = max_recompute

    def register_interface(self, interface):
        """ NOTE: Registering an interface sets the environment's
        interface to a deployment style interface.
        """
        gym_interface = GymInterface.from_interface(interface)
        super().register_interface(gym_interface)
        self.env.interface = interface

    def schedule(self, active_evs):
        """ Creates a schedule of charging rates for each EVSE in the
        network.

        Implements BaseAlgorithm.schedule().
        """
        self.env.update_state()
        self.env.store_previous_state()
        self.env.action = self.model.predict(
            self.env.observation, self.env.reward,
            self.env.done, self.env.info
        )
        return self.env.action_to_schedule()
