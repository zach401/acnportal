from .base_algorithm import BaseAlgorithm
from ..acnsim.interface import GymInterface, GymTrainingInterface
from importlib.util import find_spec
from typing import Optional
if find_spec("gym") is not None:
    from ..acnsim.gym_acnsim.envs import BaseSimEnv
del find_spec


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
    step through the simulation. See GymTrainingAlgorithm for this case.

    Args:
        max_recompute (int): See BaseAlgorithm.
    """
    _model: Optional[SimRLModelWrapper]
    _env: Optional[BaseSimEnv]

    def __init__(self, max_recompute=1):
        super().__init__()
        self._model = None
        self._env = None
        self.max_recompute = max_recompute

    def register_interface(self, interface):
        """ NOTE: Registering an interface sets the environment's
        interface to GymInterface.
        """
        if not isinstance(interface, GymInterface):
            gym_interface = GymInterface.from_interface(interface)
        else:
            gym_interface = interface
        super().register_interface(gym_interface)
        self.env.interface = interface

    @property
    def model(self):
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

    def register_model(self, model):
        """ Register a model that outputs schedules for the simulation.

        Args:
            model (SimRLModelWrapper): A model that can be used for
                predictions in ACN-Sim.

        Returns:
            None
        """
        self._model = model

    @property
    def env(self):
        """ Return the algorithm's gym environment.

        Returns:
            BaseSimEnv: A gym environment that wraps a simulation.

        Raises:
            ValueError: Exception raised if env is accessed prior to
                an env being registered.
        """
        if self._env is not None:
            return self._env
        else:
            raise ValueError(
                'No env has been registered yet. Please call '
                'register_env with an appropriate environment before '
                'attempting to call env or schedule.'
            )

    def register_env(self, env):
        """ Register a model that outputs schedules for the simulation.

        Args:
            env (BaseSimEnv): An env wrapping a simulation.

        Returns:
            None
        """
        self._env = env

    def schedule(self, active_evs):
        """ Creates a schedule of charging rates for each EVSE in the
        network. This only works if a model and environment

        Implements BaseAlgorithm.schedule().
        """
        if self.model is None or self.env is None:
            raise TypeError(
                f"A model and environment must be set to call the "
                f"schedule function for GymAlgorithm."
            )
        if isinstance(self.env.interface, GymTrainingInterface):
            raise TypeError(
                "GymAlgorithm environment interface is of type "
                "GymTrainingInterface. The environment must have an "
                "interface of type GymInterface to call schedule()."
            )
        self.env.update_state()
        self.env.store_previous_state()
        self.env.action = self.model.predict(
            self.env.observation, self.env.reward,
            self.env.done, self.env.info
        )
        self.env.schedule = self.env.action_to_schedule()
        return self.env.schedule
