# coding=utf-8
"""
Algorithms used for deploying trained RL models.
"""
import numpy as np

from .base_algorithm import BaseAlgorithm
from ..acnsim.interface import GymTrainedInterface, \
    GymTrainingInterface, Interface
from importlib.util import find_spec
from typing import Optional, Dict, List, Any

if find_spec("gym") is not None:
    from ..acnsim.gym_acnsim.envs import BaseSimEnv
del find_spec


class SimRLModelWrapper:
    """ Abstract wrapper class that wraps a reinforcement learning
    agent for general use with ACN-Sim. Users should define a new
    class that implements the predict method.
    """
    model: object

    def __init__(self, model: object = None) -> None:
        """
        Wrap an input model.

        Args:
            model (object): The wrapped RL model.
        """
        self.model = model

    def predict(self,
                observation: object,
                reward: float,
                done: bool,
                info: Dict[Any, Any] = None) -> np.ndarray:
        """
        Given an observation, reward, done, and info from an
        environment, return a prediction for the next optimal action
        as a numpy array.

        Args:
            observation: An observation of the environment.
            reward: The last reward returned by the environment.
            done: If true, this environment's simulation is done
                running.
            info: Info for debugging and testing from this environment.
                In a deployment model, this information is restricted.

        Returns:

        """
        raise NotImplementedError


class GymBaseAlgorithm(BaseAlgorithm):
    """ Abstract algorithm class for Simulations using a reinforcement
    learning agent that operates in an Open AI Gym environment.

    Implements abstract class BaseAlgorithm.

    Simulations that run GymAlgorithm-style schedulers have two entry
    points for simulation control. First, a reinforcement learning
    agent may call model.learn(vec_env), which will step through the
    simulation. In this case, schedule() will never be called as all
    simulation progression would be handled by the agent stepping the
    environment. As such, the GymTrainingAlgorithm class does not
    implement a schedule function. See GymTrainingAlgorithm for this
    case.

    Alternatively, the Simulator may call scheduler.run(), causing
    this GymAlgorithm to run model.predict() on a trained model. See
    GymTrainedAlgorithm for this case.

    Algorithm class for Simulations using a reinforcement learning
    agent that operates in an Open AI Gym environment.

    Implements abstract class GymBaseAlgorithm.

    Simulations that run GymAlgorithm-style schedulers have two entry
    points for simulation control. First, the Simulator may call
    scheduler.run(), causing this GymAlgorithm to run model.predict().
    Alternatively, one may call model.learn(vec_env), which instead will
    step through the simulation. See GymTrainingAlgorithm for this case.

    Args:
        max_recompute (int): See BaseAlgorithm.
    """

    _env: Optional[BaseSimEnv]
    max_recompute: Optional[int]

    def __init__(self, max_recompute: int = 1) -> None:
        super().__init__()
        self._env = None
        self.max_recompute = max_recompute

    def __deepcopy__(self, memodict: Optional[Dict] = None
                     ) -> "GymBaseAlgorithm":
        return type(self)(max_recompute=self.max_recompute)

    def register_interface(self, interface: Interface) -> None:
        """ NOTE: Registering an interface sets the environment's
        interface to GymTrainedInterface.
        """
        if not isinstance(interface, GymTrainedInterface):
            # Note that in this case, the actual interface object is not
            # used by the algorithm; rather, a copy of interface of type
            # GymTrainedInterface is used.
            gym_interface: GymTrainedInterface = \
                GymTrainedInterface.from_interface(interface)
        elif isinstance(interface, GymTrainingInterface):
            raise TypeError("Interface GymTrainingInterface cannot be "
                            "registered to a scheduler. Register "
                            "GymTrainedInterface")
        else:
            gym_interface: GymTrainedInterface = interface
        super().register_interface(gym_interface)
        if self._env is not None:
            self.env.interface = interface

    @property
    def env(self) -> BaseSimEnv:
        """ Return the algorithm's gym environment.

        Returns:
            BaseSimEnv: A gym environment that wraps a simulation.

        Raises:
            ValueError: Exception raised if vec_env is accessed prior to
                an vec_env being registered.
        """
        if self._env is not None:
            return self._env
        else:
            raise ValueError(
                'No env has been registered yet. Please call '
                'register_env with an appropriate environment before '
                'attempting to call env or schedule.'
            )

    def register_env(self, env: BaseSimEnv) -> None:
        """ Register a model that outputs schedules for the simulation.

        Args:
            env (BaseSimEnv): An vec_env wrapping a simulation.

        Returns:
            None
        """
        self._env = env
        if self._interface is not None:
            self.env.interface = self._interface

    def schedule(self, active_evs) -> Dict[str, List[float]]:
        """ NOT IMPLEMENTED IN GymBaseAlgorithm. """
        raise NotImplementedError


class GymTrainedAlgorithm(GymBaseAlgorithm):
    """ Algorithm class for Simulations using a reinforcement learning
    agent that operates in an Open AI Gym environment.

    Implements abstract class GymBaseAlgorithm.

    Simulations that run GymAlgorithm-style schedulers have two entry
    points for simulation control. First, the Simulator may call
    scheduler.run(), causing this GymAlgorithm to run model.predict().
    Alternatively, one may call model.learn(vec_env), which instead will
    step through the simulation. See GymTrainingAlgorithm for this case.
    """
    _env: BaseSimEnv
    _model: Optional[SimRLModelWrapper]

    def __init__(self, max_recompute: int = 1) -> None:
        super().__init__(max_recompute=max_recompute)
        self._model = None

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

    def register_model(self, model: SimRLModelWrapper) -> None:
        """ Register a model that outputs schedules for the simulation.

        Args:
            model (SimRLModelWrapper): A model that can be used for
                predictions in ACN-Sim.

        Returns:
            None
        """
        self._model = model

    def schedule(self, active_evs):
        """ Creates a schedule of charging rates for each EVSE in the
        network. This only works if a model and environment have been
        registered.

        Implements BaseAlgorithm.schedule().
        """
        if self._model is None or self._env is None:
            raise TypeError(
                f"A model and environment must be set to call the "
                f"schedule function for GymAlgorithm."
            )
        if not isinstance(self.env.interface, GymTrainedInterface):
            raise TypeError(
                "GymAlgorithm environment must have an interface of "
                "type GymTrainedInterface to call schedule(). "
            )
        self.env.update_state()
        self.env.store_previous_state()
        self.env.action = self.model.predict(
            self.env.observation, self.env.reward,
            self.env.done, self.env.info
        )
        self.env.schedule = self.env.action_to_schedule()
        return self.env.schedule
