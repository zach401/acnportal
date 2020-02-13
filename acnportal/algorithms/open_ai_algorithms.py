from .base_algorithm import BaseAlgorithm
from ..acnsim.interface import GymInterface


class GymAlgorithm(BaseAlgorithm):
    """
    Algorithm class for Simulations using a reinforcement learning
    agent that operates in an Open AI Gym environment.

    This algorithm's schedule function is never called; instead, a Gym
    environment is stepped when a new schedule is needed.
    """
    def __init__(self):
        super().__init__()
        self.max_recompute = 1

    def register_interface(self, interface):
        if not isinstance(interface, GymInterface):
            gym_interface = GymInterface.from_interface(interface)
        else:
            gym_interface = interface
        super().register_interface(gym_interface)

    def schedule(self, active_evs):
        """ Implements BaseAlgortihm.schedule(). """
        raise NotImplementedError(
            "GymAlgorithm does not implement a schedule function. "
            "Instead, call step on the environment containing this "
            "simulation."
        )
