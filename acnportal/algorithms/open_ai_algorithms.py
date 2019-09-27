from .base_algorithm import BaseAlgorithm
from ..acnsim.interface import OpenAIInterface


class OpenAIAlgorithm(BaseAlgorithm):
    """ Placeholder algorithm class for OpenAI simulations. This is never called, instead the environment
    is stepped """
    def __init__(self):
        super().__init__()
        self.max_recompute = 1

    def register_interface(self, interface):
    	self._interface = OpenAIInterface.from_interface(interface)