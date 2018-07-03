import TestCase
from datetime import datetime
from BaseAlgorithm import *
from Interface import Interface
from Simulator import Simulator
from GraphDrawer import GraphDrawer

class ACNsim:

    def __init__(self):
        pass

    def simulate(self, test_case, scheduler):
        sim = Simulator(test_case)
        interface = Interface(sim)
        scheduler.interface_setup(interface)
        sim.define_scheduler(scheduler)

        self.simulation_data = sim.run()
        return self.simulation_data

