import TestCase
from datetime import datetime
from BaseAlgorithm import *
from Interface import Interface
from Simulator import Simulator
from Garage import Garage

class ACNsim:

    def __init__(self):
        pass

    def simulate(self, test_case, scheduler):
        garage = Garage()
        garage.set_test_case(test_case)

        sim = Simulator(garage)
        interface = Interface(sim)
        scheduler.interface_setup(interface)
        sim.define_scheduler(scheduler)

        self.simulation_data = sim.run()
        return self.simulation_data

