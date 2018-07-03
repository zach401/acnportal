import TestCase
from datetime import datetime
from BaseAlgorithm import *
from Interface import Interface
from Simulator import Simulator

class ACNsim:

    def __init__(self, test_case, scheduler):
        self.test_case = test_case
        self.scheduler = scheduler

    def run(self):
        sim = Simulator(self.test_case)
        interface = Interface(sim)
        self.scheduler.interface_setup(interface)
        sim.define_scheduler(self.scheduler)

        charging_data = sim.run()
        return charging_data