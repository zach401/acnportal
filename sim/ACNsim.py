import TestCase
from datetime import datetime, timedelta
from BaseAlgorithm import *
from Interface import Interface
from Simulator import Simulator
from Garage import Garage

class ACNsim:

    def __init__(self):
        pass

    def simulate(self, scheduler, garage):
        sim = Simulator(garage)
        interface = Interface(sim)
        scheduler.interface_setup(interface)
        sim.define_scheduler(scheduler)

        simulation_output = sim.run()
        return simulation_output

    def simulate_real(self, scheduler, test_case):
        garage = Garage()
        garage.set_test_case(test_case)
        return self.simulate(scheduler, garage)

    def simulate_model(self, scheduler, start=datetime.now(), end=(datetime.now() + timedelta(days=2)), period=1):
        garage = Garage()
        garage.generate_test_case(start, end, period)
        return self.simulate(scheduler, garage)

