import TestCase
from datetime import datetime, timedelta
from BaseAlgorithm import *
from Interface import Interface
from Simulator import Simulator
from Garage import Garage

class ACNsim:

    def __init__(self):
        pass

    def simulate(self, test_case, scheduler):
        garage = Garage()
        #garage.set_test_case(test_case)
        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0)
        garage.generate_test_case(today, today + timedelta(days=5))

        sim = Simulator(garage)
        interface = Interface(sim)
        scheduler.interface_setup(interface)
        sim.define_scheduler(scheduler)

        self.simulation_data = sim.run()
        return garage.test_case

