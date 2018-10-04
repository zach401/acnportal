from datetime import datetime, timedelta

from acnlib.Garage import Garage
from acnlib.Interface import Interface
from acnlib.Simulator import Simulator


class ACNsim:

    def __init__(self):
        pass

    def simulate(self, scheduler, garage):
        sim = Simulator(garage, scheduler)
        interface = Interface(sim)
        scheduler.interface_setup(interface)

        simulation_output = sim.run()
        return simulation_output

    def simulate_real(self, scheduler, test_case):
        '''
        Simulate the system with real data. To do this a test case must have been manually created.

        :param scheduler: The scheduling algorithm used for this simulation. Should extend ``BaseAlgorithm``.
        :type scheduler: BaseAlgorithm
        :param test_case: The test case holding the session data.
        :type test_case: TestCase
        :return: The output from the simulation
        :rtype: SimulationOutput
        '''
        garage = Garage()
        garage.test_case = test_case
        return self.simulate(scheduler, garage)
