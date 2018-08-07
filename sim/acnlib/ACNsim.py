from datetime import datetime, timedelta
from acnlib.Interface import Interface
from acnlib.Simulator import Simulator
from acnlib.Garage import Garage

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
        garage.set_test_case(test_case)
        return self.simulate(scheduler, garage)

    def simulate_model(self, scheduler, start=None, end=None, voltage=220, max_rate=32, period=1, model='empirical'):
        '''
        Simulate the system with data generated from the statistical model.

        :param scheduler: The scheduling algorithm used for this simulation. Should extend ``BaseAlgorithm``.
        :type scheduler: BaseAlgorithm
        :param start: Start time of simulation
        :type start: datetime
        :param end: End time of simulation
        :type end: datetime
        :param voltage: The voltage level of the power grid [V].
        :type voltage: float
        :param max_rate: The maximum rate the EVs can be charged with [A].
        :type max_rate: float
        :param period: The length of one iteration in the simulation [minutes].
        :type period: int
        :return: The output from the simulation
        :rtype: SimulationOutput
        '''
        if start == None:
            start = datetime.now()
        if end == None:
            end = (datetime.now()+ timedelta(days=2))
        start = start.replace(hour=0, minute=0, second=0)
        end = end.replace(hour=0, minute=0, second=0)
        garage = Garage()
        garage.generate_test_case(start_dt=start, end_dt=end, period=period, voltage=voltage, max_rate=max_rate, model=model)
        return self.simulate(scheduler, garage)

