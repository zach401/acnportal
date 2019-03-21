"""
ACN-Sim Tutorial: Lesson 2
Developing a Custom Algorithm
by Zachary Lee
Last updated: 03/19/2019
--

In this lesson we will learn how to develop a custom algorithm and run it using ACN-Sim. For this example we will be
writing an Earliest Deadline First Algorithm. This algorithm is already available as part of the SortingAlgorithm in the
algorithms package, so we will compare the results of our implementation with the included one.
"""

# -- Custom Algorithm --------------------------------------------------------------------------------------------------
from algorithms import BaseAlgorithm

# All custom algorithms should inherit from the abstract class BaseAlgorithm. It is the responsibility of all derived
# classes to implement the schedule method. This method takes as an input a list of EVs which are currently connected
# to the system but have not yet finished charging. Its output is a dictionary which maps a station_id to a list of
# charging rates. Each charging rate is valid for one period measured relative to the current period.
# For Example:
#   * schedule['abc'][0] is the charging rate for station 'abc' during the current period
#   * schedule['abc'][1] is the charging rate for the next period
#   * and so on.
#
# If an algorithm only produces charging rates for the current time period, the length of each list should be 1.
# If this is the case, make sure to also set the maximum resolve period to be 1 period so that the algorithm will be
# called each period. An alternative is to repeat the charging rate a number of times equal to the max recompute period.


class EarliestDeadlineFirstAlgo(BaseAlgorithm):
    """ Algorithm which assigns charging rates to each EV in order or departure time.

    Implements abstract class BaseAlgorithm.

    For this algorithm EVs will first be sorted by departure time. We will then allocate as much current as possible
    to each EV in order until the EV is finished charging or an infrastructure limit is met.

    Args:
        increment (number): Minimum increment of charging rate. Default: 1.
    """
    def __init__(self, increment=1):
        super().__init__()
        self._increment = increment

    def schedule(self, active_evs):
        """ Schedule EVs by first sorting them by departure time, then allocating them their maximum feasible rate.

        Implements abstract method schedule from BaseAlgorithm.

        See class documentation for description of the algorithm.

        Args:
            active_evs (List[EV]): see BaseAlgorithm

        Returns:
            Dict[str, List[float]]: see BaseAlgorithm
        """
        # First we define a schedule, this will be the output of our function
        schedule = {ev.station_id: [0] for ev in active_evs}

        # Next, we sort the active_evs by their departure time.
        sorted_evs = sorted(active_evs, key=lambda x: x.departure)

        # We now iterate over the sorted list of EVs.
        for ev in sorted_evs:
            # First try to charge the EV at its maximum rate. Remember that each schedule value must be a list, even
            #   if it only has one element.
            schedule[ev.station_id] = [ev.max_rate]

            # If this is not feasible, we will reduce the rate.
            #   interface.is_feasible() is one way to interact with the constraint set of the network. We will explore
            #   another more direct method in lesson 3.
            while not self.interface.is_feasible(schedule):

                # Since the maximum rate was not feasible, we should try a lower rate.
                schedule[ev.station_id][0] -= self._increment

                # EVs should never charge below 0 (i.e. discharge) so we will clip the value at 0.
                if schedule[ev.station_id][0] < 0:
                    schedule[ev.station_id] = [0]
                    break
        return schedule


# -- Run Simulation ----------------------------------------------------------------------------------------------------
from datetime import datetime
import pytz
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from copy import deepcopy

from algorithms import SortedSchedulingAlgo
from algorithms import earliest_deadline_first
from acnsim.events import c2api
from acnsim.network.sites import CaltechACN
from acnsim.simulator import Simulator
from acnsim.analysis import *

# Now that we have implemented our algorithm, we can try it out using the same experiment setup as in lesson 1.
# The only difference will be which scheduling algorithm we use.


# -- Experiment Parameters ---------------------------------------------------------------------------------------------
timezone = pytz.timezone('America/Los_Angeles')
start = datetime(2018, 9, 5).astimezone(timezone)
end = datetime(2018, 9, 6).astimezone(timezone)
period = 5  # minute
voltage = 220  # volts
max_rate = 32 # amps
site = 'caltech'

# -- Network -----------------------------------------------------------------------------------------------------------
cn = CaltechACN(basic_evse=True)

# -- Events ------------------------------------------------------------------------------------------------------------
API_KEY = 'DEMO_TOKEN'
events = c2api.generate_events(API_KEY, site, start, end, period, voltage, max_rate)


# -- Scheduling Algorithm ----------------------------------------------------------------------------------------------
sch = EarliestDeadlineFirstAlgo(increment=1)
sch2 = SortedSchedulingAlgo(earliest_deadline_first)

# -- Simulator ---------------------------------------------------------------------------------------------------------
sim = Simulator(deepcopy(cn), sch, deepcopy(events), start, period=period, max_recomp=1)
sim.run()

# For comparison we will also run the builtin earliest deadline first algorithm
sim2 = Simulator(deepcopy(cn), sch2, deepcopy(events), start, period=period, max_recomp=1)
sim2.run()

# -- Analysis ----------------------------------------------------------------------------------------------------------
# We can now compare the two algorithms side by side by looking that the plots of aggregated current.
# We see from these plots that our implementation matches th included one quite well. If we look closely however, we
# might see a small difference. This is because the included algorithm uses a more efficient bisection based method
# instead of our simpler linear search to find a feasible rate.
plt.plot(aggregate_current(sim), label='Our EDF')
plt.plot(aggregate_current(sim2), label='Included EDF')
plt.legend()
plt.xlabel('Time (periods)')
plt.ylabel('Current (A)')
plt.title('Total Aggregate Current')
plt.show()
