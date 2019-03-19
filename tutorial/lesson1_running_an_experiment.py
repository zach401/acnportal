from datetime import datetime
import pytz
import numpy as np
from matplotlib import pyplot as plt

from algorithms import UncontrolledCharging
from algorithms import SortingAlgorithm
from algorithms import earliest_deadline_first
from acnsim.events import EventQueue
from acnsim.network.sites import CaltechACN
from acnsim.simulator import Simulator
from acnsim.utils.generate_events import generate_test_case_api
from acnsim.analysis import *

# -- Experiment Parameters ---------------------------------------------------------------------------------------------
# The timezone of the experiment should be the timezone of the ACN from which data is being used. If real data
# is not used, then the timezone can be omitted.
timezone = pytz.timezone('America/Los_Angeles')

# Start and End times are used when collecting data.
start = datetime(2018, 9, 5).astimezone(timezone)
end = datetime(2018, 9, 6).astimezone(timezone)

# The period of the simulation is how long each time interval will be.
period = 5  # minute

# We assume a set voltage for the entire network. This is used for converting units of power to current.
voltage = 220  # volts


# -- Network -----------------------------------------------------------------------------------------------------------
# Each experiment should have a network which represents an Adaptive Charging Network. You are able to define your own
# network, but for now we will use the predefined Caltech ACN.
cn = CaltechACN(basic_evse=True)


# -- Scheduling Algorithm ----------------------------------------------------------------------------------------------
# For this simple experiment we will use the predefined Uncontrolled Charging algorithm. We will cover more advanced
# algorithms and how to define a custom algorithm in future tutorials.
# sch = UncontrolledCharging()
sch = SortingAlgorithm(earliest_deadline_first)

# -- Events ------------------------------------------------------------------------------------------------------------
# Each simulation needs a queue of events. These events can be defined manually, or more commonly, are created from
# real data. In this case we will use the Caltech Charging Dataset API to download real events from the time period
# of our simulation.

# For this tutorial we will use a demonstration token to access the API, but when using real simulations you will
# want to register for your own free API token at ev.caltech.edu/dataset.html.
API_KEY = 'DEMO_TOKEN'

# An EventQueue is a special container which stores the events for the simulation.
events = EventQueue()

# We use the generate_test_case_api() method to fill the EventQueue.
events.add_events(generate_test_case_api(API_KEY, start, end))


# -- Simulator ---------------------------------------------------------------------------------------------------------
# We can now load the simulator enviroment with the network, scheduler, and events we have already defined.
sim = Simulator(cn, sch, events, start, period=period, max_recomp=1)

# To execute the simulation we simply call the run() function.
sim.run()

# -- Analysis ----------------------------------------------------------------------------------------------------------
# After running the simulation, we can analyze the results using data stored in the simulator.

# Find percentage of requested energy which was delivered.
print('Proportion of requested energy delivered: {0}'.format(proportion_of_energy_delivered(sim)))

# Find peak aggregate current during the simulation
print('Peak aggregate current: {0} A'.format(sim.peak))

# Plotting aggregate current
agg_current = aggregate_current(sim)
plt.plot(agg_current)
plt.xlabel('Time (periods)')
plt.ylabel('Current (A)')
plt.title('Total Aggregate Current')
plt.show()
