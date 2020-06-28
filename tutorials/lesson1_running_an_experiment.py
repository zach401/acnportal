"""
ACN-Sim Tutorial: Lesson 1
Running an Experiment
by Zachary Lee
Last updated: 03/19/2019
--

In this first lesson we will learn how to setup and run a simulation using a built-in scheduling algorithm.
After running the simulation we will learn how to use the analysis subpackage to analyze the results of the simulation.
"""
import pytz
from datetime import datetime

import matplotlib.pyplot as plt

from acnportal import acnsim
from acnportal import algorithms

# -- Experiment Parameters ---------------------------------------------------------------------------------------------
# Timezone of the ACN we are using.
timezone = pytz.timezone("America/Los_Angeles")

# Start and End times are used when collecting data.
start = timezone.localize(datetime(2018, 9, 5))
end = timezone.localize(datetime(2018, 9, 6))

# How long each time discrete time interval in the simulation should be.
period = 5  # minutes

# Voltage of the network.
voltage = 220  # volts

# Default maximum charging rate for each EV battery.
default_battery_power = 32 * voltage / 1000  # kW

# Identifier of the site where data will be gathered.
site = "caltech"

# -- Network -----------------------------------------------------------------------------------------------------------
# For this experiment we use the predefined CaltechACN network.
cn = acnsim.sites.caltech_acn(basic_evse=True, voltage=voltage)


# -- Events ------------------------------------------------------------------------------------------------------------
#  In this case we will use the Caltech Charging Dataset API to download real events from the time period of our
#  simulation.

# For this tutorial we will use a demonstration token to access the API, but when using real simulations you will
# want to register for your own free API token at ev.caltech.edu/dataset.html.
API_KEY = "DEMO_TOKEN"

# An EventQueue is a special container which stores the events for the simulation. In this case we use the
# acndata_events utility to pre-fill the event queue based on real events in the Caltech Charging Dataset.
events = acnsim.acndata_events.generate_events(
    API_KEY, site, start, end, period, voltage, default_battery_power
)


# -- Scheduling Algorithm ----------------------------------------------------------------------------------------------
# For this simple experiment we will use the predefined Uncontrolled Charging algorithm. We will cover more advanced
# algorithms and how to define a custom algorithm in future tutorials.
sch = algorithms.UncontrolledCharging()


# -- Simulator ---------------------------------------------------------------------------------------------------------
# We can now load the simulator enviroment with the network, scheduler, and events we have already defined.
sim = acnsim.Simulator(cn, sch, events, start, period=period)

# To execute the simulation we simply call the run() function.
sim.run()

# -- Analysis ----------------------------------------------------------------------------------------------------------
# After running the simulation, we can analyze the results using data stored in the simulator.

# Find percentage of requested energy which was delivered.
total_energy_prop = acnsim.proportion_of_energy_delivered(sim)
print("Proportion of requested energy delivered: {0}".format(total_energy_prop))

# Find peak aggregate current during the simulation
print("Peak aggregate current: {0} A".format(sim.peak))

# Plotting aggregate current
agg_current = acnsim.aggregate_current(sim)
plt.plot(agg_current)
plt.xlabel("Time (periods)")
plt.ylabel("Current (A)")
plt.title("Total Aggregate Current")
plt.show()
