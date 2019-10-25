import pandapower as pp

"""Simple pandapower experiment with acnsim"""
# TODO: attribute pandapower simple network tutorial as a lot of this code is from there

# In this tutorial, we look at how multiple caltech ACNs running an EDF algorithm effect the grid,
# with and without local PV generation. This requires inputting simulator results into pandapower
# as static loads (that change in time), functionality which the simulator provides through its
# logging of charging rates and analysis functions.

# Global parameters
hv = 0.48 # high voltage in kV
lv = 0.12 # low voltage in kV
n_acns = 4 # number of caltech ACNs

def pp_network_generator(caltech_acn_loads):
	""" This function generates a pandapower network in which the ACN networks will live.

	Args:
        caltech_acn_loads (List[float]): list of loads on caltech ACNs with which the network loads
        are initialized
	"""
	# First initialize the pandapower network in which the caltech ACNs exist
	net = pp.create_empty_network() #create an empty network

	# There is one bus connection that leads from the external grid to 4 0.1 km lines (each line
	# leads to a caltech ACN)
	top_bus = pp.create_bus(net, name="HV Busbar", vn_kv=hv, type="b")
	pp.create_ext_grid(net, top_bus) # Create an external grid connection

	# In this network, we have 4 caltech ACNs, two of which have local PV generation.
	# Create 4 node buses and 4 lines leading from the top bus to the nodes
	hv_nodes = [pp.create_bus(net, name="HV Node {0}".format(i+1), vn_kv=hv, type="n") for i in range(n_acns)]

	hv_lines = [pp.create_line(net, top_bus, hv_nodes[i], length_km=0.1,
		std_type="NAYY 4x50 SE", name="Line {0}".format(i+1)) for i in range(n_acns)]

	# We'll use the 480/120 150 kVA transformers at each ACN. Each transformer
	# needs a low voltage node
	lv_nodes = [pp.create_bus(net, name="LV Node {0}".format(i+1), vn_kv=lv, type="n") for i in range(n_acns)]

	# TODO: ask zach for actual vkr_percent, vk_percent, pfe_kw, i0_percent
	trafos = [pp.create_transformer_from_parameters(net, hv_nodes[i], lv_nodes[i],
		sn_mva=0.15, vn_hv_kv=hv, vn_lv_kv=lv, vk_percent=4.0,
		vkr_percent=1.0, pfe_kw=1.0, i0_percent=0.1,
		name="480V/120V transformer {0}".format(i+1)) for i in range(n_acns)]

	# Create switch buses for each network
	switch_buses = [pp.create_bus(net, name="Switch Bus {0}".format(i+1),
		vn_kv=lv, type="b") for i in range(n_acns)]

	# Create switches for each network. these are circuit breakers
	switches = [pp.create_switch(net, lv_nodes[i], switch_buses[i], et="b", type="CB", closed=True) for i in range(n_acns)]

	# Create the caltech ACN loads from the input loads
	loads = [pp.create_load(net, switch_buses[i], caltech_acn_loads[i]) for i in range(n_acns)]

	# The network is now complete. Return the network along with the switch buses to allow for more connections
	return net, switch_buses

# Now let us run 4 caltech acns for a day each. We'll use the same time period as tutorial 2:

import pytz
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

from acnportal import acnsim
from acnportal import algorithms

def run_sim():
	""" This function runs the simulation from tutorial 2 and returns the resultant simulation
	It uses the EDF algorithm included with acnsim. """
	timezone = pytz.timezone('America/Los_Angeles')
	start = timezone.localize(datetime(2018, 9, 5))
	end = timezone.localize(datetime(2018, 9, 6))
	period = 5  # minute
	voltage = 220  # volts
	default_battery_power = 32 * voltage / 1000 # kW
	site = 'caltech'

	# -- Network -----------------------------------------------------------------------------------------------------------
	cn = acnsim.sites.caltech_acn(basic_evse=True, voltage=voltage)

	# -- Events ------------------------------------------------------------------------------------------------------------
	API_KEY = 'DEMO_TOKEN'
	events = acnsim.acndata_events.generate_events(API_KEY, site, start, end, period, voltage, default_battery_power)
	sch = algorithms.SortedSchedulingAlgo(algorithms.earliest_deadline_first)
	# sch = algorithms.UncontrolledCharging()
	sim = acnsim.Simulator(deepcopy(cn), sch, deepcopy(events), start, period=period)
	sim.run()
	return sim

# TODO: it would be better if we had 4 unique networks or experiments instead of 4 of the same
load_powers = [acnsim.aggregate_current(run_sim()) * lv / 1000 for i in range(n_acns)]
sim_len = len(load_powers[0])
# Now let's see what the power demand on the grid looks like in time with 4 ACNs operating over
# this time period
external_grid_powers = np.zeros((sim_len,))

# We need to do a pandapower Power Flow for each timestep that the simulator ran for:
for i in range(sim_len):
	print("timestep {0}".format(i))
	# get power loads at this time step
	caltech_acn_loads = [load_powers[j][i] for j in range(n_acns)]
	# generate network with these loads
	net, _ = pp_network_generator(caltech_acn_loads)
	# run power flow
	# pp.diagnostic(net)
	pp.runpp(net)
	# get power demand on external grid at this timestep
	external_grid_powers[i] = net.res_ext_grid.p_mw

# Now let's try adding local PV generation to two of the branches in the previous experiment
# We can use NREL data to get an idea of the generation curve over the given time period, 
# but let's scale down the actual power generated as if we had onsite PV generation that
# generated proportional to the load caused by the caltech ACN. We'll use NREL solar
# data near caltech coordinates (data downloaded from https://www.nrel.gov/grid/solar-power-data.html)
# and scale the PV generation to the max demand of the network

file_name = "Actual_34.15_-118.15_2006_DPV_74MW_5_Min.csv"
actual_pv_cap_mw = 74
simulation_pv_cap_mw = max([np.max(load_powers[i]) for i in range(n_acns)])

# Read in NREL PV data for the relevant dates: first cut off all earlier dates:
pv_data = pd.read_csv(file_name).set_index("LocalTime")["09/05/06 00:00":].to_numpy()
# Now get data for sim_len 5-minute time steps into the future
pv_data = pv_data[:sim_len]

# Now scale the pv data based on ACN's required capacity
pv_data = pv_data * simulation_pv_cap_mw / actual_pv_cap_mw

# We can now add in PV generation to half the caltech networks in the previous experiment

def pp_network_generator_with_pv(caltech_acn_loads, pv_gens):
	net, switch_buses = pp_network_generator(caltech_acn_loads)
	# connect PV generation to half the networks
	generators = [pp.create_sgen(net, switch_buses[i], pv_gens[i]) for i in range(n_acns//2)]
	return net, switch_buses

external_grid_powers_with_pv = np.zeros((sim_len,))

# We need to do a pandapower Power Flow for each timestep that the simulator ran for:
for i in range(sim_len):
	print("timestep {0}".format(i))
	# get power loads at this time step
	caltech_acn_loads = [load_powers[j][i] for j in range(n_acns)]
	# get power generation at this time step
	pv_gens = [pv_data[i] for _ in range(n_acns//2)]
	# generate network with these loads
	net, _ = pp_network_generator_with_pv(caltech_acn_loads, pv_gens)
	# run power flow
	# pp.diagnostic(net)
	pp.runpp(net)
	# get power demand on external grid at this timestep
	external_grid_powers_with_pv[i] = net.res_ext_grid.p_mw

# Now let's compare our results to see if adding solar helped

import matplotlib.pyplot as plt
import json

fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
ax[0].plot(external_grid_powers, label='No PV Generation')
ax[1].plot(external_grid_powers_with_pv, label='With PV Generation')
ax[1].set_xlabel('Time (periods)')
ax[0].set_ylabel('Demand on Grid (MW)')
ax[1].set_ylabel('Demand on Grid (MW)')
ax[0].set_title('No PV Generation')
ax[1].set_title('With PV Generation')
plt.show()

data_dump = {"with_pv": external_grid_powers_with_pv.tolist(), "no_pv": external_grid_powers.tolist()}
with open("pp_results.json", "w") as myfile:
	json.dump(data_dump, myfile)