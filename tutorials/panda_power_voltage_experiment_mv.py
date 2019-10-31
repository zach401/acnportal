# Build Pandapower Network --------------------------------------------------
import pandapower as pp
from pandapower import networks as pn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import pytz
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

from acnportal import acnsim
from acnportal import algorithms
import pickle as pkl

import matplotlib.pyplot as plt
import json

from adacharge import *

# Run Simulations for ACN Loads ---------------------------------------------
def run_sim(sim_algorithm, external_signal=None, info=""):
    """ This function runs the simulation from tutorial 2 and returns the resultant simulation
    It uses the EDF algorithm included with acnsim. """
    timezone = pytz.timezone('America/Los_Angeles')
    start = timezone.localize(datetime(2019, 8, 5))
    end = timezone.localize(datetime(2019, 8, 6))
    period = 5  # minute
    voltage = 208  # volts
    default_battery_power = 32 * voltage / 1000 # kW
    site = 'caltech'
    print(site, info)
    # -- Network -----------------------------------------------------------------------------------------------------------
    cn = acnsim.sites.caltech_acn(basic_evse=True, voltage=voltage)

    # -- Events ------------------------------------------------------------------------------------------------------------
    API_KEY = 'DEMO_TOKEN'
    events = acnsim.acndata_events.generate_events(API_KEY, site, start, end, period, voltage, default_battery_power)
    # print(events._queue)
    if external_signal is not None:
        sch = sim_algorithm(external_signal=external_signal)
    else:
        sch = sim_algorithm()
    sim = acnsim.Simulator(deepcopy(cn), sch, deepcopy(events), start, period=period, verbose=True)
    sim.run()
    print(acnsim.proportion_of_energy_delivered(sim))
    with open("sim_res_{0}".format(info), "wb") as simfile:
        pkl.dump(sim, simfile)

    site = 'jpl'
    print(site, info)
    # -- Network -----------------------------------------------------------------------------------------------------------
    cn = acnsim.sites.jpl_acn(basic_evse=True, voltage=voltage)

    # -- Events ------------------------------------------------------------------------------------------------------------
    API_KEY = 'DEMO_TOKEN'
    events = acnsim.acndata_events.generate_events(API_KEY, site, start, end, period, voltage, default_battery_power)
    # print(events._queue)
    if external_signal is not None:
        sch = sim_algorithm(external_signal=external_signal)
    else:
        sch = sim_algorithm()
    sim2 = acnsim.Simulator(deepcopy(cn), sch, deepcopy(events), start, period=period, verbose=True)
    sim2.run()
    print(acnsim.proportion_of_energy_delivered(sim2))
    with open("sim_res_{0}_jpl".format(info), "wb") as simfile:
        pkl.dump(sim2, simfile)
    return acnsim.aggregate_current(sim) * voltage / 1000 / 1000, acnsim.aggregate_current(sim2) * voltage / 1000 / 1000


# We run three cases:
# 1) Uncontrolled EV charging
# 2) EDF Scheduling
# 3) Load flattening with building loads + solar as input

# Uncontrolled EV charging
sim_unc, sim_unc_jpl = run_sim(algorithms.UncontrolledCharging, info="uncontrolled")
len_sim = min(len(sim_unc), len(sim_unc_jpl))
sim_unc = sim_unc[:len_sim]
sim_unc_jpl = sim_unc_jpl[:len_sim]
sim_peak = max(np.max(sim_unc), np.max(sim_unc_jpl))
print(sim_peak)
# EDF
sim_edf, sim_edf_jpl = run_sim(lambda: algorithms.SortedSchedulingAlgo(algorithms.earliest_deadline_first), info="edf")
sim_edf = sim_edf[:len_sim]
sim_edf_jpl = sim_edf_jpl[:len_sim]
assert len_sim == len(sim_edf)
assert len_sim == len(sim_edf_jpl) 

# For load flattening, we first need to match the building load and PV signals
# We will also normalize generation to the peak utilization of uncontrolled EV charging

load_start_idx = 62208

# Load in PV Data -----------------------------------------------------------
# File name for data
file_name_pv = "Actual_34.15_-118.15_2006_DPV_74MW_5_Min.csv"

# Read in NREL PV data for the relevant dates: first cut off all earlier dates:
# Run for entire year?
pv_data_full = -1*pd.read_csv(file_name_pv).set_index("LocalTime")["Power(MW)"].to_numpy()
# MW max capacity for the site (used for scaling)
pv_data_full = -1*pv_data_full/np.min(pv_data_full) * 0.15
pv_data = pv_data_full[load_start_idx:load_start_idx+len_sim]
assert len_sim == len(pv_data)

# Load in Building Data -----------------------------------------------------
# Large Office
# File name for data
file_name_mo = "RefBldgLargeOfficeNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv"

# Read in NREL building load data, only taking total facility power
# TODO: might want to keep the times here.
mo_data = pd.read_csv(file_name_mo).set_index("Date/Time")["Electricity:Facility [kW](Hourly)"].to_numpy()
# MW max capacity for the site (used for scaling)
mo_data = mo_data/1000

# Warehouse
# File name for data
file_name_wh = "RefBldgWarehouseNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv"

# Read in NREL building load data, only taking total facility power
wh_data = pd.read_csv(file_name_wh).set_index("Date/Time")["Electricity:Facility [kW](Hourly)"].to_numpy()
# MW max capacity for the site (used for scaling)
wh_data = wh_data/1000

def shift_one(arr):
    newarr = arr[:-1]
    last_elt = arr[-1]
    return np.append([last_elt], newarr)

# Building loads are shifted by 1 hour relative to PV
mo_data = shift_one(mo_data)
wh_data = shift_one(wh_data)

# Expand building loads to match pv gen signal
mo_data_full = np.tile(mo_data, (12, 1)).flatten(order='F')
wh_data_full = np.tile(wh_data, (12, 1)).flatten(order='F')
mo_data = mo_data_full[load_start_idx:load_start_idx+len_sim]
wh_data = wh_data_full[load_start_idx:load_start_idx+len_sim]
print(np.max(mo_data))
print(np.max(wh_data))
print(np.min(pv_data))
assert len(mo_data) == len_sim
assert len(wh_data) == len_sim

# sim_lf_pv, sim_lf_pv_jpl = run_sim(AdaChargeLoadFlattening, external_signal=mo_data_full[load_start_idx:]/5+pv_data_full[load_start_idx:], info="lf_bl_pv")
# sim_lf, sim_lf_jpl = run_sim(AdaChargeLoadFlattening, external_signal=mo_data_full[load_start_idx:]/5, info="lf_bl")
def load_from_pkl(name):
    with open(name, 'rb') as infile:
        sim = pkl.load(infile)
    return acnsim.aggregate_current(sim) * 208 / 1000 / 1000
sim_lf_pv = load_from_pkl("sim_res_lf_bl_pv")
sim_lf = load_from_pkl("sim_res_lf_bl")
sim_lf_pv_jpl = load_from_pkl("sim_res_lf_bl_pv_jpl")
sim_lf_jpl = load_from_pkl("sim_res_lf_bl_jpl")

sim_lf_pv = sim_lf_pv[:len_sim]
sim_lf = sim_lf[:len_sim]
sim_lf_jpl = sim_lf_jpl[:len_sim]
sim_lf_pv_jpl = sim_lf_pv_jpl[:len_sim]

assert len(sim_lf_pv) == len_sim
assert len(sim_lf) == len_sim
assert len(sim_lf_jpl) == len_sim
assert len(sim_lf_pv_jpl) == len_sim

# Now we conduct power flows for different cases
def assign_loads(network, input_loads):
    # names =  ["Load C12 - Med Office 1", "Load C13 - Caltech ACN", "Load C14 - PV Array 1",
    #     "Load C17 - Med Office 2", "Load C18 - JPL ACN", "Load 19 - PV Array 2", 
    #     "Load 20 - Warehousee", "Load 21 - PV Array 3"]
    # Loads fall into 3 "sites"
    # Site 1 has 5 Caltech ACNs, 500 kW (capacity) of PV generation, and a Large office. On load CI10
    network.load["p_mw"][network.load["name"] == "Load CI10"] = 5 * input_loads['caltech'] \
    	+ 5 * input_loads['pv'] + input_loads['office_load']
    # Site 2 has 5 JPL ACNs, 500 kW of PV generation, and a Large office. On load CI7
    network.load["p_mw"][network.load["name"] == "Load CI7"] = 5 * input_loads['jpl'] \
    	+ 5 * input_loads['pv'] + input_loads['office_load']
    # Site 3 has 500 kW of PV generation and a Large office. On load CI9
    network.load["p_mw"][network.load["name"] == "Load CI7"] = 5 * input_loads['pv'] \
    	+ input_loads['office_load']

    return deepcopy(network)

def write_results():
    with open("load_results.json", "w") as outfile:
            json.dump(output_loads, outfile)
    with open("bus_results.json", "w") as outfile:
            json.dump(output_buses, outfile)
    with open("line_results.json", "w") as outfile:
            json.dump(output_lines, outfile)
    with open("trafo_results.json", "w") as outfile:
            json.dump(output_trafos, outfile)

def read_results():
    with open("load_results.json", "r") as infile:
            output_loads = json.load(infile)
    with open("bus_results.json", "r") as infile:
            output_buses = json.load(infile)
    with open("line_results.json", "r") as infile:
            output_lines = json.load(infile)
    with open("trafo_results.json", "r") as infile:
            output_trafos = json.load(infile)
    return output_loads, output_buses, output_lines, output_trafos

def recover_pp_network(jsoned_network_dict):
    for key, value in jsoned_network_dict:
        new_lst = value
        for i in range(len(new_lst)):
            with open(value[i], "r") as innet:
                new_lst[i] = pp.from_json(innet)
        jsoned_network_dict[key] = new_lst
    return jsoned_network_dict

def process_network(net):
	"""Open all switches except feeder 1, make all power factors 1, only consider commercial loads (others out of service) """
	net.switch["closed"] = False
	net.switch["closed"][6] = True
	net.load["q_mvar"] = 0
	for i in range(10):
		net.load["in_service"][i] = False
	return net

network_keys = ["bl", "bl-pv", "bl-ev-uc", "bl-ev-edf", "bl-ev-lf", "bl-ev-pv-uc", "bl-ev-pv-edf", "bl-ev-pv-lf"]
networks = {network_key : process_network(pn.create_cigre_network_mv()) for network_key in network_keys}

output_loads = {network_key : [] for network_key in network_keys}
output_buses = {network_key : [] for network_key in network_keys}
output_lines = {network_key : [] for network_key in network_keys}
output_trafos = {network_key : [] for network_key in network_keys}
for i in range(len_sim):
    print("timestep {0}".format(i))
    outloads = {}
    outloads["bl"] = {'caltech' : 0, 'jpl' : 0, 'office_load' : mo_data[i], 'pv' : 0}
    outloads["bl-pv"] = {'caltech' : 0, 'jpl' : 0, 'office_load' : mo_data[i], 'pv' : pv_data[i]}
    outloads["bl-ev-uc"] = {'caltech' : sim_unc[i], 'jpl' : sim_unc_jpl[i], 'office_load' : mo_data[i], 'pv' : 0}
    outloads["bl-ev-edf"] = {'caltech' : sim_edf[i], 'jpl' : sim_edf_jpl[i], 'office_load' : mo_data[i], 'pv' : 0}
    outloads["bl-ev-lf"] = {'caltech' : sim_lf[i], 'jpl' : sim_lf_jpl[i], 'office_load' : mo_data[i], 'pv' : 0}
    outloads["bl-ev-pv-uc"] = {'caltech' : sim_unc[i], 'jpl' : sim_unc_jpl[i], 'office_load' : mo_data[i], 'pv' : pv_data[i]}
    outloads["bl-ev-pv-edf"] = {'caltech' : sim_edf[i], 'jpl' : sim_edf_jpl[i], 'office_load' : mo_data[i], 'pv' : pv_data[i]}
    outloads["bl-ev-pv-lf"] = {'caltech' : sim_lf[i], 'jpl' : sim_lf_jpl[i], 'office_load' : mo_data[i], 'pv' : pv_data[i]}
    for key in network_keys:
        staged_net = assign_loads(networks[key], outloads[key])
        snapshot = deepcopy(staged_net)
        try:
            pp.runpp(staged_net)
        except:
            print(key)
            print(outloads[key])
            print(outloads["bl-ev-lf"])
            diag_results = pp.diagnostic(snapshot, overload_scaling_factor=0.5)
            print(diag_results)
            input()
        output_loads[key].append(pp.to_json(staged_net.res_load))
        output_buses[key].append(pp.to_json(staged_net.res_bus))
        output_lines[key].append(pp.to_json(staged_net.res_line))
        output_trafos[key].append(pp.to_json(staged_net.res_trafo))
    if i % 200 == 0:
        write_results()
write_results()

# output_loads, output_buses, output_lines, output_trafos = read_results()
# for elt in [output_loads, output_buses, output_lines, output_trafos]:
#     elt = recover_pp_network(elt)



# # Plot trafo utilization vs time for bl ev edf, bl ev lf, bl ev pv edf, bl ev pv lf
# trafo_bl_ev_edf = np.array([output_trafos['bl-ev-edf'][t].loading_percent])
# fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
# ax[0].plot(acnsim.aggregate_current(sim), label='Our EDF')
# ax[1].plot(acnsim.aggregate_current(sim2), label='Included EDF')
# ax[1].set_xlabel('Time (periods)')
# ax[0].set_ylabel('Current (A)')
# ax[1].set_ylabel('Current (A)')
# ax[0].set_title('Our EDF')
# ax[1].set_title('Included EDF')
# plt.show()
