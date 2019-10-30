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

voltage = 208
# We build a pp network based on the Low Voltage CIGRE network included with
# pandapower. We only use the commercial subnetwork and modify some of the
# voltages. Much of this code is copied from pandapower/networks/
# cigre_networks.py
def create_cigre_lv_commercial(name=None):
    net_cigre_lv = pp.create_empty_network(name=name)

    # Linedata
    # OH1
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.4917,
                 'x_ohm_per_km': 0.2847, 'max_i_ka': 10.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH1', element='line')

    # OH2
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 1.3207,
                 'x_ohm_per_km': 0.321, 'max_i_ka': 10.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH2', element='line')

    # OH3
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 2.0167,
                 'x_ohm_per_km': 0.3343, 'max_i_ka': 10.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH3', element='line')

    # Busses
    bus0 = pp.create_bus(net_cigre_lv, name='Bus 0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    busC0 = pp.create_bus(net_cigre_lv, name='Bus C0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    busC1 = pp.create_bus(net_cigre_lv, name='Bus C1', vn_kv=0.4, type='b', zone='CIGRE_LV')
    busC2 = pp.create_bus(net_cigre_lv, name='Bus C2', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC3 = pp.create_bus(net_cigre_lv, name='Bus C3', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC4 = pp.create_bus(net_cigre_lv, name='Bus C4', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC5 = pp.create_bus(net_cigre_lv, name='Bus C5', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC6 = pp.create_bus(net_cigre_lv, name='Bus C6', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC7 = pp.create_bus(net_cigre_lv, name='Bus C7', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC8 = pp.create_bus(net_cigre_lv, name='Bus C8', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC9 = pp.create_bus(net_cigre_lv, name='Bus C9', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC10 = pp.create_bus(net_cigre_lv, name='Bus C10', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC11 = pp.create_bus(net_cigre_lv, name='Bus C11', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC12 = pp.create_bus(net_cigre_lv, name='Bus C12', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC13 = pp.create_bus(net_cigre_lv, name='Bus C13', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC14 = pp.create_bus(net_cigre_lv, name='Bus C14', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC15 = pp.create_bus(net_cigre_lv, name='Bus C15', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC16 = pp.create_bus(net_cigre_lv, name='Bus C16', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC17 = pp.create_bus(net_cigre_lv, name='Bus C17', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC18 = pp.create_bus(net_cigre_lv, name='Bus C18', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC19 = pp.create_bus(net_cigre_lv, name='Bus C19', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC20 = pp.create_bus(net_cigre_lv, name='Bus C20', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC21 = pp.create_bus(net_cigre_lv, name='Bus C21', vn_kv=0.4, type='m', zone='CIGRE_LV')

    # Lines
    pp.create_line(net_cigre_lv, busC1, busC2, length_km=0.030, std_type='OH1',
                   name='Line C1-C2')
    pp.create_line(net_cigre_lv, busC2, busC3, length_km=0.030, std_type='OH1',
                   name='Line C2-C3')
    pp.create_line(net_cigre_lv, busC3, busC4, length_km=0.030, std_type='OH1',
                   name='Line C3-C4')
    pp.create_line(net_cigre_lv, busC4, busC5, length_km=0.030, std_type='OH1',
                   name='Line C4-C5')
    pp.create_line(net_cigre_lv, busC5, busC6, length_km=0.030, std_type='OH1',
                   name='Line C5-C6')
    pp.create_line(net_cigre_lv, busC6, busC7, length_km=0.030, std_type='OH1',
                   name='Line C6-C7')
    pp.create_line(net_cigre_lv, busC7, busC8, length_km=0.030, std_type='OH1',
                   name='Line C7-C8')
    pp.create_line(net_cigre_lv, busC8, busC9, length_km=0.030, std_type='OH1',
                   name='Line C8-C9')
    pp.create_line(net_cigre_lv, busC3, busC10, length_km=0.030, std_type='OH2',
                   name='Line C3-C10')
    pp.create_line(net_cigre_lv, busC10, busC11, length_km=0.030, std_type='OH2',
                   name='Line C10-C11')
    pp.create_line(net_cigre_lv, busC11, busC12, length_km=0.030, std_type='OH3',
                   name='Line C11-C12')
    pp.create_line(net_cigre_lv, busC11, busC13, length_km=0.030, std_type='OH3',
                   name='Line C11-C13')
    pp.create_line(net_cigre_lv, busC10, busC14, length_km=0.030, std_type='OH3',
                   name='Line C10-C14')
    pp.create_line(net_cigre_lv, busC5, busC15, length_km=0.030, std_type='OH2',
                   name='Line C5-C15')
    pp.create_line(net_cigre_lv, busC15, busC16, length_km=0.030, std_type='OH2',
                   name='Line C15-C16')
    pp.create_line(net_cigre_lv, busC15, busC17, length_km=0.030, std_type='OH3',
                   name='Line C15-C17')
    pp.create_line(net_cigre_lv, busC16, busC18, length_km=0.030, std_type='OH3',
                   name='Line C16-C18')
    pp.create_line(net_cigre_lv, busC8, busC19, length_km=0.030, std_type='OH3',
                   name='Line C8-C19')
    pp.create_line(net_cigre_lv, busC9, busC20, length_km=0.030, std_type='OH3',
                   name='Line C9-C20')
    pp.create_line(net_cigre_lv, busC7, busC21, length_km=0.030, std_type='OH3',
                   name='Line C7-C21')


    # Trafos
    pp.create_transformer_from_parameters(net_cigre_lv, busC0, busC1, sn_mva=100.0, vn_hv_kv=20.0,
                                          vn_lv_kv=0.4, vkr_percent=0.993750, vk_percent=4.115529,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=30.0,
                                          tap_pos=0.0, name='Trafo C0-C1')

    # External grid
    pp.create_ext_grid(net_cigre_lv, bus0, vm_pu=1.0, va_degree=0.0, s_sc_max_mva=100.0,
                       s_sc_min_mva=100.0, rx_max=1.0, rx_min=1.0)
# [mo_data[i], sim_unc[i], pv_data[i], mo_data[i], sim_unc_jpl[i], pv_data[i], wh_data[i], pv_data[i]]
    # Loads - Initialized to 0, but will be set depending on experiment run
    pp.create_load(net_cigre_lv, busC12, p_mw=0, name='Load C12 - Med Office 1')
    pp.create_load(net_cigre_lv, busC13, p_mw=0, name='Load C13 - Caltech ACN')
    pp.create_load(net_cigre_lv, busC14, p_mw=0, name='Load C14 - PV Array 1')
    pp.create_load(net_cigre_lv, busC17, p_mw=0, name='Load C17 - Med Office 2')
    pp.create_load(net_cigre_lv, busC18, p_mw=0, name='Load C18 - JPL ACN')
    pp.create_load(net_cigre_lv, busC19, p_mw=0, name='Load C19 - PV Array 2')
    pp.create_load(net_cigre_lv, busC20, p_mw=0, name='Load C20 - Warehouse')
    pp.create_load(net_cigre_lv, busC21, p_mw=0, name='Load C21 - PV Array 3')

    # Switches
    pp.create_switch(net_cigre_lv, bus0, busC0, et='b', closed=True, type='CB', name='S3')

    return net_cigre_lv

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
file_name_mo = "RefBldgMediumOfficeNew2004_7.1_5.0_3B_USA_CA_LOS_ANGELES.csv"

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

# sim_lf_pv, sim_lf_pv_jpl = run_sim(AdaChargeLoadFlattening, external_signal=mo_data_full[load_start_idx:]+pv_data_full[load_start_idx:], info="lf_bl_pv")
# sim_lf, sim_lf_jpl = run_sim(AdaChargeLoadFlattening, external_signal=mo_data_full[load_start_idx:], info="lf_bl")
def load_from_pkl(name):
    with open(name, 'rb') as infile:
        sim = pkl.load(infile)
    return acnsim.aggregate_current(sim) * voltage / 1000 / 1000
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
    names =  ["Load C12 - Med Office 1", "Load C13 - Caltech ACN", "Load C14 - PV Array 1",
        "Load C17 - Med Office 2", "Load C18 - JPL ACN", "Load 19 - PV Array 2", 
        "Load 20 - Warehousee", "Load 21 - PV Array 3"]
    for i in range(len(input_loads)):
        network.load["p_mw"][network.load["name"] == names[i]] = input_loads[i]
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

network_keys = ["bl", "bl-pv", "bl-ev-uc", "bl-ev-edf", "bl-ev-lf", "bl-ev-pv-uc", "bl-ev-pv-edf", "bl-ev-pv-lf"]
networks = {network_key : create_cigre_lv_commercial(name=network_key) for network_key in network_keys}
output_loads = {network_key : [] for network_key in network_keys}
output_buses = {network_key : [] for network_key in network_keys}
output_lines = {network_key : [] for network_key in network_keys}
output_trafos = {network_key : [] for network_key in network_keys}
for i in range(80, len_sim):
    print("timestep {0}".format(i))
    outloads = {}
    outloads["bl"] = [mo_data[i], 0, 0, mo_data[i], 0, 0, wh_data[i], 0]
    outloads["bl-pv"] = [mo_data[i], 0, pv_data[i], mo_data[i], pv_data[i], wh_data[i], pv_data[i]]
    outloads["bl-ev-uc"] = [mo_data[i], sim_unc[i], 0, mo_data[i], sim_unc_jpl[i], 0, wh_data[i], 0]
    outloads["bl-ev-edf"] = [mo_data[i], sim_edf[i], 0, mo_data[i], sim_edf_jpl[i], 0, wh_data[i], 0]
    outloads["bl-ev-lf"] = [mo_data[i], sim_lf[i], 0, mo_data[i], sim_lf_jpl[i], 0, wh_data[i], 0]
    outloads["bl-ev-pv-uc"] = [mo_data[i], sim_unc[i], pv_data[i], mo_data[i], sim_unc_jpl[i], pv_data[i], wh_data[i], pv_data[i]]
    outloads["bl-ev-pv-edf"] = [mo_data[i], sim_edf[i], pv_data[i], mo_data[i], sim_edf_jpl[i], pv_data[i], wh_data[i], pv_data[i]]
    outloads["bl-ev-pv-lf"] = [mo_data[i], sim_lf_pv[i], pv_data[i], mo_data[i], sim_lf_pv_jpl[i], pv_data[i], wh_data[i], pv_data[i]]
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

output_loads, output_buses, output_lines, output_trafos = read_results()
for elt in [output_loads, output_buses, output_lines, output_trafos]:
    elt = recover_pp_network(elt)



# Plot trafo utilization vs time for bl ev edf, bl ev lf, bl ev pv edf, bl ev pv lf
trafo_bl_ev_edf = np.array([output_trafos['bl-ev-edf'][t].loading_percent])
fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
ax[0].plot(acnsim.aggregate_current(sim), label='Our EDF')
ax[1].plot(acnsim.aggregate_current(sim2), label='Included EDF')
ax[1].set_xlabel('Time (periods)')
ax[0].set_ylabel('Current (A)')
ax[1].set_ylabel('Current (A)')
ax[0].set_title('Our EDF')
ax[1].set_title('Included EDF')
plt.show()

# with open("network_results.json", "r") as infile:
#     output_networks_loaded = json.load(infile)

# for key in output_networks.keys():
#     unch_input = output_networks[key]
#     for i in range(len(unch_input)):
#         unch_input[i] = pp.from_json(unch_input[i])
#     output_networks[key] = unch_input

# external_res = [net.res_ext_grid.p_mw for net in output_networks["bl-ev-pv-lf"]]
# plt.plot(external_res)
# plt.show()