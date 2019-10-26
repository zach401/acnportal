import pandapower as pp
from pandapower import networks as pn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

net = pn.panda_four_load_branch()

file_name = "Actual_34.15_-118.15_2006_DPV_74MW_5_Min.csv"
actual_pv_cap_mw = 74
pv_cap = 0.3

# Read in NREL PV data for the relevant dates: first cut off all earlier dates:
pv_data = pd.read_csv(file_name).set_index("LocalTime")["09/05/06 00:00":].to_numpy()
# Now get data for sim_len 5-minute time steps into the future
pv_data = pv_data[:12*24] / actual_pv_cap_mw * pv_cap

voltages = []
transformer_load = []
external_grid = []
for i in range(len(pv_data)):
    net.load['p_mw'][2] = -pv_data[i]
    net.load['q_mvar'] = 0
    pp.runpp(net)
    voltages.append(net.res_bus['vm_pu'])
    transformer_load.append(net.res_trafo['loading_percent'])
    external_grid.append(net.res_ext_grid['p_mw'])

pd.DataFrame(voltages).plot()
plt.show()

pd.DataFrame(external_grid).plot()
plt.show()