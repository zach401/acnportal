from acnlib.simulator import Simulator
from acnlib.network import get_caltech_acn_basic, ChargingNetwork
from acnlib.event_queue import EventQueue, PluginEvent
from contrib.acn_opt_algorithm import ProfitMaximization
from acnlib.utils.generate_events import generate_test_case_mongo
from acnlib.models.battery import Battery
from acnlib.models.ev import EV
from acnlib.models.evse import EVSE, get_EVSE_by_type
from acnlib.models.prices import TOUPrices
import pymongo
import bson
from datetime import datetime
import pickle
import numpy as np
from matplotlib import pyplot as plt


simulation_name = 'Minimize_Price_PA_constraints_basic_infrastructure_Sept1_Sept7'

client = pymongo.MongoClient(host='131.215.140.211', username='DataWriter', password='acndata')
options = bson.codec_options.CodecOptions(tz_aware=True)
col = client.CaltechACN.get_collection('sessions', codec_options=options)

start = datetime(2018, 9, 1).astimezone()
end = datetime(2018, 9, 7).astimezone()

period = 1  # minute
voltage = 220  # volts

# Network
cn = get_caltech_acn_basic(200)
# cn = ChargingNetwork(40)
# cn.register_evse(get_EVSE_by_type('CA-148', 'BASIC'))
# cn.register_evse(get_EVSE_by_type('CA-149', 'BASIC'))

# Scheduling Algorithm
sch = ProfitMaximization()

# Event Queue
events = EventQueue()
events.add_events(generate_test_case_mongo(col, start, end))
# events.add_event(PluginEvent(5, EV(5, 180, 400, 'CA-148', '0001', Battery(400, 0, 32))))
# events.add_event(PluginEvent(7, EV(7, 120, 400, 'CA-149', '0002', Battery(400, 0, 32))))

# Pricing
pricing = TOUPrices(1, start, 6e4/period/voltage)

sim = Simulator(cn, sch, events, start,  pricing, store_schedule_history=True)
sim.run()
agg_current = sum(np.array(rates) for rates in sim.charging_rates.values())
plt.plot(agg_current)
plt.show()