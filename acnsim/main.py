from simulator import Simulator
from network import get_caltech_acn
from event_queue import EventQueue
from utils.generate_events import generate_test_case_api
from contrib.basic_network import get_caltech_cs
from contrib.acn_opt_algorithm import ProfitMaximization
from datetime import datetime
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from signals.prices import TOUPrices

simulation_name = 'Demo'

API_KEY = 'Ln4QoRJ_lcBUnZ0aU98BTpnY-AWIkByYoYyoGNIsSjw'

# client = pymongo.MongoClient(host='131.215.140.211', username='DataWriter', password='acndata')
# options = bson.codec_options.CodecOptions(tz_aware=True)
# col = client.CaltechACN.get_collection('sessions', codec_options=options)

start = datetime(2018, 9, 1).astimezone()
end = datetime(2018, 9, 2).astimezone()

period = 5  # minute
voltage = 220  # volts

# Network
cn = get_caltech_acn('BASIC')
cs = get_caltech_cs()

# Scheduling Algorithm
sch = ProfitMaximization(cs, force_unique=True)

# Event Queue
events = EventQueue()
events.add_events(generate_test_case_api(API_KEY, start, end))
events2 = deepcopy(events)

# Pricing
pricing = TOUPrices(period, start, voltage)

sim = Simulator(cn, sch, events, start,  prices=pricing, store_schedule_history=True)
sim.run()
agg_current = sum(np.array(rates) for rates in sim.charging_rates.values())
plt.plot(agg_current)
plt.show()