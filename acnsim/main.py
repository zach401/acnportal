from copy import deepcopy
from datetime import datetime

import numpy as np
from algorithms import UncontrolledCharging
from events import EventQueue
from matplotlib import pyplot as plt
from network.sites import CaltechACN
from signals.prices import TOUPrices
from simulator import Simulator
from utils.generate_events import generate_test_case_api

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
cn = CaltechACN()

# Scheduling Algorithm
# sch = ProfitMaximization(cs, force_unique=True)
sch = UncontrolledCharging()

# Event Queue
events = EventQueue()
events.add_events(generate_test_case_api(API_KEY, start, end))
events2 = deepcopy(events)

# Pricing
pricing = TOUPrices(period, start, voltage)

sim = Simulator(cn, sch, events, start, prices=pricing, store_schedule_history=True)
sim.run()
agg_current = sum(np.array(rates) for rates in sim.charging_rates.values())
plt.plot(agg_current)
plt.show()
