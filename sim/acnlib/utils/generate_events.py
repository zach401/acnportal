from acnlib.event_queue import PluginEvent
from acnlib.models.ev import EV
from acnlib.models.battery import Battery
import math


def datetime_to_timestamp(dt, period, round_up=False):
    ts = dt.timestamp() / (60 * period)
    if round_up:
        return int(math.ceil(ts))
    else:
        return int(ts)


def get_sessions_mongo(collection, start, end, voltage=220, max_rate=32, period=1, max_len=float('inf'),
                       force_feasible=False, battery_type=Battery, batt_args=None, battery_cap_fn=None):
    docs = collection.find({'connectionTime': {'$gte': start, '$lt': end}}, {'chargingCurrent': 0, 'pilotSignal': 0}).sort('connectionTime', 1)
    start_offset = datetime_to_timestamp(start, period)
    sessions = []
    for d in docs:
        arrival = datetime_to_timestamp(d['connectionTime'], period) - start_offset
        departure = datetime_to_timestamp(d['disconnectTime'], period) - start_offset
        if departure - arrival > max_len:
            departure = arrival + max_len
        requested_energy = d['kWhDelivered'] * 1000 * (60/period) / voltage  # A*periods
        max_rate = max_rate
        if force_feasible:
            requested_energy = min(requested_energy, max_rate*(departure - arrival))
        session_id = d['sessionID']
        station_id = d['spaceID']
        batt_args = {} if batt_args is None else batt_args
        if battery_cap_fn is None:
            cap = requested_energy
            init = 0
        else:
            cap, init = battery_cap_fn(requested_energy, departure - arrival, voltage, period)
        batt = battery_type(cap, init, max_rate, **batt_args)
        ev = EV(arrival, departure, requested_energy, station_id, session_id, batt)
        sessions.append(ev)
    return sessions


def generate_test_case_mongo(collection, start, end, voltage=220, max_rate=32, period=1, max_len=float('inf'),
                             force_feasible=False, battery_type=Battery, batt_args=None, battery_cap_fn=None):
    sessions = get_sessions_mongo(collection, start, end, voltage, max_rate, period, max_len, force_feasible=force_feasible,
                                  battery_type=battery_type, batt_args=batt_args, battery_cap_fn=battery_cap_fn)
    events = [PluginEvent(sess.arrival, sess) for sess in sessions]
    return events