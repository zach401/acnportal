from acnlib.event_queue import PluginEvent
from acnlib.models.ev import EV
from acnlib.models.battery import Battery


def datetime_to_timestamp(dt, period):
    return int(dt.timestamp() / (60 * period))


def get_sessions_mongo(collection, start, end, voltage=220, max_rate=32, period=1, max_len=float('inf')):
    docs = collection.find({'connectionTime': {'$gte': start, '$lt': end}})
    start_offset = datetime_to_timestamp(start, period)
    sessions = []
    for d in docs:
        arrival = datetime_to_timestamp(d['connectionTime'], period) - start_offset
        departure = datetime_to_timestamp(d['disconnectTime'], period) - start_offset
        if departure - arrival > max_len:
            departure = arrival + max_len
        requested_energy = d['kWhDelivered'] * 1000 * (60/period) / voltage  # A*periods
        max_rate = max_rate
        session_id = d['sessionID']
        station_id = d['spaceID']
        batt = Battery(requested_energy, 0, max_rate)
        ev = EV(arrival, departure, requested_energy, station_id, session_id, batt)
        sessions.append(ev)
    return sessions


def generate_test_case_mongo(collection, start, end, voltage=220, max_rate=32, period=1):
    sessions = get_sessions_mongo(collection, start, end, voltage, max_rate, period)
    events = [PluginEvent(sess.arrival, sess) for sess in sessions]
    return events