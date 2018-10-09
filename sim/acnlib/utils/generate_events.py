from acnlib.event_queue import PluginEvent


def _datetime_to_timestamp(dt, period):
    return int(dt.timestamp() / (60 * period))


def generate_test_case_mongo(collection, start, end, voltage=220, max_rate=32, period=1):
    docs = collection.find({'connectionTime': {'$gte': start, '$lt': end}})
    events = []
    min_timestamp = float('inf')
    for d in docs:
        ev_params = {'arrive': _datetime_to_timestamp(d['connectionTime'], period),
                     'depart': _datetime_to_timestamp(d['disconnectTime'], period),
                     'requested_energy': d['kWhDelivered'] * (60/period) / voltage,  # A*periods
                     'max_rate': max_rate,
                     'session_id': d['sessionID'],
                     'station_id': d['spaceID']
                     }
        timestamp = ev_params['arrive']
        events.append(PluginEvent(timestamp, ev_params))
        min_timestamp = min(timestamp, min_timestamp)
    for e in events:
        e.timestamp -= min_timestamp
        e.ev_params['arrive'] -= min_timestamp
        e.ev_params['depart'] -= min_timestamp
    return events