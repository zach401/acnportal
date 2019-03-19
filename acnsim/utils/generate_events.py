import math

from events import PluginEvent
from models.battery import Battery
from models.ev import EV
from utils import c2_client


def datetime_to_timestamp(dt, period, round_up=False):
    """ Convert a datetime object to a timestamp measured in simulation periods.

    :param datetime.datetime dt: datetime to be converted to timestamp
    :param int period: length of one period in the _simulator
    :param bool round_up: whether the timestamp should be rounded up or down when casting to int
    :return: timestamp of the datetime in simulation periods.
    :rtype: int
    """
    ts = dt.timestamp() / (60 * period)
    if round_up:
        return int(math.ceil(ts))
    else:
        return int(ts)


def _convert_session_units(docs, start, voltage=220, max_rate=32, period=1, max_len=float('inf'),
                           force_feasible=False, battery_type=Battery, batt_args=None, battery_cap_fn=None):
    """ Convert iterable of sessions to _simulator units.

    :param Iterable(dict) docs: iterable of sessions as dicts.
    :param datetime.datetime start: only get sessions which start before this time.
    :param Union[int, float] voltage: voltage of the network used for calculating current from power.
    :param Union[int, float] max_rate: default maximum charging rate of the EV.
    :param int period: length of one simulation period.
    :param int max_len: maximum number of simulation periods a session can last. Truncate session if longer.
    :param bool force_feasible: adjust energy requested to be feasible at max rate in the allotted time.
    :param battery_type: subclass of Battery which should be used to create the EVs battery.
    :param dict batt_args: keyword arguments to be passed to the Battery constructor.
    :param function battery_cap_fn: function used to define the initial capacity of the battery.
    :return: list of dictionaries each of which describe a charging session.
    :rtype list(dict)
    """
    start_offset = datetime_to_timestamp(start, period)
    sessions = []
    for d in docs:
        arrival = datetime_to_timestamp(d['connectionTime'], period) - start_offset
        departure = datetime_to_timestamp(d['disconnectTime'], period) - start_offset
        if departure - arrival > max_len:
            departure = arrival + max_len
        requested_energy = d['kWhDelivered'] * 1000 * (60 / period) / voltage  # A*periods
        max_rate = max_rate
        if force_feasible:
            requested_energy = min(requested_energy, max_rate * (departure - arrival))
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


def get_sessions_mongo(collection, start, end, **kwargs):
    """ Get list of sessions from MongoDB and convert them to the _simulator units.

    :param pymongo.collection.Collection collection: pymongo collection from which session info should be retrieved.
    :param datetime.datetime start: only get sessions which start before this time.
    :param datetime.datetime end: only get sessions which start after this time.
    :param Union[int, float] voltage: voltage of the network used for calculating current from power.
    :param Union[int, float] max_rate: default maximum charging rate of the EV.
    :param int period: length of one simulation period.
    :param int max_len: maximum number of simulation periods a session can last. Truncate session if longer.
    :param bool force_feasible: adjust energy requested to be feasible at max rate in the allotted time.
    :param battery_type: subclass of Battery which should be used to create the EVs battery.
    :param dict batt_args: keyword arguments to be passed to the Battery constructor.
    :param function battery_cap_fn: function used to define the initial capacity of the battery.
    :return: list of dictionaries each of which describe a charging session.
    :rtype list(dict)
    """
    docs = collection.find({'connectionTime': {'$gte': start, '$lt': end}},
                           {'chargingCurrent': 0, 'pilotSignal': 0}).sort('connectionTime', 1)
    return _convert_session_units(docs, start, **kwargs)


def get_sessions_api(token, start, end, **kwargs):
    """ Get list of sessions from C2 API and convert them to the _simulator units.

    :param str token: API token needed to access the C2 API.
    :param datetime.datetime start: only get sessions which start before this time.
    :param datetime.datetime end: only get sessions which start after this time.
    :param Union[int, float] voltage: voltage of the network used for calculating current from power.
    :param Union[int, float] max_rate: default maximum charging rate of the EV.
    :param int period: length of one simulation period.
    :param int max_len: maximum number of simulation periods a session can last. Truncate session if longer.
    :param bool force_feasible: adjust energy requested to be feasible at max rate in the allotted time.
    :param battery_type: subclass of Battery which should be used to create the EVs battery.
    :param dict batt_args: keyword arguments to be passed to the Battery constructor.
    :param function battery_cap_fn: function used to define the initial capacity of the battery.
    :return: list of dictionaries each of which describe a charging session.
    :rtype list(dict)
    """
    docs = c2_client.get_sessions_by_time(token, start, end)
    return _convert_session_units(docs, start, **kwargs)


def generate_test_case_mongo(collection, start, end, **kwargs):
    """ Get list of PluginEvents for session sourced directly from MongoDB.

    :param pymongo.collection.Collection collection: pymongo collection from which session info should be retrieved.
    :param datetime.datetime start: only get sessions which start before this time.
    :param datetime.datetime end: only get sessions which start after this time.
    :param Union[int, float] voltage: voltage of the network used for calculating current from power.
    :param Union[int, float] max_rate: default maximum charging rate of the EV.
    :param int period: length of one simulation period.
    :param int max_len: maximum number of simulation periods a session can last. Truncate session if longer.
    :param bool force_feasible: adjust energy requested to be feasible at max rate in the allotted time.
    :param battery_type: subclass of Battery which should be used to create the EVs battery.
    :param dict batt_args: keyword arguments to be passed to the Battery constructor.
    :param function battery_cap_fn: function used to define the initial capacity of the battery.
    :return: list of PluginEvents for each session
    :rtype list(PluginEvent)
    """
    sessions = get_sessions_mongo(collection, start, end, **kwargs)
    events = [PluginEvent(sess.arrival, sess) for sess in sessions]
    return events


def generate_test_case_api(key, start, end, **kwargs):
    """ Get list of PluginEvents for session sourced from C2 API.

    :param str token: API token needed to access the C2 API.
    :param datetime.datetime start: only get sessions which start before this time.
    :param datetime.datetime end: only get sessions which start after this time.
    :param Union[int, float] voltage: voltage of the network used for calculating current from power.
    :param Union[int, float] max_rate: default maximum charging rate of the EV.
    :param int period: length of one simulation period.
    :param int max_len: maximum number of simulation periods a session can last. Truncate session if longer.
    :param bool force_feasible: adjust energy requested to be feasible at max rate in the allotted time.
    :param battery_type: subclass of Battery which should be used to create the EVs battery.
    :param dict batt_args: keyword arguments to be passed to the Battery constructor.
    :param function battery_cap_fn: function used to define the initial capacity of the battery.
    :return: list of dictionaries each of which describe a charging session.
    :rtype list(dict)
    """
    sessions = get_sessions_api(key, start, end, **kwargs)
    events = [PluginEvent(sess.arrival, sess) for sess in sessions]
    return events
