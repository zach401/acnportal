import math
from datetime import datetime

from ..models.ev import EV
from ..models.battery import Battery
from . import PluginEvent
from .event_queue import EventQueue
from acnportal.acndata import DataClient


def generate_events(token, site, start, end, period, voltage, max_rate, **kwargs):
    """ Return EventQueue filled using events gathered from the acndata API.

    Args:
        See get_evs().

    Returns:
        EventQueue: An EventQueue filled with Events gathered through the acndata API.

    """
    evs = get_evs(token, site, start, end, period, voltage, max_rate, **kwargs)
    events = [PluginEvent(sess.arrival, sess) for sess in evs]
    return EventQueue(events)


def get_evs(
    token,
    site,
    start: datetime,
    end: datetime,
    period,
    voltage,
    max_battery_power,
    max_len=None,
    battery_params=None,
    force_feasible=False,
):
    """ Return a list of EVs gathered from the acndata API.

    Args:
        token (str): API token needed to access the acndata API.
        site (str): ACN id for the site where data should be gathered.
        start (datetime): Only return sessions which began after start.
        end (datetime): Only return session which began before end.
        period (int): Length of each time interval. (minutes)
        voltage (float): Voltage of the network.
        max_battery_power (float): Default maximum charging power for batteries.
        max_len (int): Maximum length of a session. (periods) Default None.
        battery_params (Dict[str, object]): Dictionary containing parameters for the EV's battery. Three keys are
            supported. If none, Battery type is used with default configuration. Default None.
            - 'type' maps to a Battery-like class. (required)
            - 'capacity_fn' maps to a function which takes in the the energy delivered to the car, the length of the
                session, the period of the simulation, and the voltage of the system. It should return a tuple with
                the capacity of the battery and the initial charge of the battery both in A*periods.
            - 'kwargs' maps to a dictionary of keyword arguments which will be based to the Battery constructor.
        force_feasible (bool): If True, the requested_energy of each session will be reduced if it exceeds the amount
            of energy which could be delivered at maximum rate during the duration of the charging session.
            Default False.
    Returns:

    """
    client = DataClient(token)
    docs = client.get_sessions_by_time(site, start, end)
    evs = []
    offset = _datetime_to_timestamp(start, period)
    for d in docs:
        evs.append(
            _convert_to_ev(
                d,
                offset,
                period,
                voltage,
                max_battery_power,
                max_len,
                battery_params,
                force_feasible,
            )
        )
    return evs


def _convert_to_ev(
    d,
    offset,
    period,
    voltage,
    max_battery_power,
    max_len=None,
    battery_params=None,
    force_feasible=False,
):
    """ Convert a json document for a single charging session from acndata into an EV object.

    Args:
        d (dict): Session expressed as a dictionary. See acndata API for more details.
        offset (int): Simulation timestamp of the beginning of the simulation.
        See get_evs() for additional args.

    Returns:
        EV: EV object with data from the acndata session doc.
    """
    arrival = _datetime_to_timestamp(d["connectionTime"], period) - offset
    departure = _datetime_to_timestamp(d["disconnectTime"], period) - offset

    if max_len is not None and departure - arrival > max_len:
        departure = arrival + max_len

    # requested_energy = d['kWhDelivered'] * 1000 * (60 / period) / voltage  # A*periods

    if force_feasible:
        delivered_energy = min(
            d["kWhDelivered"], max_battery_power * (departure - arrival) * (period / 60)
        )
    else:
        delivered_energy = d["kWhDelivered"]

    session_id = d["sessionID"]
    station_id = d["spaceID"]

    if battery_params is None:
        battery_params = {"type": Battery}
    batt_kwargs = battery_params["kwargs"] if "kwargs" in battery_params else {}
    if "capacity_fn" in battery_params:
        cap, init = battery_params["capacity_fn"](
            delivered_energy, departure - arrival, voltage, period
        )
    else:
        cap = delivered_energy
        init = 0
    batt = battery_params["type"](cap, init, max_battery_power, **batt_kwargs)

    # delivered_energy_amp_periods = delivered_energy * 1000 * (60 / period) / voltage
    return EV(arrival, departure, delivered_energy, station_id, session_id, batt)


def _datetime_to_timestamp(dt, period, round_up=False):
    """ Convert a datetime object to a timestamp measured in simulation periods.

    Args:
        dt (datetime): Datetime to be converted to a simulation timestamp.
        period (int): Length of one time interval in the simulation. (minutes)
        round_up (bool): If True, round up when casting timestamp to int, else round down.

    Returns:
        int: dt expressed as a simulation timestamp.
    """
    ts = dt.timestamp() / (60 * period)
    if round_up:
        return int(math.ceil(ts))
    else:
        return int(ts)
