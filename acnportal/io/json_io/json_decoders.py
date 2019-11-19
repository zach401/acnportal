import json
from acnportal import acnsim
from acnportal import algorithms
import numpy as np
from datetime import datetime

# TODO: load JSON from file instead of just from JSON string.
class InvalidJSONError(Exception):
    """
    Exception which is raised when trying to read or write a JSON
    object whose class has not been registered.
    """
    pass

class RegistrationError(Exception):
    """
    Exception which is raised when trying to register a class with
    invalid parameters.
    """
    pass

DECODER_REGISTRY = {}

def from_json(in_json):
    """
    Converts a JSON representation of an object into an instance of
    the class. If the object class is registered with a decoder, that
    decoder is used to decode the object. If not, an InvalidJSONError
    is raised.

    Args:
        obj_json (JSON str): JSON string from which to construct the
            object.

    Returns:
        Object

    Raises:
        InvalidJSONError: Exception raised when object whose class
            has not yet been registered is attempted to be
            read from JSON.
    """
    json_dict = json.loads(in_json)

    # If the object does not have a class attribute in the json_dict,
    # acnsim's decoding will not work with it. An InvalidJSONError is
    # raised.
    if 'class' not in json_dict:
        raise InvalidJSONError(f"Decoded JSON has no 'class' "
            f"attribute for decoder selection. Use JSON loading "
            f"directly or check JSON source.")

    # If the object has a class attribute but the class is not
    # registered with a decoder, an InvalidJSONError is raised. The
    # user may need to register the class to decode.
    if json_dict['class'] not in DECODER_REGISTRY:
        raise InvalidJSONError(f"Class {json_dict['class']} to "
            f"decode is not registered with a decoder.")

    decoder = DECODER_REGISTRY[json_dict['class']]
    return decoder(json_dict['args'])


# ====================================================================
# Decoders
# Battery decoders
def battery_from_json(instr):
    in_dict = json.loads(instr)

    # Note second arg in below construction is a placeholder
    out_obj = acnsim.Battery(
        in_dict['_capacity'], 0, in_dict['_max_power'])

    out_obj._current_charging_power = \
        in_dict['_current_charging_power']
    out_obj._current_charge = in_dict['_current_charge']

    return out_obj
DECODER_REGISTRY['Battery'] = battery_from_json

def linear_2_stage_battery_from_json(instr):
    in_dict = json.loads(instr)

    # Note second arg in below construction is a placeholder
    out_obj = acnsim.Linear2StageBattery(
        in_dict['_capacity'], 0, in_dict['_max_power'],
        noise_level=in_dict['_noise_level'],
        transition_soc=in_dict['_transition_soc']
    )

    out_obj._current_charging_power = \
        in_dict['_current_charging_power']
    out_obj._current_charge = in_dict['_current_charge']

    return out_obj
DECODER_REGISTRY['Linear2StageBattery'] = \
    linear_2_stage_battery_from_json

# EV decoders
def ev_from_json(instr):
    in_dict = json.loads(instr)

    battery = from_json(in_dict['_battery'])
    out_obj = acnsim.models.EV(
        in_dict['_arrival'],
        in_dict['_departure'],
        in_dict['_requested_energy'],
        in_dict['_station_id'],
        in_dict['_session_id'],
        battery,
        estimated_departure=in_dict['_estimated_departure']
    )

    out_obj._energy_delivered = in_dict['_energy_delivered']
    out_obj._current_charging_rate = \
        in_dict['_current_charging_rate']

    return out_obj
DECODER_REGISTRY['EV'] = ev_from_json

# EVSE decoders
def evse_from_json(instr):
    in_dict = json.loads(instr)

    if in_dict['_ev'] is not None:
        ev = from_json(in_dict['_ev'])
    else:
        ev = None

    out_obj = acnsim.EVSE(
        in_dict['_station_id'],
        max_rate=in_dict['_max_rate'],
        min_rate=in_dict['_min_rate']
    )

    out_obj._current_pilot = in_dict['_current_pilot']
    out_obj._ev = ev
    out_obj.is_continuous = True

    return out_obj
DECODER_REGISTRY['EVSE'] = evse_from_json

def deadband_evse_from_json(instr):
    in_dict = json.loads(instr)

    if in_dict['_ev'] is not None:
        ev = from_json(in_dict['_ev'])
    else:
        ev = None

    out_obj = acnsim.DeadbandEVSE(
        in_dict['_station_id'],
        deadband_end=in_dict['_deadband_end'],
        max_rate=in_dict['_max_rate'],
        min_rate=in_dict['_min_rate']
    )

    out_obj._current_pilot = in_dict['_current_pilot']
    out_obj._ev = ev
    out_obj.is_continuous = True

    return out_obj
DECODER_REGISTRY['DeadbandEVSE'] = deadband_evse_from_json

def finite_rates_evse_from_json(instr):
    in_dict = json.loads(instr)

    if in_dict['_ev'] is not None:
        ev = from_json(in_dict['_ev'])
    else:
        ev = None

    out_obj = acnsim.FiniteRatesEVSE(
        in_dict['_station_id'],
        in_dict['allowable_rates']
    )

    out_obj._min_rate = in_dict['_min_rate']
    out_obj._current_pilot = in_dict['_current_pilot']
    out_obj._ev = ev
    out_obj.is_continuous = False

    return out_obj
DECODER_REGISTRY['FiniteRatesEVSE'] = finite_rates_evse_from_json

# Event decoders
def event_from_json(instr):
    in_dict = json.loads(instr)

    out_obj = acnsim.Event(in_dict['timestamp'])
    out_obj.type = in_dict['type']
    out_obj.precedence = in_dict['precedence']

    return out_obj
DECODER_REGISTRY['Event'] = event_from_json

def plugin_event_from_json(instr):
    in_dict = json.loads(instr)

    out_obj = acnsim.PluginEvent(
        in_dict['timestamp'], from_json(in_dict['ev']))

    out_obj.type = in_dict['type']
    out_obj.precedence = in_dict['precedence']

    return out_obj
DECODER_REGISTRY['PluginEvent'] = plugin_event_from_json

def unplug_event_from_json(instr):
    in_dict = json.loads(instr)

    out_obj = acnsim.UnplugEvent(
        in_dict['timestamp'], in_dict['station_id'], 
        in_dict['session_id'])

    out_obj.type = in_dict['type']
    out_obj.precedence = in_dict['precedence']

    return out_obj
DECODER_REGISTRY['UnplugEvent'] = unplug_event_from_json

def recompute_event_from_json(instr):
    in_dict = json.loads(instr)

    out_obj = acnsim.RecomputeEvent(in_dict['timestamp'])

    out_obj.type = in_dict['type']
    out_obj.precedence = in_dict['precedence']

    return out_obj
DECODER_REGISTRY['RecomputeEvent'] = recompute_event_from_json

def event_queue_from_json(instr):
    in_dict = json.loads(instr)

    out_obj = acnsim.EventQueue()
    out_obj._queue = [(ts, from_json(event))
        for (ts, event) in in_dict['_queue']]
    out_obj._timestep = in_dict['_timestep']

    return out_obj
DECODER_REGISTRY['EventQueue'] = event_queue_from_json

def charging_network_from_json(instr):
    in_dict = json.loads(instr)

    out_obj = acnsim.ChargingNetwork()

    out_obj._EVSEs = {station_id : from_json(evse)
        for station_id, evse in in_dict['_EVSEs'].items()}

    out_obj.constraint_matrix = \
        np.array(in_dict['constraint_matrix'])
    out_obj.magnitudes = \
        np.array(in_dict['magnitudes'])

    out_obj.constraint_index = in_dict['constraint_index']
    out_obj._voltages = np.array(in_dict['_voltages'])
    out_obj._phase_angles = np.array(in_dict['_phase_angles'])

    return out_obj
DECODER_REGISTRY['ChargingNetwork'] = charging_network_from_json

def simulator_from_json(instr):
    in_dict = json.loads(instr)

    network = from_json(in_dict['network'])
    assert isinstance(network, acnsim.ChargingNetwork)

    events = from_json(in_dict['event_queue'])
    assert isinstance(events, acnsim.EventQueue)

    # TODO: Add option to actually initialize scheduler.
    # scheduler = in_dict['scheduler']

    out_obj = acnsim.Simulator(
        network,
        algorithms.BaseAlgorithm(),
        events,
        datetime.fromisoformat(in_dict['start']),
        period=in_dict['period'],
        signals=in_dict['signals'],
        verbose=in_dict['verbose']
    )

    # TODO: Overwriting scheduler with string. Have an info attr in
    # Simulator instead.
    # out_obj.scheduler = scheduler 

    out_obj._iteration = in_dict['_iteration']
    out_obj.peak = in_dict['peak']

    out_obj.pilot_signals = np.array(in_dict['pilot_signals'])
    out_obj.charging_rates = np.array(in_dict['charging_rates'])

    out_obj.ev_history = {session_id : from_json(ev) 
        for session_id, ev in in_dict['ev_history'].items()}
    out_obj.event_history = [from_json(event) 
        for event in in_dict['event_history']]

    return out_obj
DECODER_REGISTRY['Simulator'] = simulator_from_json