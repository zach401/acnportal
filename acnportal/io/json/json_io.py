import json
from acnportal import acnsim
from datetime import datetime, date
from dateutil import parser
import copy
import warnings
import numpy as np

# JSON Serialization module. All classes in acnsim are registered with
# encoder and decoder functions that detail how to write and read an
# acnsim object (e.g. Simulator, Network, EV, etc.) to and from a JSON
# file. If the user defines a new class, the user must register the
# new class for the new class to be JSON serializable.

# # TODO: Where should these registries go?
# ENCODER_REGISTRY = {
#     'Simulator' : simulator_to_json,
#     'Network' : network_to_json,
#     'EventQueue' : event_queue_to_json,
#     'Event' : event_to_json,
#     'PluginEvent' : plugin_event_to_json,
#     'UnplugEvent' : unplug_event_to_json,
#     'RecomputeEvent' : recompute_event_to_json,
#     'EVSE' : evse_to_json,
#     'DeadbandEVSE' : deadband_evse_to_json,
#     'FiniteRatesEVSE' : finite_evse_to_json,
#     'EV' : ev_to_json,
#     'Battery' : battery_to_json,
#     'Linear2StageBattery' : linear_2_stage_battery_to_json
# }

# DECODER_REGISTRY = {
#     'Simulator' : simulator_from_json,
#     'Network' : network_from_json,
#     'EventQueue' : event_queue_from_json,
#     'Event' : event_from_json,
#     'PluginEvent' : plugin_event_from_json,
#     'UnplugEvent' : unplug_event_from_json,
#     'RecomputeEvent' : recompute_event_from_json,
#     'EVSE' : evse_from_json,
#     'DeadbandEVSE' : deadband_evse_from_json,
#     'FiniteRatesEVSE' : finite_evse_from_json,
#     'EV' : ev_from_json,
#     'Battery' : battery_from_json,
#     'Linear2StageBattery' : linear_2_stage_battery_from_json
# }

def register_serializable_class(reg_cls, encoder=None, decoder=None):
    """
    Registers a class for JSON serialization, using user-provided
    encoder and decoder functions to handle reading/writing.

    If no encoder function is provided, the superclass is checked for
    membership in the registry; if the superclass is a member, the
    superclass encoder function is used. Otherwise, an error is
    thrown.

    If no decoder function is provided, the superclass is checked for
    membership in the registry; if the superclass is a member, the
    superclass decoder function is used. Otherwise, an error is
    thrown.

    Args:
        reg_cls (Class): Class to register with encoder/decoder
            registry.
        encoder (obj -> JSON str) : Takes as input an instance
            of reg_cls and returns a JSON string representing the
            instance.
        decoder (JSON str -> obj) : Takes as input a JSON string
            representing an instance of reg_cls and returns an
            instance of reg_cls.

    Returns:
        None

    Raises:
        RegistrationError: Exception raised when class registration
            fails.
    """
    assert ENCODER_REGISTRY.keys() == DECODER_REGISTRY.keys()
    class_name = reg_cls.__name__

    # Check if class is already registered
    if class_name in ENCODER_REGISTRY:
        warnings.warn(
            f"Class {class_name} has already been registered. "
            "Attempting to override previous encoder/decoder "
            "functions.")

    # If one of encoder/decoder is omitted, try using the superclass
    # encoder/decoder. If the superclass isn't registered, throw an
    # error.
    if encoder is None or decoder is None:
        assert hasattr(reg_cls, super)
        if reg_cls.super().__name__ in ENCODER_REGISTRY:
            ENCODER_REGISTRY[class_name] = \
                ENCODER_REGISTRY[reg_cls.super().__name__]
            DECODER_REGISTRY[class_name] = \
                DECODER_REGISTRY[reg_cls.super().__name__]
        else:
            raise RegistrationError(f"Class {class_name} has no "
                "registered superclass. Please provide encoder and "
                "decoder functions.")
    else:
        ENCODER_REGISTRY[class_name] = encoder
        DECODER_REGISTRY[class_name] = decoder

def to_json(obj):
    """
    Returns a JSON representation of the object obj if the object is
    either natively JSON serializable or the object's class is
    registered. Raises an InvalidJSONError if the class is not
    registered and is not natively JSON serializable. Precedence is
    given to the registered class encoder over any native
    serialization method if such a conflict occurs.

    Args:
        obj (Object): Object to be converted into JSON-serializable
            form.

    Returns:
        JSON str

    Raises:
        InvalidJSONError: Exception raised when object whose class
            has not yet been registered is attempted to be
            serialized.
    """
    class_name = obj.__class__.__name__

    # If the class is not registered, try directly encoding to JSON.
    if class_name not in ENCODER_REGISTRY:
        try:
            return json.dumps(obj)
        except TypeError:
            raise InvalidJSONError(f"Attempting to encode object "
                f"{obj} without registered encoder for class "
                f"{class_name}")

    # Use registered encoder function to encode to JSON.
    encoder = ENCODER_REGISTRY[class_name]
    # Include class in final json string.
    return json.dumps({
        'class': class_name,
        'args': encoder(copy.deepcopy(obj))
    })

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

# Encoders
def simulator_to_json(obj):
    json_dict = {}
    assert isinstance(obj, acnsim.Simulator)

    # Serialize non-nested attributes.
    nn_attr_lst = [
        'period', 'max_recompute', 'verbose', 'peak',
        '_iteration', '_resolve', '_last_schedule_update'
    ]
    for attr in nn_attr_lst:
        json_dict[attr] = getattr(obj, attr)

    json_dict['network'] = obj.network.to_json()
    json_dict['scheduler'] = obj.scheduler.to_json()
    json_dict['event_queue'] = obj.event_queue.to_json()

    # TODO: No clean way to serialize datetime object yet
    json_dict['start'] = obj.start.isoformat()

    # TODO: Serialize signals
    json_dict['signals'] = {}

    json_dict['pilot_signals'] = obj.pilot_signals.tolist()
    json_dict['charging_rates'] = obj.charging_rates.tolist()

    json_dict['ev_history'] = {session_id : ev.to_json() 
        for session_id, ev in obj.ev_history.items()}
    json_dict['event_history'] = [event.to_json() 
        for event in obj.event_history]

    return json.dumps(json_dict)

def network_to_json(obj):
    json_dict = {}
    assert isinstance(obj, acnsim.ChargingNetwork)

    json_dict['_EVSEs'] = {station_id : evse.to_json() 
        for station_id, evse in obj._EVSEs.items()}

    json_dict['constraint_matrix'] = obj.constraint_matrix.tolist()
    json_dict['magnitudes'] = obj.magnitudes.tolist()
    json_dict['_voltages'] = obj._voltages.tolist()
    json_dict['_phase_angles'] = obj._phase_angles.tolist()

    json_dict['constraint_index'] = obj.constraint_index

    return json.dumps(json_dict)

# Event Encoders
def event_queue_to_json(obj):
    json_dict = {}
    assert isinstance(obj, acnsim.EventQueue)

    json_dict['_queue'] = [(ts, event.to_json()) 
        for (ts, event) in obj._queue]
    json_dict['_timestep'] = obj._timestep
    
    return json.dumps(json_dict)

def event_base_attr(obj, json_dict):
    assert isinstance(obj, acnsim.Event)

    # TODO: To avoid repeated code, can dump using _event_to_json and
    # load here to get dict of Event attributes, but this will be
    # slower than below.
    json_dict['timestamp'] = obj.timestamp
    json_dict['type'] = obj.type
    json_dict['precedence'] = obj.precedence

def event_to_json(obj):
    json_dict = {}
    assert isinstance(obj, acnsim.Event)

    _event_base_attr(obj, json_dict)

    return json.dumps(json_dict)

def plugin_event_to_json(obj):
    json_dict = {}
    assert isinstance(obj, acnsim.PluginEvent)

    _event_base_attr(obj, json_dict)

    # Plugin-specific attributes
    json_dict['ev'] = obj.ev.to_json()

    return json.dumps(json_dict)

def unplug_event_to_json(obj):
    json_dict = {}
    assert isinstance(obj, acnsim.UnplugEvent)

    _event_base_attr(obj, json_dict)

    # Unplug-specific attributes
    json_dict['station_id'] = obj.station_id
    json_dict['session_id'] = obj.session_id

    return json.dumps(json_dict)

def recompute_event_to_json(obj):
    json_dict = {}
    assert isinstance(obj, acnsim.RecomputeEvent)

    _event_base_attr(obj, json_dict)

    return json.dumps(json_dict)

# Model Encoders
# EVSE Encoders
def evse_base_attr(obj, json_dict):
    assert isinstance(obj, acnsim.EVSE)

    # Serialize non-nested attributes.
    nn_attr_lst = [
        '_station_id', '_max_rate', '_min_rate', '_current_pilot',
        'is_continuous'
    ]
    for attr in nn_attr_lst:
        json_dict[attr] = getattr(obj, attr)

    if obj._ev is not None:
        json_dict['_ev'] = obj.ev.to_json()
    else:
        json_dict['_ev'] = None

def evse_to_json(obj):
    json_dict = {}
    assert isinstance(obj, acnsim.EVSE)

    _evse_base_attr(obj, json_dict)

    return json.dumps(json_dict)

def deadband_evse_to_json(obj):
    json_dict = {}
    assert isinstance(obj, acnsim.DeadbandEVSE)

    _evse_base_attr(obj, json_dict)

    json_dict['_deadband_end'] = obj._deadband_end

    return json.dumps(json_dict)

def finite_rates_evse_to_json(obj):
    json_dict = {}
    assert isinstance(obj, acnsim.FiniteRatesEVSE)

    _evse_base_attr(obj, json_dict)

    json_dict['allowable_rates'] = obj.allowable_rates

    return json.dumps(json_dict)

# EV Encoders
def ev_to_json(obj):
    json_dict = {}
    assert isinstance(obj, acnsim.EV)
    
    nn_attr_lst = [
        '_arrival', '_departure', '_session_id', '_station_id',
        '_requested_energy', '_estimated_departure',
        '_energy_delivered', '_current_charging_rate'
    ]
    for attr in nn_attr_lst:
        json_dict[attr] = getattr(obj, attr)

    json_dict['_battery'] = obj._battery.to_json()

    return json.dumps(json_dict)

# Battery Encoders
def battery_base_attr(obj, json_dict):
    assert isinstance(obj, acnsim.Battery)

    nn_attr_lst = [
        '_max_power', '_current_charging_power', '_current_charge',
        '_capacity'
    ]
    for attr in nn_attr_lst:
        json_dict[attr] = getattr(obj, attr)

def battery_to_json(obj):
    json_dict = {}
    assert isinstance(obj, acnsim.Battery)

    _battery_base_attr(obj, json_dict)

    return json.dumps(json_dict)

def linear_2_stage_battery_to_json(obj):
    json_dict = {}
    assert isinstance(obj, acnsim.Linear2StageBattery)

    _battery_base_attr(obj, json_dict)

    json_dict['_noise_level'] = obj._noise_level
    json_dict['_transition_soc'] = obj._transition_soc

    return json.dumps(json_dict)
# ====================================================================
# Decoders
# TODO: Split into encoder, decoder files?
def simulator_from_json(instr):
    in_dict = json.loads(instr)

    network = from_json(in_dict['network'])
    assert isinstance(network, acnsim.ChargingNetwork)

    events = from_json(in_dict['event_queue'])
    assert isinstance(events, acnsim.EventQueue)

    # TODO: Add option to actually initialize scheduler.
    scheduler = in_dict['scheduler']

    # TODO: Deserialize datetime object
    out_obj = acnsim.Simulator(
        network,
        scheduler,
        events,
        parser.parse(in_dict['start']),
        period=in_dict['period'],
        signals=in_dict['signals'],
        verbose=in_dict['verbose']
    )

    out_obj._iteration = in_dict['_iteration']
    out_obj.peak = in_dict['peak']

    out_obj.pilot_signals = np.array(in_dict['pilot_signals'])
    out_obj.charging_rates = np.array(in_dict['charging_rates'])

    out_obj.ev_history = {session_id : from_json(ev) 
        for session_id, ev in in_dict['ev_history'].items()}
    out_obj.event_history = [from_json(event) 
        for event in in_dict['event_history']]

    return out_obj

def network_from_json(instr):
    in_dict = json.loads(instr)

    out_obj = acnsim.network.ChargingNetwork()

    out_obj._EVSEs = {station_id : from_json(evse)
        for station_id, evse in in_dict['_EVSEs']}

    out_obj.constraint_matrix = \
        np.array(in_dict['constraint_matrix'])
    out_obj.magnitudes = \
        np.array(in_dict['magnitudes'])

    out_obj.constraint_index = in_dict['constraint_index']
    out_obj._voltages = np.array(in_dict['_voltages'])
    out_obj._phase_angles = np.array(in_dict['_phase_angles'])

    return out_obj

# Event decoders
def event_queue_from_json(instr):
    in_dict = json.loads(instr)

    out_obj = acnsim.events.EventQueue()
    out_obj._queue = [(ts, from_json(event))
        for (ts, event) in in_dict['_queue']]
    out_obj._timestep = in_dict['_timestep']

    return out_obj

def event_from_json(instr):
    in_dict = json.loads(instr)

    out_obj = acnsim.events.Event(in_dict['timestamp'])
    out_obj.type = in_dict['type']
    out_obj.precedence = in_dict['precedence']

    return out_obj

def plugin_event_from_json(instr):
    in_dict = json.loads(instr)

    out_obj = acnsim.events.PluginEvent(
        in_dict['timestamp'], from_json(in_dict['ev']))

    out_obj.type = in_dict['type']
    out_obj.precedence = in_dict['precedence']

    return out_obj

def unplug_event_from_json(instr):
    in_dict = json.loads(instr)

    out_obj = acnsim.events.UnplugEvent(
        in_dict['timestamp'], in_dict['station_id'], 
        in_dict['session_id'])

    out_obj.type = in_dict['type']
    out_obj.precedence = in_dict['precedence']

    return out_obj

def recompute_event_from_json(instr):
    in_dict = json.loads(instr)

    out_obj = acnsim.events.RecomputeEvent(in_dict['timestamp'])

    out_obj.type = in_dict['type']
    out_obj.precedence = in_dict['precedence']

    return out_obj

# EVSE decoders
def evse_from_json(instr):
    in_dict = json.loads(instr)

    if in_dict['_ev'] is not None:
        ev = from_json(in_dict['_ev'])
    else:
        ev = None

    out_obj = acnsim.models.EVSE(
        in_dict['_station_id'],
        max_rate=in_dict['_max_rate'],
        min_rate=in_dict['_min_rate']
    )

    out_obj._current_pilot = in_dict['_current_pilot']
    out_obj._ev = ev
    out_obj.is_continuous = True

    return out_obj

def deadband_evse_from_json(instr):
    in_dict = json.loads(instr)

    if in_dict['_ev'] is not None:
        ev = from_json(in_dict['_ev'])
    else:
        ev = None

    out_obj = acnsim.models.DeadbandEVSE(
        in_dict['_station_id'],
        deadband_end=in_dict['_deadband_end'],
        max_rate=in_dict['_max_rate'],
        min_rate=in_dict['_min_rate']
    )

    out_obj._current_pilot = in_dict['_current_pilot']
    out_obj._ev = ev
    out_obj.is_continuous = True

    return out_obj

def finite_rates_evse_from_json(instr):
    in_dict = json.loads(instr)

    if in_dict['_ev'] is not None:
        ev = from_json(in_dict['_ev'])
    else:
        ev = None

    out_obj = acnsim.models.FiniteRatesEVSE(
        in_dict['_station_id'],
        in_dict['allowable_rates']
    )

    out_obj.min_rate = in_dict['_min_rate']
    out_obj._current_pilot = in_dict['_current_pilot']
    out_obj._ev = ev
    out_obj.is_continuous = False

    return out_obj

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

    return out_obj

# Battery decoders
def battery_from_json(instr):
    in_dict = json.loads(instr)

    # Note second arg in below construction is a placeholder
    out_obj = acnsim.models.Battery(
        in_dict['_capacity'], 0, in_dict['_max_power'])

    out_obj._current_charging_power = \
        in_dict['_current_charging_power']
    out_obj._current_charge = in_dict['_current_charge']

    return out_obj

def linear_2_stage_battery_from_json(instr):
    in_dict = json.loads(instr)

    # Note second arg in below construction is a placeholder
    out_obj = acnsim.models.Linear2StageBattery(
        in_dict['_capacity'], 0, in_dict['_max_power'],
        noise_level=in_dict['_noise_level'],
        transition_soc=in_dict['_transition_soc']
    )

    out_obj._current_charging_power = \
        in_dict['_current_charging_power']
    out_obj._current_charge = in_dict['_current_charge']

    return out_obj

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

# TODO: Where should these registries go?
ENCODER_REGISTRY = {
    'Simulator' : simulator_to_json,
    'Network' : network_to_json,
    'EventQueue' : event_queue_to_json,
    'Event' : event_to_json,
    'PluginEvent' : plugin_event_to_json,
    'UnplugEvent' : unplug_event_to_json,
    'RecomputeEvent' : recompute_event_to_json,
    'EVSE' : evse_to_json,
    'DeadbandEVSE' : deadband_evse_to_json,
    'FiniteRatesEVSE' : finite_rates_evse_to_json,
    'EV' : ev_to_json,
    'Battery' : battery_to_json,
    'Linear2StageBattery' : linear_2_stage_battery_to_json
}

DECODER_REGISTRY = {
    'Simulator' : simulator_from_json,
    'Network' : network_from_json,
    'EventQueue' : event_queue_from_json,
    'Event' : event_from_json,
    'PluginEvent' : plugin_event_from_json,
    'UnplugEvent' : unplug_event_from_json,
    'RecomputeEvent' : recompute_event_from_json,
    'EVSE' : evse_from_json,
    'DeadbandEVSE' : deadband_evse_from_json,
    'FiniteRatesEVSE' : finite_rates_evse_from_json,
    'EV' : ev_from_json,
    'Battery' : battery_from_json,
    'Linear2StageBattery' : linear_2_stage_battery_from_json
}