import json
from .json_encoders import *
from .json_decoders import *
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
