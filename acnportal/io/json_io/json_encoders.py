import json
from acnportal import acnsim
from copy import deepcopy

class JSONWriter:
    """
    Class that handles writing an input ACN-Sim object to a JSON
    serializable object. The JSON representation of the object is such
    that using JSONReader to load this representation will result in
    an identical object graph.

    Args:
        obj_reg (Dict[int, JSON-able]): registry of all objects
            serialized by this Writer, stored in object id, JSON
            string pairs.
    """

    def __init__(self, obj_reg):
        self.obj_reg = obj_reg

    def write(self, obj):
        """
        Writes obj to a JSON string.

        Args:
            obj (ACN-Sim object): Object to serialize.

        Returns:
            JSON str: JSON string representing serialized object.
        """

    def _write(self, obj):
        """
        This method adds a serializable version of obj to the Writer's
        object registry.

        Tries the following cases before throwing an error:
        - Python's native serialization (json.dumps)
        - Object's to_json method (all ACN-Sim objects have this)
        - Recursive serialization of the object's dict using _write on
            each attribute
        - If object is a list, builds a new list in which each
            element is JSON serializable and uses json.dumps
        - If object is a dict, builds a new dict in which each
            element is JSON serializable and uses json.dumps
        - If object is a numpy array, calls the to_list method and
            uses json.dumps

        In all cases, a serializable version of obj is added to the
        object registry, not a JSON string. Also, the class name is
        included with the object's representation, so each value
        in the object registry is a tuple.

        Args:
            obj (ACN-Sim object): Object to serialize.

        Returns:
            int: int id for the object in the Writer's object 
                registry.

        """
        # Check if this object has already been serialized.
        if id(obj) in self.obj_reg:
            return id(obj)

        # Try native JSON serialization.
        try:
            out_json = json.dumps(obj)
        except TypeError:
            pass
        else:
            self.obj_reg[id(obj)] = (type(obj), obj)
            return id(obj)

        # Try object's to_json method.
        try:
            out_json = obj.to_json()
        except AttributeError:
            pass
        else:
            self.obj_reg[id(obj)] = (type(obj), out_json)
            return id(obj)

        # Try recursive serialization of object's attributes.
        try:
            obj_dict = obj.__dict__
        except AttributeError:
            pass
        else:
            new_dict = {}
            for name, value in obj_dict.items():
                new_dict[name] = self._write(value)
            self.obj_reg[id(obj)] = (type(obj), new_dict)

        # Try unwrapping a list of objects
        if isinstance(obj, list):
            new_lst = [self._write(elt) for elt in obj]
            self.obj_reg[id(obj)] = (type(obj), new_lst)
            return id(obj)
        elif isinstance(obj, dict):
            new_dict = {self._write(key): self._write(value)
                for key, value in obj.items()}
            self.obj_reg[id(obj)] = (type(obj), new_dict)
            return id(obj)
        elif isinstance(obj, np.ndarray):
            new_lst = obj.to_list()
            self.obj_reg[id(obj)] = (type(obj), new_lst)
            return id(obj)
        else:
            # TODO: raise error
            pass

ENCODER_REGISTRY = {}

# TODO: registering here makes order of definition matter. Is
# there a better way to do this?

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
        'args': encoder(deepcopy(obj))
    })

# Encoders
## Model Encoders
### Battery Encoders
def _battery_base_attr(obj, args_dict):

    nn_attr_lst = [
        '_max_power', '_current_charging_power', '_current_charge',
        '_capacity'
    ]
    for attr in nn_attr_lst:
        args_dict[attr] = getattr(obj, attr)

def battery_to_json(obj):
    assert obj.__class__.__name__ == 'Battery'
    args_dict = {}

    _battery_base_attr(obj, args_dict)

    return json.dumps(args_dict)
ENCODER_REGISTRY['Battery'] = battery_to_json

def linear_2_stage_battery_to_json(obj):
    assert obj.__class__.__name__ == 'Linear2StageBattery'
    args_dict = {}

    _battery_base_attr(obj, args_dict)

    args_dict['_noise_level'] = obj._noise_level
    args_dict['_transition_soc'] = obj._transition_soc

    return json.dumps(args_dict)
ENCODER_REGISTRY['Linear2StageBattery'] = \
    linear_2_stage_battery_to_json

# EV Encoders
def ev_to_json(obj):
    assert obj.__class__.__name__ == 'EV'
    args_dict = {}
    
    nn_attr_lst = [
        '_arrival', '_departure', '_session_id', '_station_id',
        '_requested_energy', '_estimated_departure',
        '_energy_delivered', '_current_charging_rate'
    ]
    for attr in nn_attr_lst:
        args_dict[attr] = getattr(obj, attr)

    args_dict['_battery'] = to_json(obj._battery)

    return json.dumps(args_dict)
ENCODER_REGISTRY['EV'] = ev_to_json

# EVSE Encoders
def _evse_base_attr(obj, args_dict):
    assert isinstance(obj, acnsim.EVSE)

    # Serialize non-nested attributes.
    nn_attr_lst = [
        '_station_id', '_max_rate', '_min_rate', '_current_pilot',
        'is_continuous'
    ]
    for attr in nn_attr_lst:
        args_dict[attr] = getattr(obj, attr)

    if obj._ev is not None:
        args_dict['_ev'] = to_json(obj.ev)
    else:
        args_dict['_ev'] = None

def evse_to_json(obj):
    assert obj.__class__.__name__ == 'EVSE'
    args_dict = {}

    _evse_base_attr(obj, args_dict)

    return json.dumps(args_dict)
ENCODER_REGISTRY['EVSE'] = evse_to_json

def deadband_evse_to_json(obj):
    assert obj.__class__.__name__ == 'DeadbandEVSE'
    args_dict = {}

    _evse_base_attr(obj, args_dict)

    args_dict['_deadband_end'] = obj._deadband_end

    return json.dumps(args_dict)
ENCODER_REGISTRY['DeadbandEVSE'] = deadband_evse_to_json

def finite_rates_evse_to_json(obj):
    assert obj.__class__.__name__ == 'FiniteRatesEVSE'
    args_dict = {}

    _evse_base_attr(obj, args_dict)

    args_dict['allowable_rates'] = obj.allowable_rates

    return json.dumps(args_dict)
ENCODER_REGISTRY['FiniteRatesEVSE'] = finite_rates_evse_to_json

# Event Encoders
def _event_base_attr(obj, args_dict):
    assert isinstance(obj, acnsim.Event)

    args_dict['timestamp'] = obj.timestamp
    args_dict['type'] = obj.type
    args_dict['precedence'] = obj.precedence

def event_to_json(obj):
    assert obj.__class__.__name__ == 'Event'
    args_dict = {}

    _event_base_attr(obj, args_dict)

    return json.dumps(args_dict)
ENCODER_REGISTRY['Event'] = event_to_json

def plugin_event_to_json(obj):
    assert obj.__class__.__name__ == 'PluginEvent'
    args_dict = {}

    _event_base_attr(obj, args_dict)

    # Plugin-specific attributes
    args_dict['ev'] = to_json(obj.ev)

    return json.dumps(args_dict)
ENCODER_REGISTRY['PluginEvent'] = plugin_event_to_json

def unplug_event_to_json(obj):
    assert obj.__class__.__name__ == 'UnplugEvent'
    args_dict = {}

    _event_base_attr(obj, args_dict)

    # Unplug-specific attributes
    args_dict['station_id'] = obj.station_id
    args_dict['session_id'] = obj.session_id

    return json.dumps(args_dict)
ENCODER_REGISTRY['UnplugEvent'] = unplug_event_to_json

def recompute_event_to_json(obj):
    assert obj.__class__.__name__ == 'RecomputeEvent'
    args_dict = {}

    _event_base_attr(obj, args_dict)

    return json.dumps(args_dict)
ENCODER_REGISTRY['RecomputeEvent'] = recompute_event_to_json

def event_queue_to_json(obj):
    assert obj.__class__.__name__ == 'EventQueue'
    args_dict = {}

    args_dict['_queue'] = [(ts, to_json(event)) 
        for (ts, event) in obj._queue]
    args_dict['_timestep'] = obj._timestep
    
    return json.dumps(args_dict)
ENCODER_REGISTRY['EventQueue'] = event_queue_to_json

def charging_network_to_json(obj):
    assert obj.__class__.__name__ == 'ChargingNetwork'
    args_dict = {}

    args_dict['_EVSEs'] = {station_id : to_json(evse) 
        for station_id, evse in obj._EVSEs.items()}

    args_dict['constraint_matrix'] = obj.constraint_matrix.tolist()
    args_dict['magnitudes'] = obj.magnitudes.tolist()
    args_dict['_voltages'] = obj._voltages.tolist()
    args_dict['_phase_angles'] = obj._phase_angles.tolist()

    args_dict['constraint_index'] = obj.constraint_index

    return json.dumps(args_dict)
ENCODER_REGISTRY['ChargingNetwork'] = charging_network_to_json

def simulator_to_json(obj):
    assert obj.__class__.__name__ == 'Simulator'
    args_dict = {}

    # Serialize non-nested attributes.
    nn_attr_lst = [
        'period', 'max_recompute', 'verbose', 'peak',
        '_iteration', '_resolve', '_last_schedule_update'
    ]
    for attr in nn_attr_lst:
        args_dict[attr] = getattr(obj, attr)

    args_dict['network'] = to_json(obj.network)
    args_dict['scheduler'] = repr(obj.scheduler)
    args_dict['event_queue'] = to_json(obj.event_queue)

    args_dict['start'] = obj.start.isoformat()

    # TODO: Serialize signals
    args_dict['signals'] = {}

    args_dict['pilot_signals'] = obj.pilot_signals.tolist()
    args_dict['charging_rates'] = obj.charging_rates.tolist()

    args_dict['ev_history'] = {session_id : to_json(ev) 
        for session_id, ev in obj.ev_history.items()}
    args_dict['event_history'] = [to_json(event) 
        for event in obj.event_history]

    # TODO: Schedule history
    return json.dumps(args_dict)
ENCODER_REGISTRY['Simulator'] = simulator_to_json