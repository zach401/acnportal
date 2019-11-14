import json

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