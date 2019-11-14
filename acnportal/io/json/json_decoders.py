import json

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