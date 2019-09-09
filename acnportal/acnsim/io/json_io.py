import json
from acnportal import acnsim

def from_json(in_file, typ='simulator'):
	""" Load object of type given by typ from a given in_file using JSON

    Args:
        in_file (file-like): File from which to read JSON.
        typ (str): type of object to be read, default simulator. Options:
            {'simulator', 'network', 'scheduler', 'event_queue', 'ev', 'event', 'evse'}

    Returns: Type returned depends on value of typ. Possibile return types:
        {Simulator, ChargingNetwork, BasicAlgorithm-like, EventQueue, EV, Event, EVSE, Battery}
    """
    # Check that the object type being decoded is valid
    decoder_func = '_{0}_from_json_dict'.format(typ)
    if decoder_func not in locals():
		raise InvalidJSONError('Attempting to decode unrecognized object: {0}'.format(typ))
    
    in_dict = json.loads(in_file)
    
    # Check that json file is of the correct object type
    try:
        real_typ = in_dict['obj_type']
    except KeyError:
        raise InvalidJSONError('Unable to read object of type {0} from file'.format(typ))
    if typ != real_typ:
        raise InvalidJSONError('Expected object of type {0}, got {1}'.format(typ, real_typ))

    return locals()[decoder_func]()(in_dict)

def to_json(obj_dict):
	""" Returns a JSON-serializable representation of the object obj if possible
	
	Args:
		Dict[str, Object]: an object dictionary representing an instance of an object
			Possible objects:
			Simulator, ChargingNetwork, BasicAlgorithm, EventQueue, Event, EV, or EVSE like

	Returns: JSON serializable
	"""
	encoder_func = '_{0}_dict_to_json'.format(obj_dict['obj_type'])
	if encoder_func not in locals():
		raise InvalidJSONError('Attempting to encode unrecognized object: {0}'.format(typ))
	return locals()[encoder_func](obj_dict)

###-------------Encoders-------------###

def _simulator_dict_to_json(out_dict):
	# Doesn't JSONify schedule history
    del out_dict['schedule_history']

    out_dict['network'] = out_dict['network'].to_json()
    out_dict['scheduler'] = out_dict['scheduler'].to_json()
    out_dict['event_queue'] = out_dict['event_queue'].to_json()

    out_dict['pilot_signals'] = out_dict['pilot_signals'].to_list()
    out_dict['charging_rates'] = out_dict['charging_rates'].to_list()

    out_dict['ev_history'] = {session_id : ev.to_json() for session_id, ev in out_dict['ev_history'].items()}
    out_dict['event_history'] = [event.to_json() for event in out_dict['event_history']]
    
    return json.dumps(out_dict)

def _network_dict_to_json(out_dict):
    out_dict['constraint_matrix'] = out_dict['constraint_matrix'].to_list()
    out_dict['magnitudes'] = out_dict['magnitudes'].to_list()
    out_dict['_voltages'] = out_dict['_voltages'].to_list()
    out_dict['_phase_angles'] = out_dict['_voltages'].to_list()

    return json.dumps(out_dict)

def _scheduler_dict_to_json(out_dict):
	# TODO
	pass

def _event_queue_dict_to_json(out_dict):
	out_dict['_queue'] = [event.to_json() for event in out_dict['_queue']]
	
	return json.dumps(out_dict)

def _event_dict_to_json(out_dict):=
	return json.dumps(out_dict)

def _ev_dict_to_json(out_dict):
	out_dict['_battery'] = out_dict['_battery'].to_json()

	return json.dumps(out_dict)

def _evse_dict_to_json(out_dict):
	out_dict[]
	pass

def _battery_dict_to_json(out_dict):
	return json.dumps(out_dict)

###-------------Decoders-------------###

def _simulator_from_json_dict(in_dict):
	# TODO
    pass

def _network_from_json_dict(in_dict):
    # TODO
    pass

def _scheduler_from_json_dict(in_dict):
    # TODO
    pass

def _event_queue_from_json_dict(in_dict):
    # TODO
    pass

def _event_from_json_dict(in_dict):
    # TODO
    pass

def _ev_from_json_dict(in_dict):
    # TODO
    pass

def _evse_from_json_dict(in_dict):
    # TODO
    pass

class InvalidJSONError(Exception):
    """ Exception which is raised when trying to read a JSON file with contents that do not match
    the given type. """
    pass