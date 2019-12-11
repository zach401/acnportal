from acnportal import acnsim_io
from acnportal.acnsim_io import json_writer, json_reader

class Event:
    """ Base class for all events.

    Args:
        timestamp (int): Timestamp when an event occurs (periods)

    Attributes:
        timestamp (int): See args.
        type (str): Name of the event type.
        precedence (float): Used to order occurrence for events that happen in the same timestep. Higher precedence
            events occur before lower precedence events.

    """
    def __init__(self, timestamp):
        self.timestamp = timestamp
        # TODO: type is a builtin, use different name.
        # Keep a type accessor w deprecation warning.
        self.type = ''
        self.precedence = float('inf')

    def __lt__(self, other):
        """ Return True if the precedence of self is less than that of other.

        Args:
            other (Event like): Another Event-like object.

        Returns:
            bool
        """
        return self.precedence < other.precedence

    @json_writer
    def to_json(self, context_dict={}):
        """ Converts the event into a JSON serializable dict

        Returns:
            JSON serializable
        """
        args_dict = {}

        args_dict['timestamp'] = self.timestamp
        args_dict['type'] = self.type
        args_dict['precedence'] = self.precedence

        return args_dict

    @classmethod
    @json_reader
    def from_json(cls, in_dict, context_dict={}, loaded_dict={}, cls_kwargs={}):
        out_obj = cls(in_dict['timestamp'], **cls_kwargs)
        out_obj.type = in_dict['type']
        out_obj.precedence = in_dict['precedence']
        return out_obj


class PluginEvent(Event):
    """ Subclass of Event for EV plugins.

    Args:
        timestamp (int): See Event.
        ev (EV): The EV which will be plugged in.
    """
    def __init__(self, timestamp, ev):
        super().__init__(timestamp)
        self.type = 'Plugin'
        self.ev = ev
        self.precedence = 10

    @json_writer
    def to_json(self, context_dict={}):
        """ Converts the event into a JSON serializable dict

        Returns:
            JSON serializable
        """
        args_dict = super().to_json.__wrapped__(self, context_dict)
        # Plugin-specific attributes
        args_dict['ev'] = self.ev.to_json(context_dict=context_dict)['id']

        return args_dict

    @classmethod
    @json_reader
    def from_json(cls, in_dict, context_dict={}, loaded_dict={}, cls_kwargs={}):
        # TODO: standardize acnsim_io.read_from_id inputs (use = or not)
        ev = acnsim_io.read_from_id(in_dict['ev'], context_dict, loaded_dict)
        cls_kwargs = {'ev': ev}
        out_obj = super().from_json.__wrapped__(cls, in_dict, context_dict, loaded_dict, cls_kwargs)
        return out_obj

class UnplugEvent(Event):
    """ Subclass of Event for EV unplugs.

    Args:
        timestamp (int): See Event.
        station_id (str): ID of the EVSE where the EV is to be unplugged.
        session_id (str): ID of the session which should be ended.
    """
    def __init__(self, timestamp, station_id, session_id):
        super().__init__(timestamp)
        self.type = 'Unplug'
        self.station_id = station_id
        self.session_id = session_id
        self.precedence = 0

    @json_writer
    def to_json(self, context_dict={}):
        """ Converts the event into a JSON serializable dict

        Returns:
            JSON serializable
        """
        args_dict = super().to_json.__wrapped__(self, context_dict)

        # Unplug-specific attributes
        args_dict['station_id'] = self.station_id
        args_dict['session_id'] = self.session_id

        return args_dict

    @classmethod
    @json_reader
    def from_json(cls, in_dict, context_dict={}, loaded_dict={}, cls_kwargs={}):
        cls_kwargs = {'station_id': in_dict['station_id'], 'session_id': in_dict['session_id']}
        out_obj = super().from_json.__wrapped__(cls, in_dict, context_dict, loaded_dict, cls_kwargs)
        return out_obj

class RecomputeEvent(Event):
    """ Subclass of Event for when the algorithm should be recomputed."""
    def __init__(self, timestamp):
        super().__init__(timestamp)
        self.type = 'Recompute'
        self.precedence = 20
