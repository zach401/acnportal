from ..base import *
import warnings

class Event(BaseSimObj):
    """ Base class for all events.

    Args:
        timestamp (int): Timestamp when an event occurs (periods)

    Attributes:
        timestamp (int): See args.
        event_type (str): Name of the event type.
        precedence (float): Used to order occurrence for events that happen in the same timestep. Higher precedence
            events occur before lower precedence events.

    """
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.event_type = ''
        self.precedence = float('inf')

    def __lt__(self, other):
        """ Return True if the precedence of self is less than that of other.

        Args:
            other (Event like): Another Event-like object.

        Returns:
            bool
        """
        return self.precedence < other.precedence

    @property
    def type(self):
        """
        Legacy accessor for event_type. This will be removed in a future
        release.
        """
        warnings.warn("Accessor 'type' for type of Event is deprecated. "
                      "Use 'event_type' instead.",
                      DeprecationWarning)
        return self.event_type

    def to_dict(self, context_dict=None):
        """ Implements BaseSimObj.to_dict. """
        context_dict, = none_to_empty_dict(context_dict)
        args_dict = {}

        args_dict['timestamp'] = self.timestamp
        args_dict['event_type'] = self.event_type
        args_dict['precedence'] = self.precedence

        return args_dict

    @classmethod
    def from_dict(cls, in_dict, context_dict=None, loaded_dict=None, cls_kwargs=None):
        """ Implements BaseSimObj.from_dict. """
        context_dict, loaded_dict, cls_kwargs = \
            none_to_empty_dict(context_dict, loaded_dict, cls_kwargs)
        out_obj = cls(in_dict['timestamp'], **cls_kwargs)
        out_obj.event_type = in_dict['event_type']
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
        self.event_type = 'Plugin'
        self.ev = ev
        self.precedence = 10


    def to_dict(self, context_dict=None):
        """ Implements BaseSimObj.to_dict. """
        context_dict, = none_to_empty_dict(context_dict)
        args_dict = super().to_dict(context_dict)
        # Plugin-specific attributes
        args_dict['ev'] = self.ev.to_registry(context_dict=context_dict)['id']

        return args_dict

    @classmethod
    def from_dict(cls, in_dict, context_dict=None, loaded_dict=None, cls_kwargs=None):
        """ Implements BaseSimObj.from_dict. """
        context_dict, loaded_dict, cls_kwargs = \
            none_to_empty_dict(context_dict, loaded_dict, cls_kwargs)
        # TODO: standardize read_from_id inputs (use = or not)
        ev = read_from_id(in_dict['ev'], context_dict, loaded_dict)
        cls_kwargs = {'ev': ev}
        out_obj = super().from_dict(in_dict, context_dict, loaded_dict, cls_kwargs)
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
        self.event_type = 'Unplug'
        self.station_id = station_id
        self.session_id = session_id
        self.precedence = 0


    def to_dict(self, context_dict=None):
        """ Implements BaseSimObj.to_dict. """
        context_dict, = none_to_empty_dict(context_dict)
        args_dict = super().to_dict(context_dict)

        # Unplug-specific attributes
        args_dict['station_id'] = self.station_id
        args_dict['session_id'] = self.session_id

        return args_dict

    @classmethod
    def from_dict(cls, in_dict, context_dict=None, loaded_dict=None, cls_kwargs=None):
        """ Implements BaseSimObj.from_dict. """
        context_dict, loaded_dict, cls_kwargs = \
            none_to_empty_dict(context_dict, loaded_dict, cls_kwargs)
        cls_kwargs = {'station_id': in_dict['station_id'], 'session_id': in_dict['session_id']}
        out_obj = super().from_dict(in_dict, context_dict, loaded_dict, cls_kwargs)
        return out_obj

class RecomputeEvent(Event):
    """ Subclass of Event for when the algorithm should be recomputed."""
    def __init__(self, timestamp):
        super().__init__(timestamp)
        self.event_type = 'Recompute'
        self.precedence = 20
