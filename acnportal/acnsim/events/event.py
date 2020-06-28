from ..base import BaseSimObj
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
        self.event_type = ""
        self.precedence = float("inf")

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
        warnings.warn(
            "Accessor 'type' for type of Event is deprecated. "
            "Use 'event_type' instead.",
            DeprecationWarning,
        )
        return self.event_type

    def _to_dict(self, context_dict=None):
        """ Implements BaseSimObj._to_dict. """
        attribute_dict = {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "precedence": self.precedence,
        }
        return attribute_dict, context_dict

    @classmethod
    def _from_dict_helper(cls, out_obj, attribute_dict):
        out_obj.event_type = attribute_dict["event_type"]
        out_obj.precedence = attribute_dict["precedence"]

    @classmethod
    def _from_dict(cls, attribute_dict, context_dict, loaded_dict=None):
        """ Implements BaseSimObj._from_dict. """
        out_obj = cls(attribute_dict["timestamp"])
        cls._from_dict_helper(out_obj, attribute_dict)
        return out_obj, loaded_dict


class PluginEvent(Event):
    """ Subclass of Event for EV plugins.

    Args:
        timestamp (int): See Event.
        ev (EV): The EV which will be plugged in.
    """

    def __init__(self, timestamp, ev):
        super().__init__(timestamp)
        self.event_type = "Plugin"
        self.ev = ev
        self.precedence = 10

    def _to_dict(self, context_dict=None):
        """ Implements BaseSimObj._to_dict. """
        attribute_dict, context_dict = super()._to_dict(context_dict)
        # Plugin-specific attributes

        # noinspection PyProtectedMember
        registry, context_dict = self.ev._to_registry(context_dict=context_dict)
        attribute_dict["ev"] = registry["id"]

        return attribute_dict, context_dict

    @classmethod
    def _from_dict(cls, attribute_dict, context_dict, loaded_dict=None):
        """ Implements BaseSimObj._from_dict. """
        # noinspection PyProtectedMember
        ev, loaded_dict = BaseSimObj._build_from_id(
            attribute_dict["ev"], context_dict, loaded_dict=loaded_dict
        )
        out_obj = cls(attribute_dict["timestamp"], ev)
        cls._from_dict_helper(out_obj, attribute_dict)
        return out_obj, loaded_dict


class UnplugEvent(Event):
    """ Subclass of Event for EV unplugs.

    Args:
        timestamp (int): See Event.
        station_id (str): ID of the EVSE where the EV is to be unplugged.
        session_id (str): ID of the session which should be ended.
    """

    def __init__(self, timestamp, station_id, session_id):
        super().__init__(timestamp)
        self.event_type = "Unplug"
        self.station_id = station_id
        self.session_id = session_id
        self.precedence = 0

    def _to_dict(self, context_dict=None):
        """ Implements BaseSimObj._to_dict. """
        attribute_dict, context_dict = super()._to_dict(context_dict)
        # Unplug-specific attributes
        attribute_dict["station_id"] = self.station_id
        attribute_dict["session_id"] = self.session_id
        return attribute_dict, context_dict

    @classmethod
    def _from_dict(cls, attribute_dict, context_dict, loaded_dict=None):
        """ Implements BaseSimObj._from_dict. """
        out_obj = cls(
            attribute_dict["timestamp"],
            attribute_dict["station_id"],
            attribute_dict["session_id"],
        )
        cls._from_dict_helper(out_obj, attribute_dict)
        return out_obj, loaded_dict


class RecomputeEvent(Event):
    """ Subclass of Event for when the algorithm should be recomputed."""

    def __init__(self, timestamp):
        super().__init__(timestamp)
        self.event_type = "Recompute"
        self.precedence = 20
