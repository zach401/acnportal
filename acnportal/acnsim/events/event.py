# coding=utf-8
"""
Defines several classes of Events in the simulation.
"""
from typing import Optional, Dict, Any, Tuple

from ..base import BaseSimObj
import warnings

from ..models.ev import EV


class Event(BaseSimObj):
    """ Base class for all events.

    Args:
        timestamp (int): Timestamp when an event occurs (periods)

    Attributes:
        timestamp (int): See args.
        event_type (str): Name of the event type.
        precedence (float): Used to order occurrence for events that happen in the same
            timestep. Higher precedence events occur before lower precedence events.

    """

    timestamp: int
    event_type: str
    precedence: float

    def __init__(self, timestamp: int) -> None:
        self.timestamp = timestamp
        self.event_type = ""
        self.precedence = float("inf")

    def __lt__(self, other: "Event") -> bool:
        """ Return True if the precedence of self is less than that of other.

        Args:
            other (Event like): Another Event-like object.

        Returns:
            bool
        """
        return self.precedence < other.precedence

    @property
    def type(self) -> str:
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

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ Implements BaseSimObj._to_dict. """
        attribute_dict = {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "precedence": self.precedence,
        }
        return attribute_dict, context_dict

    @classmethod
    def _from_dict_helper(
        cls, out_obj: "Event", attribute_dict: Dict[str, Any]
    ) -> None:
        out_obj.event_type = attribute_dict["event_type"]
        out_obj.precedence = attribute_dict["precedence"]

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, BaseSimObj]] = None,
    ) -> Tuple[BaseSimObj, Dict[str, BaseSimObj]]:
        """ Implements BaseSimObj._from_dict. """
        out_obj = cls(attribute_dict["timestamp"])
        cls._from_dict_helper(out_obj, attribute_dict)
        return out_obj, loaded_dict


class EVEvent(Event):
    """ Subclass of Event for events which deal with an EV such as Plugin and Unplug
    events.

    Args:
        timestamp (int): See Event.
        ev (EV): The EV associated with this event.
    """

    ev: EV

    def __init__(self, timestamp: int, ev: EV) -> None:
        super().__init__(timestamp)
        self.ev = ev

    @property
    def station_id(self) -> str:
        """ Return the station_id for the EV associated with this Event. """
        return self.ev.station_id

    @property
    def session_id(self) -> str:
        """ Return the session_id for the EV associated with this Event. """
        return self.ev.session_id

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ Implements BaseSimObj._to_dict. """
        attribute_dict, context_dict = super()._to_dict(context_dict)
        # Plugin-specific attributes

        # noinspection PyProtectedMember
        registry, context_dict = self.ev._to_registry(context_dict=context_dict)
        attribute_dict["ev"] = registry["id"]

        return attribute_dict, context_dict

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, BaseSimObj]] = None,
    ) -> Tuple[BaseSimObj, Dict[str, BaseSimObj]]:
        """ Implements BaseSimObj._from_dict. """
        # noinspection PyProtectedMember
        ev, loaded_dict = BaseSimObj._build_from_id(
            attribute_dict["ev"], context_dict, loaded_dict=loaded_dict
        )
        out_obj = cls(attribute_dict["timestamp"], ev)
        cls._from_dict_helper(out_obj, attribute_dict)
        return out_obj, loaded_dict


class PluginEvent(EVEvent):
    """ Subclass of Event for EV plugins.

    Args:
        timestamp (int): See Event.
        ev (EV): The EV which will be plugged in.
    """

    def __init__(self, timestamp: int, ev: EV) -> None:
        super().__init__(timestamp, ev)
        self.event_type = "Plugin"
        self.precedence = 10


class UnplugEvent(EVEvent):
    """ Subclass of Event for EV unplugs.

    Args:
        timestamp (int): See Event.
        ev (EV): The EV which will be unplugged.
    """

    def __init__(self, timestamp: int, ev: EV) -> None:
        super().__init__(timestamp, ev)
        self.event_type = "Unplug"
        self.precedence = 0

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, BaseSimObj]] = None,
    ) -> Tuple[BaseSimObj, Dict[str, BaseSimObj]]:
        """ Implements BaseSimObj._from_dict. """
        # noinspection PyProtectedMember
        try:
            ev, loaded_dict = BaseSimObj._build_from_id(
                attribute_dict["ev"], context_dict, loaded_dict=loaded_dict
            )
        except KeyError:
            # In acnportal v0.2.2, UnplugEvent had session_id and station_id attributes
            # instead of an ev attribute. For backwards compatibility with previously
            # serialized UnplugEvents, we build the EV partially (including session and
            # station ids only) and set it as the ev attribute of the current
            # implementation of UnplugEvent.
            warnings.warn(
                "Loading UnplugEvents from an older version of acnportal into a newer "
                "one. UnplugEvent EV object will be incompletely deserialized."
            )
            ev = EV(
                -1,
                -1,
                -1,
                attribute_dict["station_id"],
                attribute_dict["session_id"],
                None,
            )
            for attribute in [
                "arrival",
                "departure",
                "requested_energy",
                "estimated_departure",
                "battery",
                "energy_delivered",
                "current_charging_rate",
            ]:
                delattr(ev, f"_{attribute}")
        out_obj = cls(attribute_dict["timestamp"], ev)
        cls._from_dict_helper(out_obj, attribute_dict)
        return out_obj, loaded_dict


class RecomputeEvent(Event):
    """ Subclass of Event for when the algorithm should be recomputed."""

    def __init__(self, timestamp: int) -> None:
        super().__init__(timestamp)
        self.event_type = "Recompute"
        self.precedence = 20
