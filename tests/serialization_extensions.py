""" This file contains extensions of ACN-Sim for testing purposes. """
from typing import Optional, Dict, Any, Tuple

from acnportal import acnsim
from acnportal.acnsim.base import BaseSimObj


class NamedEvent(acnsim.Event):
    """ An extension of Event that has a name. """

    def __init__(self, timestamp, name):
        super().__init__(timestamp)
        self.name = name


class DefaultNamedEvent(acnsim.Event):
    """ An extension of Event that has a name with a default. """

    def __init__(self, timestamp, name="my_event"):
        super().__init__(timestamp)
        self.name = name


class SetAttrEvent(acnsim.Event):
    """ An extension of Event with a settable attribute. """

    def __init__(self, timestamp):
        super().__init__(timestamp)

    def set_extra_attr(self, attr_val):
        # noinspection PyAttributeOutsideInit
        self.extra_attr = attr_val


class BatteryListEvent(acnsim.Event):
    """ An extension of Event with a list of Batteries. """

    def __init__(self, timestamp, battery_list):
        super().__init__(timestamp)
        self.battery_list = battery_list

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        attribute_dict, context_dict = super()._to_dict(context_dict)
        battery_list = []
        for ev in self.battery_list:
            # noinspection PyProtectedMember
            registry, context_dict = ev._to_registry(context_dict=context_dict)
            battery_list.append(registry["id"])
        attribute_dict["battery_list"] = battery_list
        return attribute_dict, context_dict

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, BaseSimObj]] = None,
    ) -> Tuple[BaseSimObj, Dict[str, BaseSimObj]]:
        battery_list = []
        for ev in attribute_dict["battery_list"]:
            # noinspection PyProtectedMember
            ev_elt, loaded_dict = BaseSimObj._build_from_id(
                ev, context_dict=context_dict, loaded_dict=loaded_dict
            )
            battery_list.append(ev_elt)
        out_obj = cls(attribute_dict["timestamp"], battery_list)
        return out_obj, loaded_dict
