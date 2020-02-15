""" This file contains extensions of ACN-Sim for testing purposes. """
from acnportal import acnsim
import acnportal.acnsim.base as base


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
        self.extra_attr = attr_val


class BattListEvent(acnsim.Event):
    """ An extension of Event with a list of Batteries. """
    def __init__(self, timestamp, batt_list):
        super().__init__(timestamp)
        self.batt_list = batt_list

    def to_dict(self, context_dict=None):
        attribute_dict, context_dict = super().to_dict(context_dict)

        batt_list = []
        for ev in self.batt_list:
            registry, context_dict = ev.to_registry(context_dict=context_dict)
            batt_list.append(registry['id'])
        attribute_dict['batt_list'] = batt_list

        return attribute_dict, context_dict

    @classmethod
    def from_dict(cls, attribute_dict,
                  context_dict=None, loaded_dict=None, cls_kwargs=None):
        cls_kwargs, = base.none_to_empty_dict(cls_kwargs)

        batt_list = []
        for ev in attribute_dict['batt_list']:
            ev_elt, loaded_dict = base.build_from_id(
                ev, context_dict=context_dict, loaded_dict=loaded_dict)
            batt_list.append(ev_elt)

        out_obj = cls(attribute_dict['timestamp'], batt_list, **cls_kwargs)
        return out_obj, loaded_dict
