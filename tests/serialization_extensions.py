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
        context_dict, = base.none_to_empty_dict(context_dict)
        args_dict = super().to_dict(context_dict)
        args_dict['batt_list'] = \
            [ev.to_registry(context_dict=context_dict)['id']
             for ev in self.batt_list]
        return args_dict

    @classmethod
    def from_dict(cls, in_dict,
                  context_dict=None, loaded_dict=None, cls_kwargs=None):
        context_dict, loaded_dict, cls_kwargs = \
            base.none_to_empty_dict(context_dict, loaded_dict, cls_kwargs)
        batt_list = [base.read_from_id(ev,
                                       context_dict=context_dict,
                                       loaded_dict=loaded_dict)
                     for ev in in_dict['batt_list']]
        out_obj = cls(in_dict['timestamp'], batt_list, **cls_kwargs)
        return out_obj
