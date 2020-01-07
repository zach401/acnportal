from acnportal import acnsim

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
