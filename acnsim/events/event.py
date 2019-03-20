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


class RecomputeEvent(Event):
    """ Subclass of Event for when the algorithm should be recomputed."""
    def __init__(self, timestamp):
        super().__init__(timestamp)
        self.type = 'Recompute'
        self.precedence = 20