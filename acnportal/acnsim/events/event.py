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

    def execute(self, sim):
        """ Execute actions associated with the event.

        Args:
            sim (Simulator): Simulator on which the event should be executed.

        Returns:
            None

        Raises:
            NotImplementedError: Event does not implement the execute() method, it is left to subclasses.
        """
        raise NotImplementedError('Error. Event does not implement the execute() method. Please use a derived class.')


class Arrival(Event):
    """ Subclass of Event for EV plugins.

    Args:
        timestamp (int): See Event.
        ev (EV): The EV which is arriving.
    """
    def __init__(self, timestamp, ev):
        super().__init__(timestamp)
        self.type = 'Arrival'
        self.ev = ev
        self.precedence = 10

    def execute(self, sim):
        """ Process an EV arriving to the network.

        Calls the network's plugin handler and stored the EV in the sim's ev_history.
        Also automatically creates a departure event.
        Triggers resolve.

        Args:
            sim (Simulator): Simulator on which the event should be executed.

        Returns:
            None
        """
        sim.print('Plugin Event...')
        sim.network.arrive(self.ev)
        sim.ev_history[self.ev.session_id] = self.ev
        sim.event_queue.add_event(Departure(self.ev.departure, self.ev))
        sim._resolve = True


class Departure(Event):
    """ Subclass of Event for EV unplugs.

    Args:
        timestamp (int): See Event.
        ev (EV): The EV which is departing.
    """
    def __init__(self, timestamp, ev):
        super().__init__(timestamp)
        self.type = 'Departure'
        self.ev = ev
        self.precedence = 0

    def execute(self, sim):
        """ Process an EV leaving the network.

        Calls the network's unplug handler and triggers resolve.

        Args:
            sim (Simulator): Simulator on which the event should be executed.

        Returns:
            None
        """
        sim.print('Unplug Event...')
        sim.network.depart(self.ev)
        sim._resolve = True


class Recompute(Event):
    """ Subclass of Event for when the algorithm should be recomputed."""
    def __init__(self, timestamp):
        super().__init__(timestamp)
        self.type = 'Recompute'
        self.precedence = 20

    def execute(self, sim):
        """ Triggers a resolve of the scheduling algorithm.

        Args:
            sim (Simulator): Simulator on which the event should be executed.

        Returns:
            None
        """
        sim.print('Recompute Event...')
        sim._resolve = True
