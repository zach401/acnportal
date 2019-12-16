import heapq
from .event import Event
from ..base import BaseSimObj, read_from_id


class EventQueue(BaseSimObj):
    """ Queue which stores simulation events.

    Args:
        events (List[Event]): A list of Event-like objects.
    """

    def __init__(self, events=None):
        self._queue = []
        if events is not None:
            self.add_events(events)
        self._timestep = 0

    def __len__(self):
        """ Return the length of this queue.

        Returns:
            int: Length of the queue.
        """
        return len(self._queue)

    def empty(self):
        """ Return if the queue is empty.

        Returns:
            bool: True if the queue is empty.
        """
        return len(self._queue) == 0

    def add_event(self, event):
        """ Add an event to the queue.

        Args:
            event (Event like): An Event-like object.

        Returns:
            None
        """
        heapq.heappush(self._queue, (event.timestamp, event))

    def add_events(self, events):
        """ Add multiple events at a time to the queue.

        Args:
            events (List[Event like]): A list of Event-like objects.

        Returns:
            None
        """
        for e in events:
            self.add_event(e)

    def get_event(self):
        """ Return the next event in the queue.

        Returns:
            Event like: The next event in the queue.
        """
        return heapq.heappop(self._queue)[1]

    def get_current_events(self, timestep):
        """ Return all events occurring before or during timestep.

        Args:
            timestep (int): Time index in periods.

        Returns:
            List[Event like]: List of all events occurring before or during timestep.
        """
        self._timestep = timestep
        current_events = []
        while len(self._queue) > 0 and self._queue[0][0] <= self._timestep:
            current_events.append(self.get_event())
        return current_events

    def get_last_timestamp(self):
        """ Return the timestamp of the last event (chronologically) in the event queue

        Returns:
            int: Last timestamp in the event queue, or -1 if the queue is empty
        """
        if self.empty():
            return -1
        return max(self._queue, key=lambda x: x[0])[0]

    
    def to_dict(self, context_dict={}):
        """ Converts the event queue into a JSON serializable dict

        Returns:
            JSON serializable
        """
        args_dict = {}

        args_dict['_queue'] = [(ts, event.to_registry(context_dict=context_dict)['id']) 
            for (ts, event) in self._queue]
        args_dict['_timestep'] = self._timestep
        return args_dict

    @classmethod
    def from_dict(cls, in_dict, context_dict={}, loaded_dict={}, cls_kwargs={}):
        out_obj = cls(**cls_kwargs)
        out_obj._queue = [(ts, read_from_id(event, context_dict=context_dict, loaded_dict=loaded_dict))
            for (ts, event) in in_dict['_queue']]
        out_obj._timestep = in_dict['_timestep']

        return out_obj