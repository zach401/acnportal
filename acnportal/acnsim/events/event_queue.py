import heapq
from typing import Optional, Dict, Any, Tuple

from ..base import BaseSimObj


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

    @property
    def queue(self):
        """ Return the queue of events """
        return self._queue

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
            List[Event]: List of all events occurring before or during timestep.
        """
        self._timestep = timestep
        current_events = []
        while not self.empty() and self._queue[0][0] <= self._timestep:
            current_events.append(self.get_event())
        return current_events

    def get_last_timestamp(self):
        """ Return the timestamp of the last event (chronologically) in the event queue

        Returns:
            int: Last timestamp in the event queue, or None if the
                event queue is empty.
        """
        if not self.empty():
            return max(self._queue, key=lambda x: x[0])[0]
        else:
            return None

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ Implements BaseSimObj._to_dict. """
        attribute_dict = {"_timestep": self._timestep}

        event_queue = []
        for (ts, event) in self._queue:
            # noinspection PyProtectedMember
            registry, context_dict = event._to_registry(context_dict=context_dict)
            event_queue.append((ts, registry["id"]))
        attribute_dict["_queue"] = event_queue

        return attribute_dict, context_dict

    @classmethod
    def _from_dict(
        cls,
        attribute_dict: Dict[str, Any],
        context_dict: Dict[str, Any],
        loaded_dict: Optional[Dict[str, BaseSimObj]] = None,
    ) -> Tuple[BaseSimObj, Dict[str, BaseSimObj]]:
        """ Implements BaseSimObj._from_dict. """
        out_obj = cls()
        out_obj._timestep = attribute_dict["_timestep"]

        event_queue = []
        for (ts, event) in attribute_dict["_queue"]:
            # noinspection PyProtectedMember
            loaded_event, loaded_dict = BaseSimObj._build_from_id(
                event, context_dict, loaded_dict=loaded_dict
            )
            event_queue.append((ts, loaded_event))
        out_obj._queue = event_queue

        return out_obj, loaded_dict
