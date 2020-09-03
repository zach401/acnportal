import random
from collections import deque
import warnings

from acnportal.acnsim.network.charging_network import ChargingNetwork


class StochasticNetwork(ChargingNetwork):
    def __init__(self, early_departure=False):
        """ Extends ChargingNetwork to support non-deterministic space assignment."""
        super().__init__()
        self.waiting_queue = deque()
        self.early_departure = early_departure

        # Stats
        self.swaps = 0
        self.never_charged = 0
        self.early_unplug = 0

    def available_evses(self):
        """ Return a list of all EVSEs which do not have an EV attached. """
        return [evse_id for evse_id, evse in self._EVSEs.items() if evse.ev is None]

    def plugin(self, ev, station_id=None):
        """ Attaches EV to a random EVSE if one is available, otherwise places EV in the
            waiting queue.

        Args:
            ev (EV): EV object which is arriving to the network.
            station_id (str): Where the EV should be plugged in.

        Returns:
            None
        """
        available_spots = self.available_evses()
        if len(available_spots) > 0 and len(self.waiting_queue) == 0:
            chosen_spot = random.choice(available_spots)
            ev.station_id = chosen_spot
            super().plugin(ev, chosen_spot)
        else:
            self.waiting_queue.append(ev)

    def unplug(self, ev):
        """ Detach EV from a specific EVSE.

        Args:
            ev (EV): ID of the EVSE.

        Returns:
            None

        Raises:
            KeyError: Raised when the station id has not yet been registered.
        """
        if ev.station_id in self._EVSEs and ev is self.get_ev(ev.station_id):
            super().unplug(ev)
            if len(self.waiting_queue) > 0:
                next_ev = self.waiting_queue.popleft()
                next_ev.station_id = ev.station_id
                super().plugin(next_ev, ev.station_id)
                self.swaps += 1
        elif ev in self.waiting_queue:
            self.waiting_queue.remove(ev)
            self.never_charged += 1
        else:
            warnings.warn('EV {0} cannot depart as it is neither plugged in or in '
                          'the waiting queue.')

    def post_charging_update(self):
        if self.early_departure:
            fully_charged_evs = [
                evse.ev
                for evse in self._EVSEs.values()
                if evse.ev is not None and evse.ev.fully_charged
            ]
            for ev in fully_charged_evs:
                if len(self.waiting_queue) > 0:
                    self.unplug(ev)
