import random
from collections import deque
import warnings

from acnportal.acnsim.network.charging_network import ChargingNetwork


class AffinityChargingNetwork(ChargingNetwork):
    def __init__(self, early_departure=False):
        """ Extends ChargingNetwork to support non-deterministic space assignment."""
        super().__init__()
        self.waiting_queue = deque()
        self.swaps = 0
        self.never_charged = 0
        self.early_unplug = 0
        self.early_departure = early_departure

    def available_evses(self):
        """ Return a list of all EVSEs which do not have an EV attached. """
        return [evse_id for evse_id, evse in self._EVSEs.items() if evse.ev is None]

    def arrive(self, ev):
        """ Attaches EV to a random EVSE if one is available, otherwise places EV in the waiting queue.

        Args:
            ev (EV): EV object which is arriving to the network.

        Returns:
            None
        """
        available_spots = self.available_evses()
        if len(available_spots) > 0 and len(self.waiting_queue) == 0:
            chosen_spot = random.choice(available_spots)
            ev.station_id = chosen_spot
            super().arrive(ev)
        else:
            self.waiting_queue.append(ev)

    def depart(self, ev):
        """ Detach the given EV from its EVSE and, if the waiting queue is empty

        Args:
            ev:

        Returns:

        """
        # if ev.station_id in
        if ev.station_id in self._EVSEs and ev is self.get_ev(ev.station_id):
            super().depart(ev)
            if len(self.waiting_queue) > 0:
                next_ev = self.waiting_queue.popleft()
                next_ev.station_id = ev.station_id
                super().arrive(next_ev)
                self.swaps += 1
        elif ev in self.waiting_queue:
            self.waiting_queue.remove(ev)
            self.never_charged += 1
        # else:
        #     warnings.warn('EV {0} cannot depart as it is neither plugged in or in the waiting queue.')

    def post_charging_update(self):
        if self.early_departure:
            fully_charged_evs = [evse.ev for evse in self._EVSEs.values() if evse.ev is not None and evse.ev.fully_charged]
            for ev in fully_charged_evs:
                if len(self.waiting_queue) > 0:
                    self.depart(ev)





