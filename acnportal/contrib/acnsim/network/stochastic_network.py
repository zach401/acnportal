import random
from collections import OrderedDict
import warnings

from acnportal.acnsim.network.charging_network import ChargingNetwork
from acnportal.acnsim.models import EV


class StochasticNetwork(ChargingNetwork):
    def __init__(
        self,
        violation_tolerance: float = 1e-5,
        relative_tolerance: float = 1e-7,
        early_departure: bool = False,
    ):
        """ Extends ChargingNetwork to support non-deterministic space assignment."""
        super().__init__(violation_tolerance, relative_tolerance)
        self.waiting_queue = OrderedDict()
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
            ev (EV): EV object which will be attached to the EVSE.
            [Depreciated]
            station_id (str): ID of the EVSE.

        Returns:
            None

        Raises:
            KeyError: Raised when the station id has not yet been registered.
        """
        if station_id is not None:
            warnings.warn(
                "plugin arg 'station_id' is deprecated. Plugin now uses the value of ev.station_id directly.",
                DeprecationWarning,
            )
        available_spots = self.available_evses()
        if len(available_spots) > 0:  # and len(self.waiting_queue) == 0:
            chosen_spot = random.choice(available_spots)
            ev.update_station_id(chosen_spot)
            super().plugin(ev)
        else:
            ev.update_station_id(None)
            self.waiting_queue[ev.session_id] = ev
            self.waiting_queue.move_to_end(ev.session_id)

    def unplug(self, station_id: str, session_id: str = None) -> None:
        """ Detach EV from a specific EVSE.

        Args:
            station_id (str): ID of the EVSE.
            session_id (str): ID of the session to be unplugged.
        Returns:
            None

        Raises:
            KeyError: Raised when the station id has not yet been registered.
        """
        if session_id in self.waiting_queue:
            del self.waiting_queue[session_id]
            self.never_charged += 1
        elif station_id in self._EVSEs:
            if session_id is None:
                raise ValueError(
                    "StochasticNetwork requires a session_id to unplug an EV."
                )
            elif self._EVSEs[station_id].ev is None:
                pass
                # warnings.warn(
                #     f"Tried to remove EV with session_id {session_id} which was not "
                #     f"present at station {station_id}. Found no EV instead."
                # )
            elif session_id == self._EVSEs[station_id].ev.session_id:
                self._EVSEs[station_id].unplug()
                if len(self.waiting_queue) > 0:
                    _, next_ev = self.waiting_queue.popitem(last=False)
                    next_ev.update_station_id(station_id)
                    super().plugin(next_ev)
                    self.swaps += 1
        else:
            raise KeyError("Station {0} not found.".format(station_id))

    def post_charging_update(self):
        """ Unplug fully charged EVs, even if they are not scheduled to depart. """
        if self.early_departure:
            fully_charged_evs = [
                evse.ev
                for evse in self._EVSEs.values()
                if evse.ev is not None and evse.ev.fully_charged
            ]
            for ev in fully_charged_evs:
                if len(self.waiting_queue) > 0:
                    self.unplug(ev.station_id, ev.session_id)
                    self.early_unplug += 1
