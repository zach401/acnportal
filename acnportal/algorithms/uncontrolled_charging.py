# coding=utf-8
""" Simple algorithm for uncontrolled EV charging. """
from typing import List

from .base_algorithm import BaseAlgorithm
from ..acnsim.interface import SessionInfo


class UncontrolledCharging(BaseAlgorithm):
    """ Simple algorithm for uncontrolled EV charging.

    Implements abstract class BaseAlgorithm.

    This algorithm ignores all infrastructure constraints.
    All EVs will be charged as quickly as possible according to their maximum charging
    rate and their battery dynamics.
    """

    def __init__(self) -> None:
        super().__init__()
        # Call algorithm each period since it only returns a rate for the next period.
        self.max_recompute = 1

    def schedule(self, active_sessions: List[SessionInfo]):
        """ Schedule each EV to charge as quickly as possible ignoring all
        infrastructure constraints.

        Implements abstract method schedule from BaseAlgorithm.

        See class documentation for description of the algorithm.

        This algorithm SHOULD NOT be run on physical infrastructure which has safety
        constraints as it could (and likely will) violate those constraints.

        Args:
            active_sessions (List[SessionInfo]): see BaseAlgorithm

        Returns:
            Dict[str, List[float]]: see BaseAlgorithm
        """
        schedule = {}
        for session in active_sessions:
            schedule[session.station_id] = [
                self.interface.max_pilot_signal(session.station_id)
            ]
        return schedule
