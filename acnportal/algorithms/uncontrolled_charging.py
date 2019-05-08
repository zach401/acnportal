from .base_algorithm import BaseAlgorithm


class UncontrolledCharging(BaseAlgorithm):
    """ Simple algorithm for uncontrolled EV charging.

    Implements abstract class BaseAlgorithm.

    This algorithm ignores all infrastructure constraints.
    All EVs will be charged as quickly as possible according to their maximum charging rate and their battery dynamics.

    """
    def schedule(self, active_evs):
        """ Schedule each EV to charge as quickly as possible ignoring all infrastructure constraints.

        Implements abstract method schedule from BaseAlgorithm.

        See class documentation for description of the algorithm.

        This algorithm SHOULD NOT be run on physical infrastructure which has safety constraints as it could
        (and likely will) violate those constraints.

        Args:
            active_evs (List[EV]): see BaseAlgorithm

        Returns:
            Dict[str, List[float]]: see BaseAlgorithm
        """
        schedule = {}
        for ev in active_evs:
            schedule[ev.station_id] = [self.interface.max_pilot_signal(ev.station_id)]
        return schedule
