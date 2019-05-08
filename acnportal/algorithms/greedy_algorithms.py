import math

import numpy as np
from .sorted_algorithms import SortedSchedulingAlgo


class GreedyCostMinimization(SortedSchedulingAlgo):
    """ Greedy algorithm which minimizes cost for each EV individually.

    For this family of algorithms, active EVs are first sorted by some metric, then each EV is allowed to select a
    schedule which minimizes its own cost. Once an EV has selected its schedule, it is fixed and must be accounted
    for in constraints for all future arrivals.

    The argument sort_fn controlled how the EVs are sorted.

    Args:
        sort_fn (Callable[List[EV]]): Function which takes in a list of EVs and returns a list of the same EVs but
            sorted according to some metric.
    """

    def schedule(self, active_evs):
        """ Schedule EVs using greedy minimum cost scheme.

        See class documentation for description of the algorithm.

        Args:
            active_evs (List[EV]): see BaseAlgorithm

        Returns:
            Dict[str, List[float]]: see BaseAlgorithm
        """
        if len(active_evs) < 1:
            return {}
        T = int(math.ceil(np.max([ev.departure for ev in active_evs]))) + 1
        t = self.interface.get_current_time()

        ev_queue = self._sort_fn(active_evs)
        schedule = {ev.station_id: [0] * (T - t) for ev in active_evs}
        for ev in ev_queue:
            prices = self.interface.get_prices(t, ev.departure - t)
            preferences = np.argsort(prices, kind='mergesort')
            e_del = 0
            for i in preferences:
                if e_del >= ev.remaining_demand:
                    break
                charging_rate = self.max_feasible_rate(ev.station_id, self.interface.max_pilot_signal(ev.station_id),
                                                       schedule, i, eps=0.01)
                schedule[ev.station_id][i] = charging_rate
                e_del += charging_rate
        return schedule
