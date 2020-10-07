# coding=utf-8
"""
Abstract base algorithm for scheduled charging.
"""
import time
from typing import Optional, List, Dict, Union


class BaseAlgorithm:
    """ Abstract base class meant to be inherited from to implement new algorithms.

    Subclasses must implement the schedule method.

    Attributes:
        max_recompute (int): Maximum number of periods between calling the scheduling
            algorithm even if no events occur. If None, the scheduling algorithm is
            only called when an event occurs. Default: None.
    """

    _interface: Optional["Interface"]
    max_recompute: Optional[int]
    solve_stats: List[Dict[str, Union[float, int]]]

    def __init__(self) -> None:
        self._interface = None
        self.max_recompute = None
        self.solve_stats = []

    def __repr__(self) -> str:
        arg_str = ", ".join([f"{key}={value}" for key, value in self.__dict__.items()])
        return f"{self.__class__.__name__}({arg_str})"

    @property
    def interface(self) -> "Interface":
        """ Return the algorithm's interface with the environment.

        Returns:
            Interface: An interface to the environment.

        Raises:
            ValueError: Exception raised if interface is accessed prior to an interface
                being registered.
        """
        if self._interface is not None:
            return self._interface
        else:
            raise ValueError(
                "No interface has been registered yet. Please call register_interface "
                "prior to using the algorithm."
            )

    def register_interface(self, interface: "Interface") -> None:
        """ Register interface to the _simulator/physical system.

        This interface is the only connection between the algorithm and what it is
        controlling. Its purpose is to abstract the underlying network so that the same
        algorithms can run on a simulated environment or a physical one.

        Args:
            interface (Interface): An interface to the underlying network whether
            simulated or real.

        Returns:
            None
        """
        self._interface = interface

    def schedule(self, active_sessions: List["SessionInfo"]) -> Dict[str, List[float]]:
        """ Creates a schedule of charging rates for each ev in the active_evs list.

        NOT IMPLEMENTED IN BaseAlgorithm. This method MUST be implemented in all
        subclasses.

        This method returns a schedule of charging rates for each
        Args:
            active_sessions (List[SessionInfo]): List of SessionInfo objects
                which are currently ready to be charged and not finished
                charging.

        Returns:
            Dict[str, List[float]]: Dictionary mapping a station_id to a list of
                charging rates. Each charging rate is valid for one period measured
                relative to the current period, i.e. schedule['abc'][0] is the
                charging rate for station 'abc' during the current period and
                schedule['abc'][1] is the charging rate for the next period,
                and so on. If an algorithm only produces charging rates for the
                current time period, the length of each list should be 1. If this is
                the case, make sure to also set the maximum resolve period to be 1
                period so that the algorithm will be called each period. An
                alternative is to repeat the charging rate a number of times equal to
                the max recompute period.
        """
        raise NotImplementedError

    def run(self) -> Dict[str, List[float]]:
        """ Runs the scheduling algorithm for the current period and returns the
        resulting schedules.

        Returns:
            See schedule.
        """
        active_sessions = self.interface.active_sessions()
        start_time = time.perf_counter()
        schedules = self.schedule(active_sessions)
        solve_time = time.perf_counter() - start_time
        self.solve_stats.append(
            {"solve_time": solve_time, "active_sessions": len(active_sessions)}
        )
        return schedules
