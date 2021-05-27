# coding=utf-8
"""
Classes implementing different methods of estimating upper bounds on charging rate.
"""
from typing import Optional, Dict, List

import numpy as np

from acnportal.acnsim.interface import SessionInfo, Interface


class UpperBoundEstimatorBase:
    """ Abstract base class meant to be inherited from to implement new
        algorithms to estimate upper bounds on charging rate imposed by the
        EV's battery or on-board charger.

        Subclassed must implement the get_maximum_rates method.
    """

    _interface: Optional[Interface]

    def __init__(self) -> None:
        self._interface = None

    @property
    def interface(self) -> Interface:
        """ Return the algorithm's interface with the environment.

        Returns:
            Interface: An interface to the environment.

        Raises:
            ValueError: Exception raised if interface is accessed prior to an
                interface being registered.
        """
        if self._interface is not None:
            return self._interface
        else:
            raise ValueError(
                "No interface has been registered yet. Please call "
                "register_interface prior to using the algorithm."
            )

    def register_interface(self, interface: Interface) -> None:
        """ Register interface to the _simulator/physical system.
            This interface is the only connection between the algorithm and
            what it is controlling. Its purpose is to abstract the
            underlying network so that the same algorithms can run on a
            simulated environment or a physical one.

        Args:
            interface (Interface): An interface to the underlying network
                whether simulated or real.
        Returns:
            None
        """
        self._interface = interface

    def get_maximum_rates(self, sessions: List[SessionInfo]) -> Dict[str, float]:
        """ Return the maximum rate allowed for these EVs under an upper bound
        estimation algorithm.

        NOT IMPLEMENTED IN UpperBoundEstimatorBase. This method MUST be implemented in
        all subclasses.

        Args:
            sessions list(SessionInfo): List of sessions

        Returns:
            dict(str, float): Dictionary mapping session_ids to maximum
                charging rates.
        """
        raise NotImplementedError("UpperBoundEstimatorBase is an abstract class.")


class SimpleRampdown(UpperBoundEstimatorBase):
    """ Simple algorithm reclaiming unused charging capacity.

        Implements abstract class UpperBoundEstimatorBase.

        The maximum pilot is reduced whenever the actual charging rate is
        more than down_threshold lower than the pilot signal. The maximum
        pilot is increased by up_increment, whenever the actual charging rate
        is within up_threshold of the current maximum pilot signal.
    """

    up_threshold: float
    down_threshold: float
    up_increment: float
    upper_bounds: Dict[str, float]

    def __init__(
        self,
        up_threshold: float = 1,
        down_threshold: float = 1,
        up_increment: float = 1,
    ) -> None:
        super().__init__()
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.up_increment = up_increment
        self.upper_bounds = {}

    def get_maximum_rates(self, sessions: List[SessionInfo]) -> Dict[str, float]:
        """ Return the maximum rate allowed for these EVs by lowering according
            to the simple rampdown algorithm.

            (See class description for algorithm details)

        Args:
            sessions list(SessionInfo): List of sessions

        Returns:
            dict(str, float): Dictionary mapping session_ids to maximum charging rates.
        """
        prev_pilot = self.interface.last_applied_pilot_signals
        prev_rate = self.interface.last_actual_charging_rate

        for session in sessions:
            if session.session_id not in self.upper_bounds:
                ub = self.interface.max_pilot_signal(session.station_id)
                self.upper_bounds[session.session_id] = ub

            # We can only apply the rampdown algorithm if we have
            # data from from the last time period of pilot signal and
            # observed charging current.
            if session.session_id in prev_pilot:
                previous_pilot = prev_pilot[session.session_id]
                previous_rate = prev_rate[session.session_id]
                ub = self.upper_bounds[session.session_id]
                if previous_pilot - previous_rate > self.down_threshold:
                    ub = previous_rate + self.up_increment
                elif ub - previous_rate < self.up_threshold:
                    ub += self.up_increment
                max_pilot = self.interface.max_pilot_signal(session.station_id)
                # ub is a float, so np.clip should return a float. The casting is to
                # satisfy the type checker.
                ub = float(np.clip(ub, a_min=0, a_max=max_pilot))
                self.upper_bounds[session.session_id] = ub
        return self.upper_bounds
