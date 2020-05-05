import numpy as np


class UpperBoundEstimatorBase:
    """ Abstract base class meant to be inherited from to implement new
        algorithms to estimate upper bounds on charging rate imposed by the
        EV's battery or on-board charger.

        Subclassed must implement the get_maximum_rates method.
    """

    def __init__(self):
        self._interface = None

    @property
    def interface(self):
        """ Return the algorithm's interface with the environment.

        Returns:
            Interface: An interface to the enviroment.

        Raises:
            ValueError: Exception raised if interface is accessed prior to an
                interface being registered.
        """
        if self._interface is not None:
            return self._interface
        else:
            raise ValueError(
                'No interface has been registered yet. Please call '
                'register_interface prior to using the algorithm.')

    def register_interface(self, interface):
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

    def get_maximum_rates(self, sessions):
        """ Return the maximum rate allowed for these EVs under the
            rampdown algorithm.

        Args:
            sessions list(SessionInfo): List of sessions

        Returns:
            dict(str, float): Dictionary mapping session_ids to maximum
                charging rates.
        """
        raise NotImplementedError('Rampdown is an abstract class.')


class SimpleRampdown(UpperBoundEstimatorBase):
    """ Simple algorithm reclaiming unused charging capacity.

        Implements abstract class UpperBoundEstimatorBase.

        The maximum pilot is reduced whenever the actual charging rate is
        more than down_threshold lower than the pilot signal. The maximum
        pilot is increased by up_increment, whenever the actual charging rate
        is within up_threshold of the pilot signal.
    """

    def __init__(self, up_threshold=2, down_threshold=1, up_increment=1):
        super().__init__()
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.up_increment = up_increment
        self.upper_bounds = {}

    def get_maximum_rates(self, sessions):
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

            # The we can only apply the rampdown algorithm if we have
            # data from from the last time period of pilot signal and
            # observed charging current.
            if session.session_id in prev_pilot:
                previous_pilot = prev_pilot[session.session_id]
                previous_rate = prev_rate[session.session_id]
                unused_capacity = previous_pilot - previous_rate
                ub = self.upper_bounds[session.session_id]
                if unused_capacity > self.down_threshold:
                    ub = previous_rate + self.up_increment
                elif unused_capacity < self.up_threshold:
                    ub += self.up_increment
                max_pilot = self.interface.max_pilot_signal(session.station_id)
                ub = np.clip(ub, a_min=0, a_max=max_pilot)
                self.upper_bounds[session.session_id] = ub
        return self.upper_bounds
