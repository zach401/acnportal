import numpy as np


class Rampdown:
    """ Abstract base class meant to be inherited from to implement new rampdown algorithms.

    Subclassed must implement the get_maximum_rates method.

    Attributes:
        interface (Interface): An interface to the environment.

    """
    def __init__(self):
        self._interface = None

    @property
    def interface(self):
        """ Return the algorithm's interface with the environment.

        Returns:
            Interface: An interface to the enviroment.

        Raises:
            ValueError: Exception raised if interface is accessed prior to an interface being registered.
        """
        if self._interface is not None:
            return self._interface
        else:
            raise ValueError('No interface has been registered yet. Please call register_interface prior to using the'
                             'algorithm.')

    def register_interface(self, interface):
        """ Register interface to the _simulator/physical system.

        This interface is the only connection between the algorithm and what it is controlling. Its purpose is to
        abstract the underlying network so that the same algorithms can run on a simulated environment or a physical
        one.

        Args:
            interface (Interface): An interface to the underlying network whether simulated or real.

        Returns:
            None
        """
        self._interface = interface

    def get_maximum_rates(self, evs):
        """ Return the maximum rate allowed for these EVs under the rampdown algorithm.

        Args:
            evs list(EV): List of EV objects to calculate maximum rates for

        Returns:
            dict(str, float): Dictionary mapping session_ids to maximum charging rates.

        """
        raise NotImplementedError('Rampdown is an abstract class.')
    

class SimpleRampdown(Rampdown):
    """ Simple algorithm reclaiming unused charging capacity.

    Implements abstract class Rampdown.

    The maximum pilot is reduced whenever the actual charging rate is more than down_threshold lower than
     the pilot signal. The maximum pilot is increased by up_increment, whenever the actual charging rate is within
     up_threshold of the pilot signal.
    """

    def __init__(self, up_threshold=2, down_threshold=1, up_increment=1):
        super().__init__()
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.up_increment = up_increment
        self.rampdown_rates = {}

    def get_maximum_rates(self, evs):
        """ Return the maximum rate allowed for these EVs by lowering according to the simple rampdown algorithm.

        The maximum pilot is reduced whenever the actual charging rate is more than down_threshold lower than
         the pilot signal. The maximum pilot is increased by up_increment, whenever the actual charging rate is within
         up_threshold of the pilot signal.

        Args:
            evs list(EV): List of EV objects to calculate maximum rates for

        Returns:
            dict(str, float): Dictionary mapping session_ids to maximum charging rates.

        """
        prev_pilot = self.interface.last_applied_pilot_signals
        prev_rate = self.interface.last_actual_charging_rate

        for ev in evs:
            if ev.session_id in prev_pilot:
                if ev.session_id not in self.rampdown_rates:
                    self.rampdown_rates[ev.session_id] = self.interface.max_pilot_signal(ev.station_id)

                if prev_pilot[ev.session_id] - prev_rate[ev.session_id] > self.down_threshold:
                    self.rampdown_rates[ev.session_id] = prev_rate[ev.session_id] + self.up_increment
                elif prev_pilot[ev.session_id] - prev_rate[ev.session_id] < self.up_threshold:
                    self.rampdown_rates[ev.session_id] += self.up_increment
                self.rampdown_rates[ev.session_id] = np.clip(self.rampdown_rates[ev.session_id],
                                                             a_min=0,
                                                             a_max=self.interface.max_pilot_signal(ev.station_id))

        return self.rampdown_rates