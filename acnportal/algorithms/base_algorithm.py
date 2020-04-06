class BaseAlgorithm:
    """ Abstract base class meant to be inherited from to implement new algorithms.

    Subclasses must implement the schedule method.

    Attributes:
        interface (Interface): An interface to the environment.
        max_recompute (int): Maximum number of periods between calling the scheduling algorithm even if no events occur.
            If None, the scheduling algorithm is only called when an event occurs. Default: None.
        rampdown (Rampdown-like): Algorithm to use for rampdown. Default: None.
    """

    def __init__(self, rampdown=None):
        self._interface = None
        self.max_recompute = None
        self.rampdown = rampdown

    def __repr__(self):
        arg_str = ", ".join([f"{key}={value}" for key, value in self.__dict__.items()])
        return f"{self.__class__.__name__}({arg_str})"

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
        if self.rampdown is not None:
            self.rampdown.register_interface(interface)

    def remove_active_evs_less_than_deadband(self, active_evs, deadband=None):
        """ Remove EVs from active_evs which have remaining demand less than the deadband limit of their EVSE.

        Args:
            active_evs (List[EV]): List of EV objects which are currently ready to be charged and not finished charging.

        Returns:
            List[EV]: List of EV objects which have remaining demand above the deadband limit of their EVSE.
        """
        new_active = []
        for ev in active_evs:
            continuous, allowable_rates = self.interface.allowable_pilot_signals(ev.station_id)
            if deadband is None:
                deadband_end = allowable_rates[0] if continuous else allowable_rates[1]
            else:
                deadband_end = deadband
            if self.interface.remaining_amp_periods(ev) >= deadband_end:
                new_active.append(ev)
        return new_active

    def schedule(self, active_evs):
        """ Creates a schedule of charging rates for each ev in the active_evs list.

        NOT IMPLEMENTED IN BaseAlgorithm. This method MUST be implemented in all subclasses.

        This method returns a schedule of charging rates for each
        Args:
            active_evs (List[EV]): List of EV objects which are currently ready to be charged and not finished charging.

        Returns:
            Dict[str, List[float]]: Dictionary mapping a station_id to a list of charging rates. Each charging rate is
                valid for one period measured relative to the current period, i.e. schedule['abc'][0] is the charging
                rate for station 'abc' during the current period and schedule['abc'][1] is the charging rate for the
                next period, and so on. If an algorithm only produces charging rates for the current time period, the
                length of each list should be 1. If this is the case, make sure to also set the maximum resolve period
                to be 1 period so that the algorithm will be called each period. An alternative is to repeat the
                charging rate a number of times equal to the max recompute period.
        """
        raise NotImplementedError

    def run(self):
        """ Runs the scheduling algorithm for the current period and returns the resulting schedules.

        Returns:
            See schedule.
        """
        schedules = self.schedule(self.interface.active_evs)
        return schedules
