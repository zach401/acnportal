from Interface import Interface

class BaseAlgorithm:
    """ This is a base-class meant to be inherited from to implement new algorithms.
        Each new algorithm should override the schedule method.
    """
    def __init__(self, interface):
        self.interface = interface
        pass

    def schedule(self, active_EVs):
        """ Creates a schedule of charging rates for each EV in the active_EV list.
            NOT IMPLEMENTED IN BaseAlgorithm!!!

        :param active_EVs: List of EVs who should be scheduled
        :return: A dictionary with key: EV id and value: schedule of charging rates as a list.
        """

        schedules = {}
        return schedules
    
    def run(self):
        """ Runs one instance of the scheduling algorithm and submits the resulting schedule.

        :return: None
        """
        active_EVs = self.interface.get_active_EVs()
        schedules = self.schedule(active_EVs)
        self.interface.submit_schedules(schedules)

class EarliestDeadlineFirstAlgorithm(BaseAlgorithm):

    def __init__(self, interface):
        super().__init__(interface)

    def schedule(self, active_EVs):
        schedule = {}
        earliest_EV = self.get_earliest_EV(active_EVs)
        for ev in active_EVs:
            charge_rates = []
            if ev.session_id == earliest_EV.session_id:
                charge_rates.append(10)
            else:
                charge_rates.append(0)
            schedule[ev.session_id] = charge_rates
        return schedule

    def get_earliest_EV(self, EVs):
        earliest_EV = None
        for ev in EVs:
            if earliest_EV == None or earliest_EV.departure > ev.departure:
                earliest_EV = ev
        return earliest_EV