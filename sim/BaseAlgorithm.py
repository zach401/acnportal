import Interface

class BaseAlgorithm:
    """ This is a base-class meant to be inherited from to implement new algorithms.
        Each new algorithm should override the schedule method.
    """
    def __init__(self):
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
        active_EVs = Interface.get_active_EVs()
        schedules = self.schedule(active_EVs)
        Interface.submit_schedules(schedules)