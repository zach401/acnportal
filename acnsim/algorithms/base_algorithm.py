class BaseAlgorithm:
    """ This is a base-class meant to be inherited from to implement new algorithms.
        Each new algorithm should override the schedule method.
    """

    def __init__(self):
        self.interface = None

    def schedule(self, active_EVs):
        """
        Creates a schedule of charging rates for each EV in the active_EV list.
        NOT IMPLEMENTED IN BaseAlgorithm.

        :param active_EVs: List of EVs who should be scheduled
        :return: A dictionary with key: session_id and value: schedule of charging rates as a list.
        """

        schedules = {}
        return schedules

    def run(self):
        """ Runs one instance of the scheduling algorithm and submits the resulting schedule.

        :return: None
        """
        active_EVs = self.interface.get_active_evs()
        schedules = self.schedule(active_EVs)
        return schedules
        # self.interface.submit_schedules(schedules)

    def register_interface(self, interface):
        """
        Used internally by the simulator to set up the required dependencies to provide the resources needed
        to write a scheduling algorithm.

        :param Interface interface: The simulation API
        :return: None
        """
        self.interface = interface
