"""
This module contains methods for directly interacting with the simulator. 
"""
def get_active_EVs():
    """ Returns a list of active EVs for use by the algorithm.

    :return: list of EVs currently plugged in and not finished charging
    """
    active_EVs = []
    return active_EVs


def submit_schedules(schedules):
    """ Sends scheduled charging rates the the appropiate next step (simulator or influxDB).
    
    :param schedules: (dict) Dictionary where key is the id of the EV and value is a list of scheduled charging rates.
    :return: None
    """
    pass