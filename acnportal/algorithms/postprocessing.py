# coding=utf-8
"""
Postprocessing functions for scheduling algorithms.
"""
import numpy as np

from acnportal.acnsim.interface import InfrastructureInfo
from acnportal.acnsim.simulator import InvalidScheduleError


def format_array_schedule(
    array_schedule: np.ndarray, infrastructure: InfrastructureInfo
):
    """ Convert schedule in array format into dictionary format.

    Args:
        array_schedule (np.array): Array of charging rates, where each
            row is the charging rates for a specific session. Rows should be in
            the same order as the station_ids attribute of infrastructure. If
            array_schedule is a 1-d array, it will be expanded into lists of
            dimension length 1 for each station.
        infrastructure (InfrastructureInfo): Description of the electrical
            infrastructure.

    Returns:
        Dict[str, List[float]]: Dictionary mapping a station_id to a list of
            charging rates.
    """
    if len(infrastructure.station_ids) != len(array_schedule):
        raise InvalidScheduleError(
            f"Proposed array_schedule has a length that is different from the number "
            f"of stations in the network."
            f"\nNumber of  stations: {len(infrastructure.station_ids)}"
            f"\nNumber of schedules: {len(array_schedule)}"
        )
    schedule = {}
    for i, station_id in enumerate(infrastructure.station_ids):
        if np.isscalar(array_schedule[i]):
            schedule[station_id] = [array_schedule[i]]
        else:
            schedule[station_id] = array_schedule[i].tolist()
    return schedule
