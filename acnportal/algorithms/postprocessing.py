import numpy as np


def format_array_schedule(array_schedule, infrastructure):
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
    schedule = {}
    for i, station_id in enumerate(infrastructure.station_ids):
        if np.isscalar(array_schedule[i]):
            schedule[station_id] = [array_schedule[i]]
        else:
            schedule[station_id] = array_schedule[i].tolist()
    return schedule
