import numpy as np


def aggregate_current(sim):
    """ Calculate the time series of aggregate current of all EVSEs within a simulation.

    Args:
        sim (Simulator): A Simulator object which has been run.

    Returns:
        np.Array: A numpy array of the aggregate current at each time.
    """
    return sum(np.array(rates) for rates in sim.charging_rates.values())


def constraint_currents(sim, constraint_ids=None):
    """ Calculate the time series of current for each constraint in the ChargingNetwork for a simulation.

    Args:
        sim (Simulator): A Simulator object which has been run.
        constraint_ids (List[str]): List of constraint names for which the current should be returned. If None, return
            all constraint currents.

    Returns:
        Dict (str, np.Array): A dictionary mapping the name of a constraint to a numpy array of the current subject to
            that constraint at each time.
    """
    cs = sim.network.constraint_set
    if constraint_ids is None:
        constraint_ids = set(c.name for c in cs.constraints)
    else:
        constraint_ids = set(constraint_ids)

    currents = {}
    sim_length = max(len(cr) for cr in sim.charging_rates.values())
    for constraint in cs.constraints:
        if constraint.name in constraint_ids:
            c = np.zeros(sim_length)
            for t in range(sim_length):
                c[t] = abs(cs.constraint_current(constraint, sim.charging_rates, t))
            currents[constraint.name] = c
    return currents



def proportion_of_energy_delivered(sim):
    """ Calculate the percentage of total energy delivered over total energy requested.

    Args:
        sim (Simulator): A Simulator object which has been run.

    Returns:
        float: Proportion of total energy requested which was delivered during the simulation.
    """
    total_requested = sum(ev.requested_energy for ev in sim.ev_history.values())
    total_delivered = sum(ev.energy_delivered for ev in sim.ev_history.values())
    return total_delivered / total_requested
