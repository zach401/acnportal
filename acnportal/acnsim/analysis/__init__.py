import numpy as np


def aggregate_current(sim):
    """ Calculate the time series of aggregate current of all EVSEs within a simulation.

    Args:
        sim (Simulator): A Simulator object which has been run.

    Returns:
        np.Array: A numpy array of the aggregate current at each time.
    """
    return sum(np.array(rates) for rates in sim.charging_rates.values())


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
