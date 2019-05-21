import numpy as np
import cmath


def aggregate_current(sim):
    """ Calculate the time series of aggregate current of all EVSEs within a simulation.

    Args:
        sim (Simulator): A Simulator object which has been run.

    Returns:
        np.Array: A numpy array of the aggregate current at each time.
    """
    return sum(np.array(rates) for rates in sim.charging_rates.values())


def constraint_currents(sim, complex=False, constraint_ids=None):
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
            if complex:
                c = np.zeros(sim_length, dtype=np.csingle)
            else:
                c = np.zeros(sim_length)
            for t in range(sim_length):
                if complex:
                    c[t] = cs.constraint_current(constraint, sim.charging_rates, sim.network.phase_angles, t)
                else:
                    c[t] = np.abs(cs.constraint_current(constraint, sim.charging_rates, sim.network.phase_angles, t))
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


def proportion_of_demands_met(sim, threshold=0.1):
    """ Calculate the percentage of charging sessions where the energy request was met.

    Args:
        sim (Simulator): A Simulator object which has been run.
        threshold (float): Close to finished a session should be to be considered finished. Default: 0.1. [kW]

    Returns:
        float: Proportion of sessions where the energy demand was fully met.
    """
    finished = sum(1 for ev in sim.ev_history.values() if ev.remaining_demand < threshold)
    return finished / len(sim.ev_history)


def current_unbalance(sim, phase_ids, type='NEMA'):
    """ Calculate the current unbalance for each time in simulation.

    Supports two definitions of unbalance.
    1)  The NEMA definition defined as the ratio of the maximum deviation of an RMS current from the average RMS current
        over the average RMS current.
            (max(|I_a|, |I_b|, |I_c|) - 1/3 (|I_a| + |I_b| + |I_c|)) / (1/3 (|I_a| + |I_b| + |I_c|))
    2)  Symmetric Components definition defined as the ratio of the magnitude of the negative sequence component (I_-)
        over the magnitude of the positive sequence component (I_+).
            |I_-| / |I_+|

    See https://www.powerstandards.com/Download/Brief%20Discussion%20of%20Unbalance%20Definitions.pdf for more info.

    Args:
        sim (Simulator): A Simulator object which has been run.
        phase_ids (List[str]): List of length 3 where each element is the identifier of phase A, B, and C respectively.
        type (str): Method to use for calculating phase unbalance. Acceptable values are 'NEMA' and 'SYM_COMP'.

    Returns:
        List[float]: Time series of current unbalance as a list with one value per timestep.
    """
    if type == 'NEMA':
        return _nema_current_unbalance(sim, phase_ids)
    elif type == 'SYM_COMP':
        return _sym_comp_current_unbalance(sim, phase_ids)
    else:
        raise ValueError('type must be NEMA or SYM_COMP, not {0}'.format(type))


def _nema_current_unbalance(sim, phase_ids):
    """ Calculate the current unbalance using the NEMA definition.

    The NEMA definition defined as the ratio of the maximum deviation of an RMS current from the average RMS current
    over the average RMS current.
        (max(|I_a|, |I_b|, |I_c|) - 1/3 (|I_a| + |I_b| + |I_c|)) / (1/3 (|I_a| + |I_b| + |I_c|))

    Args:
        sim (Simulator): A Simulator object which has been run.
        phase_ids (List[str]): List of length 3 where each element is the identifier of phase A, B, and C respectively.

    Returns:
        List[float]: Time series of current unbalance as a list with one value per timestep.
    """
    currents_dict = constraint_currents(sim, constraint_ids=phase_ids)
    currents = np.vstack([currents_dict[phase] for phase in phase_ids])
    return (np.max(currents, axis=0) - np.mean(currents, axis=0)) / np.mean(currents, axis=0)


def _sym_comp_current_unbalance(sim, phase_ids):
    """ Calculate the current unbalance using the Symmetric Components definition.

    Symmetric Components definition defined as the ratio of the magnitude of the negative sequence component (I_-)
    over the magnitude of the positive sequence component (I_+).
        |I_-| / |I_+|

    Args:
        sim (Simulator): A Simulator object which has been run.
        phase_ids (List[str]): List of length 3 where each element is the identifier of phase A, B, and C respectively.

    Returns:
        List[float]: Time series of current unbalance as a list with one value per timestep.
    """
    currents_dict = constraint_currents(sim, complex=True, constraint_ids=phase_ids)
    currents = np.vstack([currents_dict[phase] for phase in phase_ids]).T
    alpha = cmath.rect(1, (2 / 3) * cmath.pi)
    A_inv = (1 / 3) * np.array([[1, 1, 1], [1, alpha, alpha ** 2], [1, alpha ** 2, alpha]])
    sym_comp = A_inv.dot(currents.T)

    current_unbalance = np.divide(np.abs(sym_comp[2]), np.abs(sym_comp[1]),
                                  out=np.full_like(np.abs(sym_comp[2]), np.nan),
                                  where=np.abs(sym_comp[1]) != 0)
    return current_unbalance
