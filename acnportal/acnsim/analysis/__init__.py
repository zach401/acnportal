import numpy as np
import warnings
import datetime


def aggregate_current(sim):
    """ Calculate the time series of aggregate current of all EVSEs within a simulation.

    Args:
        sim (Simulator): A Simulator object which has been run.

    Returns:
        np.Array: A numpy ndarray of the aggregate current at each time. [A]
    """
    return sim.charging_rates.sum(axis=0)


def aggregate_power(sim):
    """ Calculate the time series of aggregate power of all EVSEs within a simulation.

    Args:
        sim (Simulator): A Simulator object which has been run.

    Returns:
        np.Array: A numpy ndarray of the aggregate power at each time. [kW]
    """
    return sim.network._voltages.T.dot(sim.charging_rates) / 1000


def constraint_currents(sim, return_magnitudes=False, constraint_ids=None):
    """ Calculate the time series of current for each constraint in the ChargingNetwork for a simulation.

    Args:
        sim (Simulator): A Simulator object which has been run.
        return_magnitudes (bool): If true, return constraint currents as real magnitudes instead of complex numbers.
        constraint_ids (List[str]): List of constraint names for which the current should be returned. If None, return
            all constraint currents.

    Returns:
        Dict (str, np.Array): A dictionary mapping the name of a constraint to a numpy array of the current subject to
            that constraint at each time.
    """
    if constraint_ids is None:
        constraint_ids = sim.network.constraint_index

    currents_list = sim.network.constraint_current(
        sim.charging_rates, constraints=constraint_ids
    )

    if not return_magnitudes:
        currents_list = np.abs(currents_list)
    # Ensure constraint_ids have correct order relative to constraint_index in network
    constraint_ids = [
        constraint_id
        for constraint_id in sim.network.constraint_index
        if constraint_id in constraint_ids
    ]

    return {constraint_ids[i]: currents_list[i] for i in range(len(constraint_ids))}


def proportion_of_energy_delivered(sim):
    """ Calculate the percentage of total energy delivered over total energy requested.

    Args:
        sim (Simulator): A Simulator object which has been run.

    Returns:
        float: Proportion of total energy requested which was delivered during the simulation.
    """
    total_requested = sum(ev.requested_energy for ev in sim.ev_history.values())
    total_delivered = total_energy_delivered(sim)
    return total_delivered / total_requested


def total_energy_delivered(sim):
    """ Calculate total energy delivered in kWh.

    Args:
        sim (Simulator): A Simulator object which has been run.

    Returns:
        float: Total energy delivered during the simulation [kWh]
    """
    total_delivered = sum(ev.energy_delivered for ev in sim.ev_history.values())
    return total_delivered


def total_energy_requested(sim):
    """ Calculate total energy requested in kWh.

    Args:
        sim (Simulator): A Simulator object which has been run.

    Returns:
        float: Total energy requested during the simulation [kWh]
    """
    total_requested = sum(ev.requested_energy for ev in sim.ev_history.values())
    return total_requested


def proportion_of_demands_met(sim, threshold=0.1):
    """ Calculate the percentage of charging sessions where the energy request was met.

    Args:
        sim (Simulator): A Simulator object which has been run.
        threshold (float): Close to finished a session should be to be considered finished. Default: 0.1. [kW]

    Returns:
        float: Proportion of sessions where the energy demand was fully met.
    """
    finished = sum(
        1 for ev in sim.ev_history.values() if ev.remaining_demand < threshold
    )
    return finished / len(sim.ev_history)


def current_unbalance(sim, phase_ids, unbalance_type="NEMA", type=None):
    """ Calculate the current unbalance for each time in simulation.

    Supports two definitions of unbalance.
    1)  The NEMA definition defined as the ratio of the maximum deviation of an RMS current from the average RMS current
        over the average RMS current.
            (max(|I_a|, |I_b|, |I_c|) - 1/3 (|I_a| + |I_b| + |I_c|)) / (1/3 (|I_a| + |I_b| + |I_c|))

    See https://www.powerstandards.com/Download/Brief%20Discussion%20of%20Unbalance%20Definitions.pdf for more info.

    Args:
        sim (Simulator): A Simulator object which has been run.
        phase_ids (List[str]): List of length 3 where each element is the identifier of phase A, B, and C respectively.
        unbalance_type (str): Method to use for calculating phase unbalance. Acceptable values are 'NEMA'.

    Returns:
        List[float]: Time series of current unbalance as a list with one value per timestep.
    """
    if type is not None:
        warnings.warn(
            "current_unbalance kwarg 'type' is deprecated. "
            "Use 'unbalance_type' instead.",
            DeprecationWarning,
        )
        unbalance_type = type
    if unbalance_type == "NEMA":
        return _nema_current_unbalance(sim, phase_ids)
    else:
        raise ValueError("type must be NEMA not {0}".format(unbalance_type))


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
    return (np.max(currents, axis=0) - np.mean(currents, axis=0)) / np.mean(
        currents, axis=0
    )


def energy_cost(sim, tariff=None):
    """ Calculate the total energy cost of the simulation.

    Args:
        sim (Simulator): A Simulator object which has been run.
        tariff (TimeOfUseTariff): Tariff structure to use when calculating energy costs.

    Returns:
        float: Total energy cost of the simulation ($)

    """
    if tariff is None:
        if "tariff" in sim.signals:
            tariff = sim.signals["tariff"]
        else:
            raise ValueError("No pricing method is specified.")
    agg = aggregate_power(sim)
    energy_costs = tariff.get_tariffs(sim.start, len(agg), sim.period)
    return np.array(energy_costs).dot(agg) * (sim.period / 60)


def demand_charge(sim, tariff=None):
    """ Calculate the total demand charge of the simulation.

    Note this is only an accurate depiction of true demand charge if the simulation is exactly one billing period
    long.

    Args:
        sim (Simulator): A Simulator object which has been run.
        tariff (TimeOfUseTariff): Tariff structure to use when calculating energy costs.

    Returns:
        float: Total demand charge incurred by the simulation ($)

    """
    if tariff is None:
        if "tariff" in sim.signals:
            tariff = sim.signals["tariff"]
        else:
            raise ValueError("No pricing method is specified.")
    agg = aggregate_power(sim)
    dc = tariff.get_demand_charge(sim.start)
    return dc * np.max(agg)


def datetimes_array(sim):
    """ Return a numpy array of datetimes over which the simulation was
    run.

    The resolution of the datetimes list is equal to the period of the
    simulation, and the number of datetimes in the returned list is
    equal to the number of iterations of the simulation.

    Note that timezone information is not included with the datetime array.

    Args:
        sim (Simulator): A Simulator object which has been run.

    Returns:
        np.ndarray[np.datetime64]: List of datetimes over which the
            simulation was run.

    Warns:
        UserWarning: If this is called before sim is complete.
    """
    if not sim.event_queue.empty():
        warnings.warn(
            "Simulation incomplete; not all datetimes will be included.", UserWarning
        )
    no_tz_start: datetime = sim.start.replace(tzinfo=None)
    return np.array(
        [
            np.datetime64(no_tz_start + datetime.timedelta(minutes=sim.period * i))
            for i in range(sim.iteration)
        ]
    )
