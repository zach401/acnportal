import numpy as np

from acnportal.acnsim.interface import InfrastructureInfo, SessionInfo


def infrastructure_constraints_feasible(
    rates: np.ndarray,
    infrastructure: InfrastructureInfo,
    linear: bool = False,
    violation_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-7,
) -> bool:
    """ Return if a set of current magnitudes for each load are feasible.

    For a given constraint, the larger of the violation_tolerance
    and relative_tolerance is used to evaluate feasibility.

    This is a static version of acnsim.ChargingNetwork.is_feasible.

    Args:
        rates (np.ndarray): 2-D matrix with each row corresponding to
            an EVSE and each column corresponding to a time index in the schedule.
        infrastructure (InfrastructureInfo): The InfrastructureInfo object that contains
            information about the network constraints.
        linear (bool): If True, linearize all constraints to a more conservative
            but easier to compute constraint by ignoring the phase angle and taking
            the absolute value of all load coefficients. Default False.
        violation_tolerance (float): Absolute amount by which
            rates may violate network constraints. Default 1e-5
        relative_tolerance (float): Relative amount by which
            schedule_matrix may violate network constraints. Default 1e-7

    Returns:
        bool: If load_currents is feasible according to this set of
            constraints.
    """
    tol = np.maximum(
        violation_tolerance, relative_tolerance * infrastructure.constraint_limits
    )
    if not linear:
        phase_in_rad = np.deg2rad(infrastructure.phases)
        for j, v in enumerate(infrastructure.constraint_matrix):
            a = np.stack([v * np.cos(phase_in_rad), v * np.sin(phase_in_rad)])
            line_currents = np.linalg.norm(a @ rates, axis=0)
            if not np.all(
                line_currents <= infrastructure.constraint_limits[j] + tol[j]
            ):
                return False
    else:
        for j, v in enumerate(infrastructure.constraint_matrix):
            line_currents = np.linalg.norm(np.abs(v) @ rates, axis=0)
            if not np.all(
                line_currents <= infrastructure.constraint_limits[j] + tol[j]
            ):
                return False
    return True


def remaining_amp_periods(
    session: SessionInfo, infrastructure: InfrastructureInfo, period: float
) -> float:
    """ Return the session's remaining demand in A*periods. This function is a static
    version of acnsim.Interface.remaining_amp_periods.

    Args:
        session (SessionInfo): The SessionInfo object for which to get remaining demand.
        infrastructure (InfrastructureInfo): The InfrastructureInfo object that contains
            voltage information about the network.
        period (float): Period of the simulation in minutes.

    Returns:
        float: the EV's remaining demand in A*periods.
    """
    i = infrastructure.get_station_index(session.station_id)
    amp_hours = session.remaining_demand * 1000 / infrastructure.voltages[i]
    return amp_hours * 60 / period
