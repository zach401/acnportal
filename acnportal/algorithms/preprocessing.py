# coding=utf-8
"""
Preprocessing functions for scheduling algorithms.
"""
from typing import List
import numpy as np

from acnportal.acnsim.interface import SessionInfo, InfrastructureInfo
from .upper_bound_estimator import UpperBoundEstimatorBase
from .utils import infrastructure_constraints_feasible, remaining_amp_periods


def enforce_pilot_limit(
    active_sessions: List[SessionInfo], infrastructure: InfrastructureInfo
) -> List[SessionInfo]:
    """ Update the max_rates vector for each session to be less than the max
        pilot supported by its EVSE.

    Args:
        active_sessions (List[SessionInfo]): List of SessionInfo objects for
            all active charging sessions.
        infrastructure (InfrastructureInfo): Description of the charging
            infrastructure.

    Returns:
        List[SessionInfo]: Active sessions with max_rates updated to be at
            most the max_pilot of the corresponding EVSE.
    """
    for session in active_sessions:
        i = infrastructure.get_station_index(session.station_id)
        session.max_rates = np.minimum(session.max_rates, infrastructure.max_pilot[i])
    return active_sessions


def reconcile_max_and_min(session: SessionInfo, choose_min: bool = True) -> SessionInfo:
    """ Modify session.max_rates[t] to equal session.min_rates[t] for times
        when max_rates[t] < min_rates[t]

    Args:
        session (SessionInfo): Session object.
        choose_min (bool): If True, when in conflict defer to the minimum
            rate. If False, defer to maximum.

    Returns:
        SessionInfo: session modified such that max_rates[t] is never less
            than min_rates[t]
    """
    mask = session.max_rates < session.min_rates
    if choose_min:
        session.max_rates[mask] = session.min_rates[mask]
    else:
        session.min_rates[mask] = session.max_rates[mask]
    return session


def expand_max_min_rates(active_sessions: List[SessionInfo]) -> List[SessionInfo]:
    """ Expand max_rates and min_rates to vectors if they are scalars. Doing so is
    helpful for scheduling algorithms that use a Model Predictive Control framework
    such as CVXPY.

    Args:
        active_sessions (List[SessionInfo]): List of SessionInfo objects for
            all active charging sessions.

    Returns:
        List[SessionInfo]: Active sessions with max_rates and min_rates
            expanded into vectors of length remaining_time.
    """
    for session in active_sessions:
        if np.isscalar(session.max_rates):
            session.max_rates = np.full(session.remaining_time, session.max_rates)
        if np.isscalar(session.min_rates):
            session.min_rates = np.full(session.remaining_time, session.min_rates)
    return active_sessions


def apply_upper_bound_estimate(
    ub_estimator: UpperBoundEstimatorBase, active_sessions: List[SessionInfo]
) -> List[SessionInfo]:
    """ Update max_rate in each SessionInfo object.

    If rampdown max_rate is less than min_rate, max_rate is set equal to min_rate.

    Args:
        ub_estimator (UpperBoundEstimatorBase): UpperBoundEstimatorBase-like
            object which estimates an upper bound on the charging rate of
            each EV based on historical data.
        active_sessions (List[SessionInfo]): List of SessionInfo objects for
            all active charging sessions.

    Returns:
        List[SessionInfo]: Active sessions with updated max_rate using rampdown.
    """
    new_sessions = expand_max_min_rates(active_sessions)
    upper_bounds = ub_estimator.get_maximum_rates(active_sessions)
    for j, session in enumerate(new_sessions):
        session.max_rates = np.minimum(
            session.max_rates, upper_bounds.get(session.station_id, float("inf"))
        )
        new_sessions[j] = reconcile_max_and_min(session)
        if np.any(session.max_rates < 32):
            pass
    return new_sessions


def apply_minimum_charging_rate(
    active_sessions: List[SessionInfo],
    infrastructure: InfrastructureInfo,
    period: int,
    override: float = float("inf"),
) -> List[SessionInfo]:
    """ Modify active_sessions so that min_rates[0] is equal to the greater of
        the session minimum rate and the EVSE minimum pilot. Sessions have their min
        rates applied in order of remaining time; i.e., sessions with less time
        remaining are allocated their min rates first.

    Args:
        active_sessions (List[SessionInfo]): List of SessionInfo objects for
            all active charging sessions.
        infrastructure (InfrastructureInfo): Description of the charging
            infrastructure.
        period (int): Length of each time period in minutes.
        override (float): Alternative minimum pilot which overrides the EVSE
            minimum if the EVSE minimum is greater than override.

    Returns:
        List[SessionInfo]: Active sessions with updated minimum charging rate
            for the first control period.
    """
    session_queue = sorted(active_sessions, key=lambda x: x.remaining_time)
    session_queue = expand_max_min_rates(session_queue)
    rates = np.zeros(len(infrastructure.station_ids))
    for j, session in enumerate(session_queue):
        i = infrastructure.station_ids.index(session.station_id)
        rates[i] = min(infrastructure.min_pilot[i], override)
        if rates[i] <= remaining_amp_periods(
            session, infrastructure, period
        ) and infrastructure_constraints_feasible(rates, infrastructure):
            # Preserve existing min_rate if it is greater than the new one
            session.min_rates[0] = max(rates[i], session.min_rates[0])
            # Increase the maximum rate if it is less than the new min.
            session_queue[j] = reconcile_max_and_min(session)
            # Keep this session as active
        else:
            # If an EV cannot be charged at the minimum rate, it should not be charged
            # in a solution to this problem. So, its max and min rates are set to 0.
            rates[i] = 0
            session.min_rates[0] = 0
            session.max_rates[0] = 0
    return session_queue


def remove_finished_sessions(
    active_sessions: List[SessionInfo],
    infrastructure: InfrastructureInfo,
    period: float,
) -> List[SessionInfo]:
    """ Remove any sessions where the remaining demand is less than threshold.
    Here, the threshold is defined as the amount of energy delivered by charging at
    the min_pilot of a session's station, at the station's voltage, for one simulation
    period.

    Args:
        active_sessions (List[SessionInfo]): List of SessionInfo objects for
            all active charging sessions.
        infrastructure (InfrastructureInfo): Description of the charging
            infrastructure.
        period (float): Length of each time period in minutes.


    Returns:
        List[SessionInfo]: Active sessions without any sessions that are finished.

    """
    modified_sessions = []
    for s in active_sessions:
        station_index = infrastructure.get_station_index(s.station_id)
        threshold = (
            infrastructure.min_pilot[station_index]
            * infrastructure.voltages[station_index]
            / (60 / period)
            / 1000
        )  # kWh
        if s.remaining_demand > threshold:
            modified_sessions.append(s)
    return modified_sessions
