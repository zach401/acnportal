from typing import List
from copy import deepcopy
import numpy as np

from acnportal.algorithms import UpperBoundEstimatorBase
from acnportal.acnsim.interface import SessionInfo, InfrastructureInfo
from acnportal.algorithms.utils import infrastructure_constraints_feasible


def enforce_pilot_limit(active_sessions: List[SessionInfo],
                        infrastructure: InfrastructureInfo):
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
    new_sessions = deepcopy(active_sessions)
    for session in new_sessions:
        i = infrastructure.get_station_index(session.station_id)
        session.max_rates = np.minimum(session.max_rates,
                                       infrastructure.max_pilot[i])
    return new_sessions


def reconcile_max_and_min(session: SessionInfo):
    """ Modify session.max_rates[t] to equal session.min_rates[t] for times
        when max_rates[t] < min_rates[t]

    Args:
        session (SessionInfo): Session object.

    Returns:
        SessionInfo: session modified such that max_rates[t] is never less
            than min_rates[t]
    """
    new_sess = deepcopy(session)
    mask = new_sess.max_rates < new_sess.min_rates
    new_sess.max_rates[mask] = new_sess.min_rates[mask]
    return new_sess


def expand_max_min_rates(active_sessions: List[SessionInfo]):
    """ Expand max_rates and min_rates to vectors if they are scalars.

    Args:
        active_sessions (List[SessionInfo]): List of SessionInfo objects for
            all active charging sessions.

    Returns:
        List[SessionInfo]: Active sessions with max_rates and min_rates
            expanded into vectors of length remaining_time.
    """
    new_sessions = deepcopy(active_sessions)
    for session in new_sessions:
        if np.isscalar(session.max_rates):
            session.max_rates = np.full(session.max_rates,
                                        session.remaining_time)
        if np.isscalar(session.min_rates):
            session.min_rates = np.full(session.min_rates,
                                        session.remaining_time)
    return new_sessions


def apply_upper_bound_estimate(ub_estimator: UpperBoundEstimatorBase,
                               active_sessions: List[SessionInfo]):
    """ Update max_rate in each SessionInfo object.

        If rampdown max_rate is less than min_rate, max_rate is set
        equal to min_rate.

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
        session.max_rates = np.minimum(session.max_rates,
                                       upper_bounds.get(session.station_id,
                                                        float('inf')))
        new_sessions[j] = reconcile_max_and_min(session)
    return new_sessions


def apply_minimum_charging_rate(active_sessions: List[SessionInfo],
                                infrastructure: InfrastructureInfo,
                                override=float('inf')):
    """ Modify active_sessions so that min_rates[0] is equal to the greater of
        the session minimum rate and the EVSE minimum pilot.

    Args:
        active_sessions (List[SessionInfo]): List of SessionInfo objects for
            all active charging sessions.
        infrastructure (InfrastructureInfo): Description of the charging
            infrastructure.
        override (float): Alternative minimum pilot which overrides the EVSE
            minimum if the EVSE minimum is less than override.

    Returns:
        List[SessionInfo]: Active sessions with updated minimum charging rate
            for the first control period.
    """
    session_queue = sorted(active_sessions, key=lambda x: x.arrival)
    session_queue = expand_max_min_rates(session_queue)
    rates = np.zeros(len(infrastructure.station_ids))
    for j, session in enumerate(session_queue):
        i = infrastructure.station_ids.index(session.station_id)
        rates[i] = min(infrastructure.min_pilot[i], override)
        if infrastructure_constraints_feasible(rates, infrastructure):
            session.min_rates[0] = max(rates[i], session.min_rates[0])
            session_queue[j] = reconcile_max_and_min(session)
        else:
            rates[i] = 0
            session.min_rates[0] = 0
            session.max_rates[0] = 0
    return session_queue

