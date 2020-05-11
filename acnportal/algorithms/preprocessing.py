from typing import List
from copy import deepcopy
import numpy as np

from acnportal.acnsim.interface import SessionInfo, InfrastructureInfo
from .upper_bound_estimator import UpperBoundEstimatorBase
from .utils import infrastructure_constraints_feasible


def least_laxity_first(evs, iface):
    """ Sort EVs by laxity in increasing order.

    Laxity is a measure of the charging flexibility of an EV. Here we define laxity as:
        LAX_i(t) = (departure_i - t) - (remaining_demand_i(t) / max_rate_i)

    Args:
        evs (List[EV]): List of EVs to be sorted.
        iface (Interface): Interface object.

    Returns:
        List[EV]: List of EVs sorted by laxity in increasing order.
    """

    def laxity(ev):
        """ Calculate laxity of the EV.

        Args:
            ev (EV): An EV object.

        Returns:
            float: The laxity of the EV.
        """
        lax = (ev.departure - iface.current_time) - \
              (iface.remaining_amp_periods(ev) / iface.max_pilot_signal(ev.station_id))
        return lax

    return sorted(evs, key=lambda x: laxity(x))


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


def reconcile_max_and_min(session: SessionInfo, choose_min=True):
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
    new_sess = deepcopy(session)
    mask = new_sess.max_rates < new_sess.min_rates
    if choose_min:
        new_sess.max_rates[mask] = new_sess.min_rates[mask]
    else:
        new_sess.min_rates[mask] = new_sess.max_rates[mask]
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
                                interface,
                                override=float('inf'),):
    """ Modify active_sessions so that min_rates[0] is equal to the greater of
        the session minimum rate and the EVSE minimum pilot.

    Args:
        active_sessions (List[SessionInfo]): List of SessionInfo objects for
            all active charging sessions.
        infrastructure (InfrastructureInfo): Description of the charging
            infrastructure.
        period (int): Length of each time period in minutes.
        override (float): Alternative minimum pilot which overrides the EVSE
            minimum if the EVSE minimum is less than override.

    Returns:
        List[SessionInfo]: Active sessions with updated minimum charging rate
            for the first control period.
    """
    new_sessions = []
    session_queue = least_laxity_first(active_sessions, interface)
    # session_queue = sorted(active_sessions, key=lambda x: x.remaining_time)
    session_queue = expand_max_min_rates(session_queue)
    rates = np.zeros(len(infrastructure.station_ids))
    for j, session in enumerate(session_queue):
        i = infrastructure.station_ids.index(session.station_id)
        rates[i] = min(infrastructure.min_pilot[i], override)
        # energy_ub = (session.remaining_demand * 1000 * 60 /
        #              infrastructure.voltages[i] / 12)
        # if rates[i] < energy_ub and \
        if infrastructure_constraints_feasible(rates, infrastructure):
            session.min_rates[0] = max(rates[i], session.min_rates[0])
            session_queue[j] = reconcile_max_and_min(session)
            new_sessions.append(session_queue[j])
        else:
            # If an EV cannot be charged at the minimum rate, it should be
            # removed from the problem. So it is not appended to the problem.
            rates[i] = 0
            # session.min_rates = 0
            # session.max_rates = 0
    return session_queue


# def inc_remaining_energy_to_min_allowable(active_sessions: List[SessionInfo],
#                                           infrastructure: InfrastructureInfo,
#                                           period):
#     """ Modify active_sessions so that remaining_energy is greater than or
#         equal to the energy delivered in one period charging at the
#         session's minimum rate.
#
#         Using this preprocessor ensures that EVs will not leave unsatisfied
#         if there is sufficient capacity in the system.
#
#         Note this could mean that the energy delivered to an EV could exceed
#         100% of its requested energy. This should be considered during
#         analysis.
#
#     Args:
#         active_sessions (List[SessionInfo]): List of SessionInfo objects for
#             all active charging sessions.
#         infrastructure (InfrastructureInfo): Description of the charging
#             infrastructure.
#
#     Returns:
#         List[SessionInfo]: Active sessions with updated minimum charging rate
#             for the first control period.
#     """
#     time_interval_length = period / 60
#
#     def minimum_first_period_charge(session):
#         i = infrastructure.get_station_index(session.station_id)
#         return (session.min_rates[0] * infrastructure.voltages[i] / 1000 *
#                 time_interval_length)
#
#     for session in active_sessions:
#         min_charge = minimum_first_period_charge(session)
#         diff = min_charge - session.remaining_demand
#         if diff > 0:
#             session.requested_energy += diff
#     return active_sessions