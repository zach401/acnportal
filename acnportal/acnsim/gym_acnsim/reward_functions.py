import numpy as np
"""
All reward functions have signature

    acnportal.acnsim.gym_acnsim.BaseSimEnv -> Number

That is, reward functions take in an environment instance
and return a number (reward) based on the characteristics of that
environment; namely, the previous state, previous action, and current
state.
"""


def evse_violation(env):
    """
    If a single EVSE constraint was violated by the last schedule, a
    negative reward equal to the magnitude of the violation is added to
    the total reward.

    Raises:
        KeyError: If a station_id in the last schedule is not found in
            the ChargingNetwork.
    """
    violation = 0
    for station_id in env.schedule:
        # Check that each EVSE in the schedule is actually in the network.
        if station_id not in env.interface.station_ids:
            raise KeyError(f'Station {station_id} in schedule but not found '
                           f'in network.')
        # Check that none of the EVSE pilot signal limits are violated.
        evse_is_continuous, evse_allowable_pilots = \
            env.interface.allowable_pilot_signals(station_id)
        if evse_is_continuous:
            min_rate = evse_allowable_pilots[0]
            max_rate = evse_allowable_pilots[1]
            # Add penalty for any pilot signal not in [min_rate, max_rate],
            # except for 0 pilots, which aren't penalized.
            violation += sum([
                max(min_rate - pilot, 0) + max(pilot - max_rate, 0)
                if pilot != 0 else 0
                for pilot in env.schedule[station_id]
            ])
        else:
            # Add penalty for any pilot signal not in the list of allowed
            # pilots, except for 0 pilots, which aren't penalized.
            violation += sum([
                np.abs(np.array(evse_allowable_pilots) - pilot).min()
                if pilot != 0 else 0
                for pilot in env.schedule[station_id]
            ])
    return -violation


def unplugged_ev_violation(env):
    """
    If charge is attempted to be delivered to an EVSE with no EV, or to
    an EVSE with an EV that is done charging, the charging rate is
    subtracted from the reward. This penalty is only applied to the
    schedules for the current iteration.
    """
    violation = 0
    if len(env.schedule) > 0 and len(env.schedule.values()[0]) == 0:
        return violation
    active_evse_ids = env.interface.active_station_ids
    for station_id in env.schedule:
        if station_id not in active_evse_ids:
            violation += abs(env.schedule[station_id][0])
    return -violation


def constraint_violation(env):
    """
    If a network constraint is violated, a negative reward equal to the
    norm of the total constraint violation, times the number of EVSEs,
    is added.
    """
    magnitudes = env.interface.get_constraints().magnitudes
    # Calculate aggregate currents for this charging schedule.
    out_vector = abs(env.interface.constraint_currents(
        np.array([[env.action[i]] for i in range(len(env.action))])))
    # Calculate violation of each individual constraint.
    difference_vector = np.array([0 if out_vector[i] <= magnitudes[i]
                                  else out_vector[i] - magnitudes[i]
                                  for i in range(len(out_vector))])
    # Calculate total constraint violation, scaled by number of EVSEs.
    violation = (np.linalg.norm(difference_vector)
                 * len(env.interface.station_ids))
    return -violation


def soft_charging_reward(env):
    """
    Rewards for charge delivered in the last timestep.
    """
    return np.sum(env.interface.charging_rates
                  - env.prev_interface.charging_rates)


def hard_charging_reward(env):
    """
    Rewards for charge delivered in the last timestep, but only
    if constraint and evse violations are 0.
    """
    return (soft_charging_reward(env)
            if evse_violation(env) == 0 and constraint_violation(env) == 0
            else 0)

