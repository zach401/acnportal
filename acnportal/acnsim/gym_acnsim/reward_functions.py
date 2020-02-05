import numpy as np
"""
All reward functions have signature

    acnportal.acnsim.gym_acnsim.BaseSimEnv -> Number

That is, reward functions take in an environment instance
and return a number (reward) based on the characteristics of that
environment.
"""


def evse_violation(env):
    """
    If a single EVSE constraint is violated, a negative reward equal to
    the magnitude of the violation is added to the total reward.
    """
    violation = 0
    for i in range(len(env.interface.evse_list)):
        curr_evse = env.interface.evse_list[i]
        if env.action[i] < curr_evse.min_rate:
            violation -= curr_evse.min_rate - env.action[i]
        if env.action[i] > curr_evse.max_rate:
            violation -= env.action[i] - curr_evse.max_rate
    return violation


def unplugged_ev_violation(env):
    """
    If charge is attempted to be delivered to an evse with no ev, this
    rate is subtracted from the reward.
    """
    violation = 0
    for i in range(len(env.interface.evse_list)):
        if env.interface.evse_list[i].ev is None:
            violation -= abs(env.action[i])
    return violation


def constraint_violation(env):
    """
    If a network constraint is violated, a negative reward equal to the
    abs of the constraint violation, times the number of EVSEs, is
    added.
    """
    _, magnitudes = env.interface.network_constraints
    # Calculate aggregate currents for this charging schedule
    out_vector = abs(
        env.interface.constraint_currents(
            np.array([[env.action[i]] for i in range(len(env.action))])
        )
    )
    # Calculate violation of each individual constraint violation
    difference_vector = np.array(
        [0 if out_vector[i] <= magnitudes[i]
         else out_vector[i] - magnitudes[i]
         for i in range(len(out_vector))]
    )
    # Calculate total constraint violation, scaled by number of 
    # EVSEs
    violation = np.linalg.norm(difference_vector) * env.num_evses * (-1)
    return violation


def soft_charging_reward(env):
    """
    Rewards for charge delivered in the last timestep.
    """
    # TODO: currently only takes last actual charging rates, should take all
    #  charging rates caused by this schedule
    # TODO: function for this that doesn't require private variable access
    # TODO: The test for this function should include a case where EVs just
    #  plugged in and a case where an EV just left but was charging in the
    #  last period.

    charging_rates = env.interface.last_actual_charging_rate
    charging_reward = np.sum(
        env.interface._simulator.charging_rates[:, env.interface.current_time-1]
    )
    return charging_reward


def hard_charging_reward(env):
    """
    Rewards for charge delivered in the last timestep, but only
    if constraint and evse violations are 0.
    """
    if evse_violation(env) != 0 or constraint_violation(env) != 0:
        return 0
    else:
        return soft_charging_reward(env)


# TODO: there seems to be a problem with plugging in 2 evs at the same
#  timestep having inaccurate len actives
# TODO: add options to toggle which rewards are included in the sum
