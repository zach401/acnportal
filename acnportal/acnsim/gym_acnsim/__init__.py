from importlib.util import find_spec
if find_spec("gym") is not None:
    from gym.envs.registration import register

    register(
        id='custom-acnsim-v0',
        entry_point='acnportal.acnsim.gym_acnsim.envs:CustomSimEnv',
    )

    register(
        id='default-acnsim-v0',
        entry_point='acnportal.acnsim.gym_acnsim.envs:make_default_sim_env',
    )

    register(
        id='rebuilding-acnsim-v0',
        entry_point='acnportal.acnsim.gym_acnsim.envs:RebuildingEnv',
    )

    register(
        id='default-rebuilding-acnsim-v0',
        entry_point='acnportal.acnsim.gym_acnsim.envs:'
                    'make_rebuilding_default_sim_env',
    )
    from .envs import action_spaces, observation, reward_functions
    from . import reward_functions
del find_spec
