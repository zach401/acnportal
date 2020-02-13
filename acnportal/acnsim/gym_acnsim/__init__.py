from . import reward_functions
from importlib.util import find_spec
if find_spec("gym") is not None:
    from gym.envs.registration import register

    register(
        id='default-acnsim-v0',
        entry_point='acnportal.acnsim.gym_acnsim.envs:DefaultSimEnv',
    )

    register(
        id='rebuilding-acnsim-v0',
        entry_point='acnportal.acnsim.gym_acnsim.envs:RebuildingEnv',
    )
del find_spec
