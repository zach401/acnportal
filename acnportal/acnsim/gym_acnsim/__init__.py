from gym.envs.registration import register
from . import reward_functions

register(
    id='default-acnsim-v0',
    entry_point='acnportal.acnsim.gym_acnsim.envs:DefaultSimEnv',
)

register(
    id='rebuilding-acnsim-v0',
    entry_point='acnportal.acnsim.gym_acnsim.envs:RebuildingEnv',
)