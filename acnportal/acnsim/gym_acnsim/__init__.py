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

# TODO: CustomSimEnv, which allows customization of spaces and rewards.
#  after such an environment is created, it needs to be added to the
#  envs/__init__ file too.
