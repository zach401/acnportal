from gym.envs.registration import register

register(
    id='acnsim-prototype',
    entry_point='gym_acnsim.envs:SimPrototypeEnv',
)