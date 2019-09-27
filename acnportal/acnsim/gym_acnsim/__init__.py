from gym.envs.registration import register

register(
    id='acnsim-prototype-v0',
    entry_point='acnportal.acnsim.gym_acnsim.envs:SimPrototypeEnv',
)

register(
    id='continuous-acnsim-prototype-v0',
    entry_point='acnportal.acnsim.gym_acnsim.envs:ContSimPrototypeEnv',
)