from gym.envs.registration import register

register(
    id='GymDssatPdi-v0',
    entry_point='gym_dssat_pdi.envs:DssatPdi',
)