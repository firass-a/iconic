from gym.envs.registration import register

__version__ = '0.0.5'

register(
    id='GymDssatPdi-v0',
    entry_point='gym_dssat_pdi.envs:DssatPdi',
)
