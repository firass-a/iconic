"""Quick smoke test for random_weather=True."""
from importlib import import_module

SmartFarmSoSEnv = import_module('02_smart_farm_env').SmartFarmSoSEnv

env = SmartFarmSoSEnv(mode='all', seed=123)
yields = []
for ep in range(2):
    env.reset()
    done = False
    while not done:
        _, _, done, info = env.step({'anfer': 20.0, 'amir': 5.0})
    yields.append(float(info.get('full_state', {}).get('grnwt', 0.0)))
    print(f'  season {ep + 1}: grnwt={yields[-1]:.0f} kg/ha')
env.close()
print('different seasons:', yields[0] != yields[1])
