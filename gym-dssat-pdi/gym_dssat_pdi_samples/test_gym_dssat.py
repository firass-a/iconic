"""
Test script for gym-DSSAT installation.
Run this INSIDE the Docker container:
  python3 test_gym_dssat.py
"""

print("=" * 50)
print("  Testing gym-DSSAT installation")
print("=" * 50)

# Step 1: Import
print("\n[1/4] Importing gym_dssat_pdi...")
try:
    import gym_dssat_pdi
    print("  OK - gym_dssat_pdi imported")
except ImportError as e:
    print(f"  FAIL - {e}")
    print("  Make sure spack environment is activated:")
    print("    source /opt/spack/share/spack/setup-env.sh")
    print("    spack env activate gym-dssat-pdi")
    exit(1)

import gym
import numpy as np

# Step 2: Create environment
print("\n[2/4] Creating environment...")
try:
    env = gym.make(
        'gym_dssat_pdi:GymDssatPdi-v0',
        run_dssat_location='run_dssat',
        mode='fertilization',
        seed=123456,
        random_weather=False,
    )
    print("  OK - environment created")
except Exception as e:
    print(f"  FAIL - {e}")
    exit(1)

# Step 3: Reset and inspect
print("\n[3/4] Resetting environment...")
obs = env.reset()
print(f"  Observation type: {type(obs).__name__}")
if isinstance(obs, dict):
    print(f"  Observation keys: {list(obs.keys())}")
else:
    print(f"  Observation value: {obs}")
print(f"  Action space: {env.action_space}")

# Step 4: Run a short episode
print("\n[4/4] Running 10 random steps...")
total_reward = 0
for day in range(10):
    action_sample = env.action_space.sample()
    # gym-dssat expects an action dictionary; keep compatibility with Dict/Box spaces
    if isinstance(action_sample, dict):
        action = {k: float(v) for k, v in action_sample.items()}
    else:
        action = {'anfer': float(np.array(action_sample).reshape(-1)[0])}
    obs, reward, done, info = env.step(action)
    total_reward += reward
    print(f"  Day {day+1}: reward={reward:.2f}, done={done}")
    if done:
        print("  Episode ended early (crop failure or harvest)")
        break

env.close()

print(f"\n  Total reward over {day+1} days: {total_reward:.2f}")
print("\n" + "=" * 50)
print("  ALL TESTS PASSED - gym-DSSAT is working!")
print("=" * 50)
print("\nYou can now train RL agents on this environment.")
print("Next step: run train_baseline_ppo.py")
