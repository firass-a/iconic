"""
Step 1: Baseline policy comparison on gym-DSSAT (multi-objective metrics).
This script does NOT train PPO. It compares simple policies and reports:
  - Yield (maximize)
  - Water use (minimize)
  - Energy cost proxy (minimize)

Energy proxy model (simple):
  daily_energy = sensor_drain + pump_cost + fert_cost
  sensor_drain = 0.5      # fixed SoS overhead per day
  pump_cost    = 0.3*amir # irrigation energy
  fert_cost    = 0.05*anfer

Run inside Docker: python3 /workspace/01_baseline_ppo.py
"""
import gym
import numpy as np


def make_env():
    """Create gym-DSSAT environment in joint fertilization+irrigation mode."""
    env = gym.make(
        'gym_dssat_pdi:GymDssatPdi-v0',
        mode='all',
        seed=123,
    )
    return env


def energy_cost(action):
    """Simple per-day energy proxy based on actuation effort."""
    anfer = float(action.get('anfer', 0.0))
    amir = float(action.get('amir', 0.0))
    sensor_drain = 0.5
    pump_cost = 0.3 * amir
    fert_cost = 0.05 * anfer
    return sensor_drain + pump_cost + fert_cost


def evaluate_policy(env, policy_fn, n_episodes=50, label="Policy"):
    """
    Evaluate a policy function over episodes.

    policy_fn(day, obs) -> action dict {'anfer': float, 'amir': float}
    """
    yields = []
    waters = []
    energies = []
    scalar_rewards = []
    for ep in range(n_episodes):
        obs = env.reset()
        last_valid_obs = obs if isinstance(obs, dict) else {}
        done = False
        day = 0
        total_reward = 0.0
        total_water = 0.0
        total_energy = 0.0
        while not done:
            day += 1
            action = policy_fn(day, obs)
            total_water += float(action.get('amir', 0.0))
            total_energy += energy_cost(action)
            obs, reward, done, info = env.step(action)
            if isinstance(obs, dict):
                last_valid_obs = obs
            if reward is None:
                reward_value = 0.0
            elif isinstance(reward, (list, tuple, np.ndarray)):
                reward_value = float(np.sum(reward))
            else:
                reward_value = float(reward)
            total_reward += reward_value

        grain = float(last_valid_obs.get('grnwt', 0.0))
        yields.append(grain)
        waters.append(total_water)
        energies.append(total_energy)
        scalar_rewards.append(total_reward)

        if (ep + 1) % 10 == 0:
            print(
                f"  Episode {ep+1}/{n_episodes}: "
                f"yield={grain:.0f} kg/ha, water={total_water:.1f} mm, "
                f"energy={total_energy:.1f}, scalar_reward={total_reward:.1f}"
            )

    return {
        'yield': np.array(yields, dtype=np.float32),
        'water': np.array(waters, dtype=np.float32),
        'energy': np.array(energies, dtype=np.float32),
        'scalar_reward': np.array(scalar_rewards, dtype=np.float32),
    }


def random_agent(env, n_episodes=50):
    """Baseline: random nitrogen + random irrigation every day."""
    def policy_fn(day, obs):
        return {
            'anfer': float(np.random.uniform(0, 40)),  # kg N/ha
            'amir': float(np.random.uniform(0, 20)),   # mm
        }
    return evaluate_policy(env, policy_fn, n_episodes=n_episodes, label="Random")


def fixed_schedule_agent(env, n_episodes=50):
    """Baseline: fixed nitrogen schedule + fixed irrigation schedule."""
    def policy_fn(day, obs):
        action = {'anfer': 0.0, 'amir': 0.0}
        if day in [30, 60, 90]:
            action['anfer'] = 35.0
        if day in [25, 45, 65, 85, 105]:
            action['amir'] = 15.0
        return action
    return evaluate_policy(env, policy_fn, n_episodes=n_episodes, label="Fixed")


def threshold_agent(env, n_episodes=50):
    """Baseline: react to stress indicators using simple thresholds."""
    def policy_fn(day, obs):
        n_stress = float(obs.get('nstres', 1.0))  # lower -> more N stress
        water_stress = float(obs.get('swfac', 1.0))  # lower -> more water stress
        action = {'anfer': 0.0, 'amir': 0.0}

        if n_stress < 0.8:
            action['anfer'] = 25.0
        if water_stress < 0.8:
            action['amir'] = 12.0
        return action
    return evaluate_policy(env, policy_fn, n_episodes=n_episodes, label="Threshold")


if __name__ == '__main__':
    env = make_env()
    n_episodes = 50

    print("=" * 60)
    print("BASELINE POLICY COMPARISON (3 OBJECTIVES)")
    print("=" * 60)

    print("\n1. Random Agent:")
    random_metrics = random_agent(env, n_episodes)

    print("\n2. Fixed Schedule Agent:")
    fixed_metrics = fixed_schedule_agent(env, n_episodes)

    print("\n3. Threshold Agent (stress-triggered):")
    threshold_metrics = threshold_agent(env, n_episodes)

    env.close()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    rows = [
        ("Random", random_metrics),
        ("Fixed Schedule", fixed_metrics),
        ("Threshold", threshold_metrics),
    ]
    for name, m in rows:
        print(
            f"  {name:20s}: "
            f"yield_mean={np.mean(m['yield']):8.1f} kg/ha, "
            f"water_mean={np.mean(m['water']):8.1f} mm, "
            f"energy_mean={np.mean(m['energy']):8.1f}, "
            f"reward_mean={np.mean(m['scalar_reward']):8.1f}"
        )

    print("\nInterpretation:")
    print("  - Higher yield_mean is better.")
    print("  - Lower water_mean is better.")
    print("  - Lower energy_mean is better.")
    print("\nUse this as the 3-objective baseline before MORL training.")
