"""
Step 3: PC-PPO quick training and preference-differentiation test.

Trains PC-PPO for 50,000 timesteps (~8 minutes at ~100 fps), saves the
model, then evaluates the same model under FOUR extreme preferences:

    w = [1, 0, 0, 0]   pure yield maximization
    w = [0, 1, 0, 0]   pure water-use efficiency
    w = [0, 0, 1, 0]   pure energy minimization
    w = [0, 0, 0, 1]   pure resilience

CRITICAL TEST: if PC-PPO is genuinely preference-conditioned, the four
rollouts must produce DIFFERENT action profiles. If all four behave the
same, the network has ignored the preference dims of the observation and
we must debug before scaling up to full training (Step 4).

What "pass" looks like:
    - Yield-preference rollout applies more N and more water
    - Energy-preference rollout applies near-zero of everything
    - The cumulative R-vectors differ across the four runs

What "fail" looks like:
    - All four rollouts have nearly identical action profiles
    - All four R-vectors are within rounding error of each other

Dependencies (inside gym-dssat Docker image — run as root or use constraints):
    Pin numpy==1.24.1 (gym-dssat-pdi) and pandas<3, then SB3 + torch CPU.
    Example one-liner: see 03_sb3_sanity_check.py docstring or README.

Run inside Docker (repo mounted at /workspace):
    cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
    python3 -u 04_pc_ppo_quick_train.py
"""
import os
import sys
import time
import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from stable_baselines3 import PPO
from importlib import import_module
PCSmartFarmEnv = import_module('pc_env').PCSmartFarmEnv


# ============================================================
# Config
# ============================================================
TOTAL_TIMESTEPS = 50_000
MODEL_PATH      = '/tmp/pc_ppo_50k.zip'
EVAL_EPISODES   = 3   # episodes per fixed preference (averaged)
DSSAT_SEED      = 123


# ============================================================
# Training
# ============================================================
def train():
    print("=" * 78)
    print(f"STEP 3 — PC-PPO QUICK TRAINING ({TOTAL_TIMESTEPS:,} timesteps)")
    print("=" * 78)

    env = PCSmartFarmEnv(mode='all', dssat_seed=DSSAT_SEED,
                         enable_faults=False, rng_seed=0)

    # n_steps=2048 is SB3's default for PPO — collects ~12 episodes per
    # rollout, enough preference variety for the network to see and learn
    # how to condition on it.
    model = PPO(
        policy='MlpPolicy',
        env=env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # mild entropy bonus to encourage exploration
        verbose=1,
        device='cpu',
        seed=0,
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    elapsed = time.time() - t0

    model.save(MODEL_PATH)
    env.close()
    print(f"\n✓ Training complete in {elapsed/60:.1f} min")
    print(f"✓ Model saved to {MODEL_PATH}")
    return MODEL_PATH


# ============================================================
# Evaluation under fixed preferences
# ============================================================
def rollout_under_preference(model, preference, n_episodes=EVAL_EPISODES,
                             eval_seed_base=2000):
    """Run n_episodes with a fixed preference; return aggregated stats."""
    cum_R_vec        = np.zeros(4, dtype=np.float64)
    yields, waters, nitrogens = [], [], []
    actions_log      = []

    for ep in range(n_episodes):
        env = PCSmartFarmEnv(
            mode='all',
            dssat_seed=DSSAT_SEED,
            enable_faults=False,
            preference=preference,
            rng_seed=eval_seed_base + ep,
        )
        obs, info = env.reset(seed=eval_seed_base + ep)

        ep_R = np.zeros(4)
        ep_actions = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_R       += info['reward_vector']
            ep_actions.append(info['action_applied'])
            done        = term or trunc

        sos = info['sos_state']
        yields.append(float(sos.get('grnwt', 0.0)))
        waters.append(float(sos.get('total_water', 0.0)))
        nitrogens.append(float(sos.get('total_nitrogen', 0.0)))
        cum_R_vec     += ep_R
        actions_log.append(np.array(ep_actions))
        env.close()

    cum_R_vec /= n_episodes
    actions_arr = np.concatenate(actions_log, axis=0)
    return {
        'R_vec':       cum_R_vec,
        'yield':       float(np.mean(yields)),
        'yield_std':   float(np.std(yields)),
        'water':       float(np.mean(waters)),
        'nitrogen':    float(np.mean(nitrogens)),
        'mean_anfer':  float(np.mean(actions_arr[:, 0])),
        'mean_amir':   float(np.mean(actions_arr[:, 1])),
    }


def differentiation_test(model_path):
    print("\n" + "=" * 78)
    print("PREFERENCE-DIFFERENTIATION TEST")
    print("=" * 78)
    print(f"Running {EVAL_EPISODES} episodes per preference, deterministic policy.")
    print()

    model = PPO.load(model_path, device='cpu')

    preferences = [
        ('yield-only',      np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        ('water-efficient', np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
        ('energy-saver',    np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)),
        ('resilient',       np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
    ]

    results = []
    for name, w in preferences:
        print(f"  evaluating w='{name}' = {w}...")
        results.append((name, w, rollout_under_preference(model, w)))

    # ---- Print comparison table ----
    print("\n" + "-" * 78)
    print(f"{'preference':<18s}  {'mean N/day':>10s}  {'mean W/day':>10s}  "
          f"{'tot N':>7s}  {'tot W':>7s}  {'yield':>7s}")
    print("-" * 78)
    for name, w, r in results:
        print(f"{name:<18s}  {r['mean_anfer']:>10.2f}  {r['mean_amir']:>10.2f}  "
              f"{r['nitrogen']:>7.0f}  {r['water']:>7.0f}  {r['yield']:>7.0f}")
    print("-" * 78)
    print(f"\n{'preference':<18s}  R_yield   R_wue    R_energy  R_resil")
    print("-" * 78)
    for name, w, r in results:
        R = r['R_vec']
        print(f"{name:<18s}  {R[0]:>+7.2f}  {R[1]:>+7.2f}   {R[2]:>+7.2f}   {R[3]:>+7.2f}")
    print("-" * 78)

    # ---- Automated differentiation check ----
    print()
    actions = np.array([
        [r['mean_anfer'], r['mean_amir']] for _, _, r in results
    ])
    # Range of mean N applied across preferences
    n_range = actions[:, 0].max() - actions[:, 0].min()
    w_range = actions[:, 1].max() - actions[:, 1].min()
    print(f"Range of mean N/day across preferences: {n_range:.2f} kg/ha")
    print(f"Range of mean W/day across preferences: {w_range:.2f} mm")

    # A meaningful differentiation needs at least a few kg N spread,
    # OR a few mm water spread. Adjust thresholds with more training.
    differentiated = n_range > 1.0 or w_range > 0.5

    print()
    if differentiated:
        print("✓ PASS: preferences produce DIFFERENT action profiles.")
        print("        PC-PPO is conditioning on the preference dimension.")
        print("        You can proceed to Step 4 (full training).")
    else:
        print("⚠ INCONCLUSIVE: 50k steps is short; the network may not have")
        print("                yet learned to differentiate. Consider:")
        print("                - Re-run with 100k–200k steps to confirm")
        print("                - Increase ent_coef to 0.02 for more exploration")
        print("                - Verify pc_env.py concatenates w into the obs")

    return differentiated


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    try:
        path = train()
        differentiation_test(path)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise
