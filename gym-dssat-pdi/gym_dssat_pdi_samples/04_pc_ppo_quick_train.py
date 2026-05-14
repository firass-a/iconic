"""
PC-PPO quick training and preference-differentiation test.
3-objective version: R_yield, R_wue, R_energy.

Trains PC-PPO for TOTAL_TIMESTEPS (default 500k — quick differentiation
test only), saves the model, then evaluates it under FOUR preference vectors:

    w = [1, 0, 0]   pure yield maximization
    w = [0, 1, 0]   pure water-use efficiency
    w = [0, 0, 1]   pure energy minimization
    w = [0.4, 0.4, 0.2]   balanced trade-off

CRITICAL TEST — what "pass" looks like:
    yield-only    → more N applied, more water, highest R_yield
    water-efficient → less irrigation, higher R_wue, lower total_water
    energy-saver  → less irrigation AND less N (both drive energy cost),
                    highest R_energy
    balanced      → intermediate on all three

What "fail" looks like:
    All four rollouts produce nearly identical action profiles.
    All four R-vectors are within rounding error of each other.
    → Network has ignored the preference dims. Debug pc_env.py first.

Changes from previous version:
    - Preference vectors reduced from 4-dim to 3-dim (resilience removed)
    - ep_R initialised as np.zeros(3) not np.zeros(4)
    - Print headers updated to match 3 objectives
    - 'resilient' preference replaced with 'balanced' [0.4, 0.4, 0.2]
    - Differentiation check extended to cover energy objective
    - TOTAL_TIMESTEPS comment clarifies quick-test vs full-training budget
    - Per-reward cumulative breakdown added to evaluation output

Full training budget recommendation:
    500k  — differentiation test (this script default)
    2M    — policy starts showing meaningful agronomic behaviour
    5M    — results suitable for thesis comparison table

Dependencies: run inside gym-DSSAT Docker image.
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
# 500k = quick differentiation test (~8–12 min at ~100 fps).
# For full training increase to 2_000_000 or 5_000_000.
TOTAL_TIMESTEPS = 500_000
MODEL_PATH      = '/tmp/pc_ppo_500k.zip'
EVAL_EPISODES   = 3       # episodes per fixed preference (averaged)
DSSAT_SEED      = 123


# ============================================================
# Training
# ============================================================
def train():
    print("=" * 78)
    print(f"PC-PPO QUICK TRAINING  —  {TOTAL_TIMESTEPS:,} timesteps")
    print(f"Objectives: R_yield | R_wue | R_energy  (3-dim preference simplex)")
    print("=" * 78)

    env = PCSmartFarmEnv(
        mode='all',
        dssat_seed=DSSAT_SEED,
        enable_faults=True,
        rng_seed=0,
    )

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
        ent_coef=0.01,       # mild entropy bonus to encourage exploration
        verbose=1,
        device='cpu',
        seed=0,
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    elapsed = time.time() - t0

    model.save(MODEL_PATH)
    env.close()

    print(f"\n✓ Training complete in {elapsed / 60:.1f} min")
    print(f"✓ Model saved to {MODEL_PATH}")
    return MODEL_PATH


# ============================================================
# Evaluation under a fixed preference
# ============================================================
def rollout_under_preference(model, preference, n_episodes=EVAL_EPISODES,
                             eval_seed_base=2000):
    """Run n_episodes with a fixed preference; return aggregated stats."""
    # 3-dim reward vector — must match N_PREFERENCE_DIMS in pc_env.py
    cum_R_vec = np.zeros(3, dtype=np.float64)

    yields, waters, nitrogens = [], [], []
    actions_log = []

    for ep in range(n_episodes):
        env = PCSmartFarmEnv(
            mode='all',
            dssat_seed=DSSAT_SEED,
            enable_faults=True,
            preference=preference,
            rng_seed=eval_seed_base + ep,
        )
        obs, info = env.reset(seed=eval_seed_base + ep)

        ep_R     = np.zeros(3)     # 3-dim — matches 3-objective reward vector
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
        cum_R_vec    += ep_R
        actions_log.append(np.array(ep_actions))
        env.close()

    cum_R_vec  /= n_episodes
    actions_arr = np.concatenate(actions_log, axis=0)

    return {
        'R_vec':      cum_R_vec,                           # shape (3,)
        'yield':      float(np.mean(yields)),
        'yield_std':  float(np.std(yields)),
        'water':      float(np.mean(waters)),
        'nitrogen':   float(np.mean(nitrogens)),
        'mean_anfer': float(np.mean(actions_arr[:, 0])),   # mean daily N kg/ha
        'mean_amir':  float(np.mean(actions_arr[:, 1])),   # mean daily water mm
    }


# ============================================================
# Differentiation test
# ============================================================
def differentiation_test(model_path):
    print("\n" + "=" * 78)
    print("PREFERENCE-DIFFERENTIATION TEST  —  3 objectives")
    print("=" * 78)
    print(f"Running {EVAL_EPISODES} episodes per preference, deterministic policy.")
    print()

    model = PPO.load(model_path, device='cpu')

    # 3-dim preference vectors — [w_yield, w_wue, w_energy]
    preferences = [
        ('yield-only',      np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        ('water-efficient', np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ('energy-saver',    np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ('balanced',        np.array([0.4, 0.4, 0.2], dtype=np.float32)),
    ]

    results = []
    for name, w in preferences:
        print(f"  evaluating  w='{name}'  =  {w} ...")
        results.append((name, w, rollout_under_preference(model, w)))

    # ---- Action profile table ----
    print("\n" + "-" * 78)
    print(f"{'preference':<18}  {'mean N/day':>10}  {'mean W/day':>10}  "
          f"{'tot N':>7}  {'tot W':>7}  {'yield':>7}")
    print("-" * 78)
    for name, w, r in results:
        print(f"{name:<18}  {r['mean_anfer']:>10.2f}  {r['mean_amir']:>10.2f}  "
              f"{r['nitrogen']:>7.0f}  {r['water']:>7.0f}  {r['yield']:>7.0f}")
    print("-" * 78)

    # ---- Per-reward cumulative table (3 objectives) ----
    print(f"\n{'preference':<18}  {'R_yield':>8}  {'R_wue':>8}  {'R_energy':>9}")
    print("-" * 78)
    for name, w, r in results:
        R = r['R_vec']
        print(f"{name:<18}  {R[0]:>+8.3f}  {R[1]:>+8.3f}  {R[2]:>+9.3f}")
    print("-" * 78)

    # ---- Automated differentiation check ----
    print()
    actions = np.array([
        [r['mean_anfer'], r['mean_amir']] for _, _, r in results
    ])
    R_vecs = np.array([r['R_vec'] for _, _, r in results])

    n_range      = actions[:, 0].max() - actions[:, 0].min()
    w_range      = actions[:, 1].max() - actions[:, 1].min()
    energy_range = R_vecs[:, 2].max() - R_vecs[:, 2].min()
    yield_range  = R_vecs[:, 0].max() - R_vecs[:, 0].min()
    wue_range    = R_vecs[:, 1].max() - R_vecs[:, 1].min()

    print(f"Action differentiation:")
    print(f"  N/day range across preferences  : {n_range:.2f} kg/ha")
    print(f"  W/day range across preferences  : {w_range:.2f} mm")
    print(f"Reward differentiation:")
    print(f"  R_yield range across preferences: {yield_range:.3f}")
    print(f"  R_wue   range across preferences: {wue_range:.3f}")
    print(f"  R_energy range across preferences: {energy_range:.3f}")

    # Pass if actions OR rewards are meaningfully differentiated.
    # At 500k steps the network may not fully differentiate yet —
    # reward differentiation is a looser signal than action differentiation.
    action_diff = n_range > 1.0 or w_range > 0.5
    reward_diff = yield_range > 0.05 or wue_range > 0.05 or energy_range > 0.05

    differentiated = action_diff or reward_diff

    print()
    if action_diff and reward_diff:
        print("✓ PASS (strong): both action profiles AND reward vectors differ.")
        print("  PC-PPO is conditioning on the preference dimension.")
        print("  Proceed to full training (2M–5M timesteps).")
    elif differentiated:
        print("~ PASS (weak): partial differentiation detected.")
        print("  One of actions or rewards differs but not both.")
        print("  Consider re-running with 200k–500k more steps before full training.")
    else:
        print("⚠ INCONCLUSIVE / FAIL: 500k steps may be insufficient, OR")
        print("  pc_env.py is not concatenating w into the observation.")
        print("  Checklist:")
        print("    1. Confirm obs[11:14] = w in pc_env.py _encode_observation()")
        print("    2. Try ent_coef=0.02 for more exploration")
        print("    3. Re-run with 200k more steps")
        print("    4. Check that N_PREFERENCE_DIMS=3 in pc_env.py")

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
