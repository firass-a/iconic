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
TOTAL_TIMESTEPS = 500_000
MODEL_PATH      = '/tmp/pc_ppo_500k.zip'
EVAL_EPISODES   = 3
DSSAT_SEED      = 123
ENABLE_FAULTS   = True


def train():
    print("=" * 78)
    print(f"PC-PPO TRAINING ({TOTAL_TIMESTEPS:,} timesteps) — 3 objectives")
    print("=" * 78)

    env = PCSmartFarmEnv(mode='all', dssat_seed=DSSAT_SEED,
                         enable_faults=ENABLE_FAULTS, rng_seed=0)

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
        ent_coef=0.01,
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


def rollout_under_preference(model, preference, n_episodes=EVAL_EPISODES,
                             eval_seed_base=2000):
    cum_R_vec  = np.zeros(3, dtype=np.float64)
    yields, waters, nitrogens = [], [], []
    actions_log = []

    for ep in range(n_episodes):
        env = PCSmartFarmEnv(
            mode='all',
            dssat_seed=DSSAT_SEED,
            enable_faults=ENABLE_FAULTS,
            preference=preference,
            rng_seed=eval_seed_base + ep,
        )
        obs, info = env.reset(seed=eval_seed_base + ep)

        ep_R = np.zeros(3)
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
    print("PREFERENCE-DIFFERENTIATION TEST (3 objectives)")
    print("=" * 78)
    model = PPO.load(model_path, device='cpu')

    preferences = [
        ('yield-only',      np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        ('water-efficient', np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ('energy-saver',    np.array([0.0, 0.0, 1.0], dtype=np.float32)),
    ]

    results = []
    for name, w in preferences:
        print(f"  evaluating w='{name}' = {w}...")
        results.append((name, w, rollout_under_preference(model, w)))

    print("\n" + "-" * 78)
    print(f"{'preference':<18s}  {'mean N/day':>10s}  {'mean W/day':>10s}  "
          f"{'tot N':>7s}  {'tot W':>7s}  {'yield':>7s}")
    print("-" * 78)
    for name, w, r in results:
        print(f"{name:<18s}  {r['mean_anfer']:>10.2f}  {r['mean_amir']:>10.2f}  "
              f"{r['nitrogen']:>7.0f}  {r['water']:>7.0f}  {r['yield']:>7.0f}")
    print("-" * 78)
    print(f"\n{'preference':<18s}  R_yield   R_wue    R_energy")
    print("-" * 78)
    for name, w, r in results:
        R = r['R_vec']
        print(f"{name:<18s}  {R[0]:>+7.2f}  {R[1]:>+7.2f}   {R[2]:>+7.2f}")
    print("-" * 78)


if __name__ == '__main__':
    try:
        path = train()
        differentiation_test(path)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise
