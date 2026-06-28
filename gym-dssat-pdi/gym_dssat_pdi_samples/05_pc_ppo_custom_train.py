"""
Custom PC-PPO training — verbose logging every update.

Env overrides:
    TOTAL_TIMESTEPS, N_STEPS, MODEL_PATH, ENT_COEF, LOG_EVERY, CHECKPOINT_EVERY, LOG_FILE
"""
import os
import sys
import time
from datetime import datetime

import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

try:
    import torch
except ImportError:
    print('Install PyTorch: pip install torch')
    raise

from importlib import import_module
PCSmartFarmEnv = import_module('pc_env').PCSmartFarmEnv
ACTION_LOW = import_module('pc_env').ACTION_LOW
ACTION_HIGH = import_module('pc_env').ACTION_HIGH
OBS_DIM = import_module('pc_env').OBS_DIM
N_PREFERENCE_DIMS = import_module('pc_env').N_PREFERENCE_DIMS

from ppo_agent import PPOAgent

TOTAL_TIMESTEPS = int(os.environ.get('TOTAL_TIMESTEPS', '500000'))
N_STEPS = int(os.environ.get('N_STEPS', '1024'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '64'))
N_EPOCHS = int(os.environ.get('N_EPOCHS', '10'))
ENT_COEF = float(os.environ.get('ENT_COEF', '0.02'))
LOG_EVERY = int(os.environ.get('LOG_EVERY', '5'))
CHECKPOINT_EVERY = int(os.environ.get('CHECKPOINT_EVERY', '50'))
DSSAT_SEED = int(os.environ.get('DSSAT_SEED', '123'))
MODEL_PATH = os.environ.get('MODEL_PATH', '/tmp/pc_ppo_custom.pt')
LOG_FILE = os.environ.get('LOG_FILE', '')
_log_handle = None


def _log_sink():
    global _log_handle
    if not LOG_FILE:
        return None
    if _log_handle is None:
        os.makedirs(os.path.dirname(LOG_FILE) or '.', exist_ok=True)
        _log_handle = open(LOG_FILE, 'a', encoding='utf-8', buffering=1)
    return _log_handle


def log(msg):
    print(msg, flush=True)
    fh = _log_sink()
    if fh:
        fh.write(msg + '\n')
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass


def close_log():
    global _log_handle
    if _log_handle is not None:
        _log_handle.close()
        _log_handle = None


def train():
    log('=' * 90)
    log('CUSTOM PC-PPO — verbose training log')
    log('=' * 90)
    log(f'  started     : {datetime.now().isoformat(timespec="seconds")}')
    log(f'  obs_dim     : {OBS_DIM}  (11 state + {N_PREFERENCE_DIMS} preference)')
    log(f'  n_steps     : {N_STEPS}  batch={BATCH_SIZE}  epochs={N_EPOCHS}  ent_coef={ENT_COEF}')
    log(f'  timesteps   : {TOTAL_TIMESTEPS:,}  updates={TOTAL_TIMESTEPS // N_STEPS}')
    log(f'  model_path  : {MODEL_PATH}')
    if LOG_FILE:
        log(f'  log_file    : {LOG_FILE}')
    log('=' * 90)

    os.makedirs(os.path.dirname(MODEL_PATH) or '.', exist_ok=True)
    _log_sink()

    env = PCSmartFarmEnv(mode='all', dssat_seed=DSSAT_SEED, rng_seed=0)

    agent = PPOAgent(
        obs_dim=OBS_DIM,
        act_dim=2,
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        lr=3e-4,
        vf_coef=0.5,
        ent_coef=ENT_COEF,
        max_grad_norm=0.5,
        device='cpu',
        seed=0,
    )

    n_updates = TOTAL_TIMESTEPS // N_STEPS
    t0 = time.time()
    timestep = 0
    total_seasons = 0
    best_r_mean = -float('inf')

    for update in range(1, n_updates + 1):
        t_rollout = time.time()
        buffer, stats = agent.collect_rollout(env, n_pref_dims=N_PREFERENCE_DIMS)
        rollout_sec = time.time() - t_rollout

        t_update = time.time()
        metrics = agent.update(buffer)
        update_sec = time.time() - t_update

        timestep += N_STEPS
        total_seasons += stats['n_episodes']
        mean_reward = float(buffer.rewards[:buffer.ptr].mean())
        if mean_reward > best_r_mean:
            best_r_mean = mean_reward
            agent.save(MODEL_PATH.replace('.pt', '_best.pt'))

        if update % LOG_EVERY == 0 or update == 1 or update == n_updates:
            elapsed = time.time() - t0
            fps = timestep / max(elapsed, 1e-6)
            eta_min = (n_updates - update) * (elapsed / update) / 60
            b = buffer
            log(
                f'--- update {update:4d}/{n_updates} | steps {timestep:7d} | '
                f'fps {fps:5.1f} | eta {eta_min:5.1f}m | '
                f'rollout {rollout_sec:5.1f}s | ppo {update_sec:4.1f}s ---'
            )
            log(
                f'    reward  r_mean={mean_reward:+.4f}  return={b.returns[:b.ptr].mean():+.4f}  '
                f'adv_std={b.advantages[:b.ptr].std():.4f}'
            )
            log(
                f'    actions mean_N={b.actions[:b.ptr,0].mean():6.2f}  max_N={b.actions[:b.ptr,0].max():6.2f}  '
                f'mean_W={b.actions[:b.ptr,1].mean():5.2f}  max_W={b.actions[:b.ptr,1].max():5.2f}'
            )
            log(
                f'    R_vec   yield={stats["mean_R_yield"]:+.4f}  water={stats["mean_R_water"]:+.4f}  '
                f'fert={stats["mean_R_fert"]:+.4f}'
            )
            log(
                f'    pref_w  [{stats["mean_pref"][0]:.3f}, {stats["mean_pref"][1]:.3f}, '
                f'{stats["mean_pref"][2]:.3f}]  seasons_rollout={stats["n_episodes"]}  '
                f'total_seasons={total_seasons}'
            )
            if stats['harvest_yields']:
                log(f'    harvest yields (last): {[int(y) for y in stats["harvest_yields"]]} kg/ha')
            if stats['episode_returns']:
                log(f'    episode returns (last): {[round(r,2) for r in stats["episode_returns"]]}')
            log(
                f'    losses  L_pi={metrics["policy_loss"]:+.4f}  L_v={metrics["value_loss"]:.4f}  '
                f'H={metrics["entropy"]:.4f}  L_tot={metrics["total_loss"]:+.4f}'
            )

        if CHECKPOINT_EVERY > 0 and update % CHECKPOINT_EVERY == 0:
            ckpt = MODEL_PATH.replace('.pt', f'_ckpt_{update}.pt')
            agent.save(ckpt)
            log(f'    checkpoint → {ckpt}')

    agent.save(MODEL_PATH)
    env.close()
    log('')
    log(f'Model saved     → {MODEL_PATH}')
    log(f'Best model      → {MODEL_PATH.replace(".pt", "_best.pt")}  (r_mean={best_r_mean:+.4f})')
    log(f'Training time   : {(time.time()-t0)/60:.1f} min')
    log(f'Total seasons  : {total_seasons:,}')
    log(f'Finished        : {datetime.now().isoformat(timespec="seconds")}')
    return agent


def eval_preferences(agent):
    """Quick 3-corner sanity check — use 06_pc_ppo_eval.py for thesis-grade eval."""
    log('')
    log('=' * 90)
    log('QUICK EVAL — 3 corners x 3 episodes (run .\\run_eval.ps1 for full eval)')
    log('=' * 90)

    prefs = [
        ('yield',       [1.0, 0.0, 0.0]),
        ('water',       [0.0, 1.0, 0.0]),
        ('fertilizer',  [0.0, 0.0, 1.0]),
    ]

    for name, w in prefs:
        yields, waters, nitrogens = [], [], []
        mean_n, mean_w, max_n, max_w = [], [], [], []

        for ep in range(3):
            # Vary dssat_seed per ep so each episode draws a different weather year
            # (matches the strong eval; replays of seed=123 give std=0).
            env = PCSmartFarmEnv(
                mode='all', dssat_seed=DSSAT_SEED + ep,
                preference=np.array(w, dtype=np.float32),
                rng_seed=1000 + ep,
                random_weather=True,
            )
            obs, _ = env.reset(seed=1000 + ep)
            actions = []
            cum_r = np.zeros(3)
            done = False
            while not done:
                a = agent.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                actions.append(a)
                cum_r += info['reward_vector']
                done = term or trunc

            actions = np.array(actions)
            fs = info.get('full_state', {})
            yields.append(float(fs.get('grnwt', 0.0)))
            waters.append(float(info['totals']['water']))
            nitrogens.append(float(info['totals']['nitrogen']))
            mean_n.append(float(actions[:, 0].mean()))
            mean_w.append(float(actions[:, 1].mean()))
            max_n.append(float(actions[:, 0].max()))
            max_w.append(float(actions[:, 1].max()))
            env.close()

        log(
            f'  {name:<12s}  w={w}  '
            f'mean_N={np.mean(mean_n):5.1f}  max_N={np.mean(max_n):5.1f}  '
            f'mean_W={np.mean(mean_w):5.1f}  max_W={np.mean(max_w):5.1f}  '
            f'yield={np.mean(yields):.0f} kg/ha  '
            f'tot_N={np.mean(nitrogens):.0f}  tot_W={np.mean(waters):.0f}  '
            f'cum_R={cum_r.round(2)}'
        )

    log('')
    log('For full eval (20+ eps, preference grid, baselines, CSV):')
    log('  .\\run_eval.ps1')


if __name__ == '__main__':
    try:
        agent = train()
        eval_preferences(agent)
    except Exception as e:
        log(f'\nERROR: {type(e).__name__}: {e}')
        raise
    finally:
        close_log()
