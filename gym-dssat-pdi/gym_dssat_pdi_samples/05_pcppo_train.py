"""
PC-PPO training — Preference-Conditioned PPO for multi-objective smart farming.

Each episode samples w ~ Dirichlet(1,1,1). The 14-dim observation includes w,
so the policy learns to behave differently per preference — one model covers
the full yield / N-efficiency / water-efficiency Pareto front.

Produces in /workspace/:
  pcppo_episode_log.csv      — one row per training episode (includes w)
  pcppo_training_metrics.csv — one row per PPO rollout
  pcppo_eval_results.csv     — 4 corners × EVAL_EPISODES episodes
  pcppo_plot_training.png    — training curves
  pcppo_plot_losses.png      — PPO loss curves

Run inside Docker:
    cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
    /opt/gym_dssat_pdi/bin/python3 -u 05_pcppo_train.py
"""
import csv
import os
import sys
import time
import numpy as np
from collections import deque

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from importlib import import_module
PCPPOEnv = import_module('pc_env_pcppo').PCPPOEnv


# ================================================================== #
# Config                                                               #
# ================================================================== #
TOTAL_TIMESTEPS = 1_000_000
N_STEPS         = 4096
BATCH_SIZE      = 64
N_EPOCHS        = 10
LEARNING_RATE   = 3e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2
ENT_COEF        = 0.01
MAX_GRAD_NORM   = 0.5

MODEL_PATH      = '/tmp/pcppo_1M'
VECNORM_PATH    = '/tmp/pcppo_1M_vecnorm.pkl'
EPISODE_LOG     = '/workspace/pcppo_episode_log.csv'
METRICS_LOG     = '/workspace/pcppo_training_metrics.csv'
EVAL_LOG        = '/workspace/pcppo_eval_results.csv'
PLOT_TRAINING   = '/workspace/pcppo_plot_training.png'
PLOT_LOSSES     = '/workspace/pcppo_plot_losses.png'

EVAL_EPISODES   = 5
DSSAT_SEED      = 123
ENABLE_FAULTS   = False
ROLLING_WINDOW  = 100

NORM_OBS        = True
NORM_REWARD     = True
CLIP_OBS        = 5.0
CLIP_REWARD     = 10.0

# Evaluation corners: [w_yield, w_ane, w_water]
EVAL_CORNERS = {
    'yield':    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    'n_eff':    np.array([0.0, 1.0, 0.0], dtype=np.float32),
    'water':    np.array([0.0, 0.0, 1.0], dtype=np.float32),
    'balanced': np.array([1/3, 1/3, 1/3], dtype=np.float32),
}


# ================================================================== #
# Env factory                                                          #
# ================================================================== #
def _make_env(dssat_seed):
    def _init():
        return PCPPOEnv(
            mode='all',
            dssat_seed=dssat_seed,
            run_dssat_location='run_dssat',
            enable_faults=ENABLE_FAULTS,
        )
    return _init


# ================================================================== #
# Config print                                                         #
# ================================================================== #
def _print_config():
    W = 78
    print("=" * W)
    print("PC-PPO — PREFERENCE-CONDITIONED PPO")
    print("=" * W)
    print(f"  {'Objectives (reward vector)'}")
    print(f"    {'[0] R_yield':<24} grain yield vs baseline 7620 kg/ha")
    print(f"    {'[1] R_ane':<24} N efficiency (yield/N), floor at 5000 kg/ha")
    print(f"    {'[2] R_water_eff':<24} water efficiency (less = better, target 400 mm)")
    print()
    print(f"  {'Preference sampling'}")
    print(f"    {'distribution':<24} Dirichlet([1, 1, 1]) — uniform simplex")
    print(f"    {'obs dims':<24} 11 crop+SoS + 3 preference = 14 total")
    print()
    print(f"  {'PPO Hyperparameters'}")
    print(f"    {'total_timesteps':<24} {TOTAL_TIMESTEPS:,}")
    print(f"    {'n_steps':<24} {N_STEPS}")
    print(f"    {'batch_size':<24} {BATCH_SIZE}")
    print(f"    {'n_epochs':<24} {N_EPOCHS}")
    print(f"    {'learning_rate':<24} {LEARNING_RATE}")
    print(f"    {'gamma':<24} {GAMMA}")
    print(f"    {'clip_range':<24} {CLIP_RANGE}")
    print(f"    {'ent_coef':<24} {ENT_COEF}")
    print()
    print(f"  {'Normalisation'}")
    print(f"    {'norm_obs / norm_reward':<24} {NORM_OBS} / {NORM_REWARD}")
    print(f"    {'clip_obs / clip_reward':<24} {CLIP_OBS} / {CLIP_REWARD}")
    print("=" * W)
    print()


# ================================================================== #
# Training callback                                                    #
# ================================================================== #
class PCPPOLogger(BaseCallback):

    EP_FIELDS = [
        'timestep', 'episode', 'ep_length', 'cum_reward',
        'w_yield', 'w_neff', 'w_water',
        'R_yield', 'R_ane', 'R_water_eff', 'R_seasonal',
        'R_daily_mean',
        'yield_kg_ha', 'total_N_kg_ha', 'total_W_mm', 'total_rain_mm',
    ]

    MT_FIELDS = [
        'timestep', 'n_updates', 'ep_rew_mean', 'ep_len_mean',
        'policy_loss', 'value_loss', 'entropy_loss',
        'explained_variance', 'clip_fraction', 'approx_kl', 'fps',
    ]

    def __init__(self, ep_log, mt_log, verbose=1):
        super().__init__(verbose)
        self.ep_log = ep_log
        self.mt_log = mt_log

        self._ep_count      = 0
        self._ep_cum_reward = 0.0
        self._ep_length     = 0
        self._ep_daily_r    = []
        self._recent        = deque(maxlen=ROLLING_WINDOW)

        self._ep_file = self._mt_file = None
        self._ep_writer = self._mt_writer = None

    def _on_training_start(self):
        self._ep_file   = open(self.ep_log, 'w', newline='', buffering=1)
        self._ep_writer = csv.DictWriter(self._ep_file, fieldnames=self.EP_FIELDS)
        self._ep_writer.writeheader()

        self._mt_file   = open(self.mt_log, 'w', newline='', buffering=1)
        self._mt_writer = csv.DictWriter(self._mt_file, fieldnames=self.MT_FIELDS)
        self._mt_writer.writeheader()

        print(f"  Episode log → {self.ep_log}")
        print(f"  Metrics log → {self.mt_log}\n")
        print(f"  {'ep':>5}  {'step':>8}  {'len':>4}  {'cum_R':>7}  "
              f"{'w':>14}  {'yield':>6}  {'N':>5}  {'W':>5}  "
              f"{'R_yld':>7}  {'R_ane':>7}  {'R_wef':>7}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*4}  {'-'*7}  "
              f"{'-'*14}  {'-'*6}  {'-'*5}  {'-'*5}  "
              f"{'-'*7}  {'-'*7}  {'-'*7}")

    def _on_step(self):
        dones   = self.locals.get('dones',   [False])
        infos   = self.locals.get('infos',   [{}])
        rewards = self.locals.get('rewards', [0.0])

        self._ep_cum_reward += float(rewards[0])
        self._ep_length     += 1
        self._ep_daily_r.append(float(infos[0].get('R_daily', 0.0)))

        if dones[0]:
            self._ep_count += 1
            info = infos[0]
            sos  = info.get('sos_state', {})
            rv   = info.get('reward_vec', np.zeros(3))
            w    = info.get('preference_w', np.ones(3)/3)

            row = {
                'timestep':      self.num_timesteps,
                'episode':       self._ep_count,
                'ep_length':     self._ep_length,
                'cum_reward':    round(self._ep_cum_reward, 6),
                'w_yield':       round(float(w[0]), 4),
                'w_neff':        round(float(w[1]), 4),
                'w_water':       round(float(w[2]), 4),
                'R_yield':       round(float(rv[0]), 6),
                'R_ane':         round(float(rv[1]), 6),
                'R_water_eff':   round(float(rv[2]), 6),
                'R_seasonal':    round(float(info.get('R_seasonal', 0.0)), 6),
                'R_daily_mean':  round(float(np.mean(self._ep_daily_r)), 8),
                'yield_kg_ha':   round(float(sos.get('grnwt',          0.0)), 1),
                'total_N_kg_ha': round(float(sos.get('total_nitrogen',  0.0)), 1),
                'total_W_mm':    round(float(sos.get('total_water',     0.0)), 1),
                'total_rain_mm': round(float(sos.get('total_rain',      0.0)), 1),
            }
            self._ep_writer.writerow(row)
            self._ep_file.flush()
            self._recent.append(row)

            if self.verbose >= 1:
                w_str = f"[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]"
                print(
                    f"  {self._ep_count:5d}  {self.num_timesteps:8d}  "
                    f"{row['ep_length']:4d}  {row['cum_reward']:+7.3f}  "
                    f"{w_str:>14}  "
                    f"{row['yield_kg_ha']:6.0f}  {row['total_N_kg_ha']:5.0f}  "
                    f"{row['total_W_mm']:5.0f}  "
                    f"{row['R_yield']:+7.4f}  {row['R_ane']:+7.4f}  "
                    f"{row['R_water_eff']:+7.4f}"
                )

            if self._ep_count % ROLLING_WINDOW == 0 and len(self._recent) > 0:
                def rm(k): return np.mean([r[k] for r in self._recent])
                print(
                    f"\n  ── rolling mean (last {len(self._recent)} ep) ──  "
                    f"cum_R={rm('cum_reward'):+.3f}  "
                    f"yield={rm('yield_kg_ha'):.0f} kg/ha  "
                    f"N={rm('total_N_kg_ha'):.0f}  "
                    f"W={rm('total_W_mm'):.0f} mm  "
                    f"R_yld={rm('R_yield'):+.3f}  "
                    f"R_ane={rm('R_ane'):+.3f}  "
                    f"R_wef={rm('R_water_eff'):+.3f}\n"
                )

            self._ep_cum_reward = 0.0
            self._ep_length     = 0
            self._ep_daily_r    = []

        return True

    def _on_rollout_end(self):
        lv  = self.model.logger.name_to_value
        row = {
            'timestep':           self.num_timesteps,
            'n_updates':          int(lv.get('train/n_updates', 0)),
            'ep_rew_mean':        round(float(lv.get('rollout/ep_rew_mean',    0.0)), 6),
            'ep_len_mean':        round(float(lv.get('rollout/ep_len_mean',    0.0)), 2),
            'policy_loss':        round(float(lv.get('train/policy_gradient_loss', 0.0)), 8),
            'value_loss':         round(float(lv.get('train/value_loss',       0.0)), 6),
            'entropy_loss':       round(float(lv.get('train/entropy_loss',     0.0)), 6),
            'explained_variance': round(float(lv.get('train/explained_variance', 0.0)), 6),
            'clip_fraction':      round(float(lv.get('train/clip_fraction',    0.0)), 6),
            'approx_kl':          round(float(lv.get('train/approx_kl',       0.0)), 8),
            'fps':                int(lv.get('time/fps', 0)),
        }
        self._mt_writer.writerow(row)
        self._mt_file.flush()

    def _on_training_end(self):
        for f in (self._ep_file, self._mt_file):
            if f: f.close()
        print(f"\n✓ Episode log  → {self.ep_log}")
        print(f"✓ Metrics log  → {self.mt_log}")


# ================================================================== #
# Training                                                             #
# ================================================================== #
def train():
    _print_config()

    print("=" * 78)
    print(f"TRAINING — {TOTAL_TIMESTEPS:,} timesteps")
    print("=" * 78 + "\n")

    vec_env = DummyVecEnv([_make_env(DSSAT_SEED)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=NORM_OBS,
        norm_reward=NORM_REWARD,
        clip_obs=CLIP_OBS,
        clip_reward=CLIP_REWARD,
    )

    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        verbose=0,
        device='cpu',
        seed=0,
    )

    callback = PCPPOLogger(EPISODE_LOG, METRICS_LOG, verbose=1)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)
    elapsed = time.time() - t0

    model.save(MODEL_PATH)
    vec_env.save(VECNORM_PATH)
    vec_env.close()

    print(f"\n✓ Training complete in {elapsed/60:.1f} min")
    print(f"✓ Model   → {MODEL_PATH}.zip")
    print(f"✓ VecNorm → {VECNORM_PATH}")

    _plot_training(EPISODE_LOG, METRICS_LOG)
    return MODEL_PATH, VECNORM_PATH


# ================================================================== #
# Plotting                                                             #
# ================================================================== #
def _rolling(arr, w=50):
    arr = np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(w)/w, mode='valid') if len(arr) >= w else arr


def _plot_training(ep_log, mt_log):
    try:
        with open(ep_log) as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return

    eps     = [int(r['episode'])        for r in rows]
    cum_r   = [float(r['cum_reward'])   for r in rows]
    yields  = [float(r['yield_kg_ha'])  for r in rows]
    n_vals  = [float(r['total_N_kg_ha'])for r in rows]
    w_vals  = [float(r['total_W_mm'])   for r in rows]
    r_yld   = [float(r['R_yield'])      for r in rows]
    r_ane   = [float(r['R_ane'])        for r in rows]
    r_wef   = [float(r['R_water_eff'])  for r in rows]

    W = 50
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('PC-PPO Training Curves (random weather, Dirichlet preferences)',
                 fontsize=12, fontweight='bold')

    panels = [
        (axes[0,0], eps, cum_r,  'Cumulative Reward',    'tab:blue'),
        (axes[0,1], eps, yields, 'Yield (kg/ha)',        'tab:green'),
        (axes[0,2], eps, n_vals, 'Total N (kg/ha)',      'tab:orange'),
        (axes[1,0], eps, w_vals, 'Total Water (mm)',     'tab:cyan'),
        (axes[1,1], eps, r_yld,  'R_yield',              'tab:purple'),
        (axes[1,2], eps, r_ane,  'R_ane',                'tab:red'),
    ]

    for ax, x, y, title, color in panels:
        ax.plot(x, y, alpha=0.2, color=color, linewidth=0.5)
        roll = _rolling(y, W)
        x_roll = x[W-1:] if len(x) >= W else x
        ax.plot(x_roll, roll, color=color, linewidth=1.8, label=f'rolling({W})')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Episode', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_TRAINING, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Training plot → {PLOT_TRAINING}")

    try:
        with open(mt_log) as f:
            mt = list(csv.DictReader(f))
    except FileNotFoundError:
        return

    updates   = [int(r['n_updates'])    for r in mt]
    pol_loss  = [float(r['policy_loss'])for r in mt]
    val_loss  = [float(r['value_loss']) for r in mt]
    ent_loss  = [float(r['entropy_loss'])for r in mt]

    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
    fig2.suptitle('PC-PPO Loss Curves', fontsize=12, fontweight='bold')
    for ax, y, title, color in zip(
        axes2,
        [pol_loss, val_loss, ent_loss],
        ['Policy Loss', 'Value Loss', 'Entropy Loss'],
        ['tab:blue', 'tab:orange', 'tab:green']
    ):
        ax.plot(updates, y, color=color, linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Updates', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_LOSSES, dpi=120, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Loss plot     → {PLOT_LOSSES}")


# ================================================================== #
# Evaluation — 4 corners                                              #
# ================================================================== #
def evaluate(model_path, vecnorm_path):
    print("\n" + "=" * 78)
    print(f"EVAL — preference corners  |  model: {os.path.basename(model_path)}")
    print("=" * 78)

    model = PPO.load(model_path, device='cpu')

    EVAL_FIELDS = [
        'corner', 'episode',
        'w_yield', 'w_neff', 'w_water',
        'cum_reward', 'R_yield', 'R_ane', 'R_water_eff',
        'yield_kg_ha', 'total_N_kg_ha', 'total_W_mm', 'total_rain_mm',
        'ep_length',
    ]

    all_rows = []

    for corner_name, w_corner in EVAL_CORNERS.items():
        print(f"\n  ── {corner_name}  w={w_corner.round(2)} ──")
        print(f"  {'ep':>4}  {'cum_R':>7}  {'yield':>6}  "
              f"{'N':>5}  {'W':>5}  {'R_yld':>7}  {'R_ane':>7}  {'R_wef':>7}")
        print(f"  {'-'*4}  {'-'*7}  {'-'*6}  "
              f"{'-'*5}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}")

        corner_rows = []
        for ep in range(EVAL_EPISODES):
            eval_env = DummyVecEnv([_make_env(dssat_seed=3000 + ep)])
            eval_env = VecNormalize.load(vecnorm_path, eval_env)
            eval_env.training    = False
            eval_env.norm_reward = False

            obs = eval_env.reset()
            # Force the preference corner into the observation
            obs[0, -3:] = w_corner
            cum_r = 0.0; ep_len = 0; done = False; last_info = {}

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done_arr, info_arr = eval_env.step(action)
                obs[0, -3:] = w_corner   # keep w fixed throughout episode
                cum_r    += float(r[0])
                ep_len   += 1
                done      = bool(done_arr[0])
                last_info = info_arr[0]

            eval_env.close()

            sos = last_info.get('sos_state', {})
            rv  = last_info.get('reward_vec', np.zeros(3))

            row = {
                'corner':        corner_name,
                'episode':       ep + 1,
                'w_yield':       round(float(w_corner[0]), 4),
                'w_neff':        round(float(w_corner[1]), 4),
                'w_water':       round(float(w_corner[2]), 4),
                'cum_reward':    round(cum_r, 6),
                'R_yield':       round(float(rv[0]), 6),
                'R_ane':         round(float(rv[1]), 6),
                'R_water_eff':   round(float(rv[2]), 6),
                'yield_kg_ha':   round(float(sos.get('grnwt',         0.0)), 1),
                'total_N_kg_ha': round(float(sos.get('total_nitrogen', 0.0)), 1),
                'total_W_mm':    round(float(sos.get('total_water',    0.0)), 1),
                'total_rain_mm': round(float(sos.get('total_rain',     0.0)), 1),
                'ep_length':     ep_len,
            }
            corner_rows.append(row)
            all_rows.append(row)

            print(
                f"  {ep+1:4d}  {cum_r:+7.3f}  "
                f"{row['yield_kg_ha']:6.0f}  "
                f"{row['total_N_kg_ha']:5.0f}  "
                f"{row['total_W_mm']:5.0f}  "
                f"{row['R_yield']:+7.4f}  "
                f"{row['R_ane']:+7.4f}  "
                f"{row['R_water_eff']:+7.4f}"
            )

        def ms(k): return np.mean([r[k] for r in corner_rows])
        print(f"\n  mean → yield={ms('yield_kg_ha'):.0f} kg/ha  "
              f"N={ms('total_N_kg_ha'):.0f}  W={ms('total_W_mm'):.0f} mm  "
              f"R_yld={ms('R_yield'):+.3f}  R_ane={ms('R_ane'):+.3f}  "
              f"R_wef={ms('R_water_eff'):+.3f}")

    # Summary block
    print("\n" + "=" * 78)
    print(f"  {'Corner':<10}  {'yield':>7}  {'N':>5}  {'W':>5}  "
          f"{'R_yield':>8}  {'R_ane':>8}  {'R_water':>8}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*5}  {'-'*5}  "
          f"{'-'*8}  {'-'*8}  {'-'*8}")
    for corner_name in EVAL_CORNERS:
        cr = [r for r in all_rows if r['corner'] == corner_name]
        def m(k): return np.mean([r[k] for r in cr])
        print(f"  {corner_name:<10}  {m('yield_kg_ha'):7.0f}  "
              f"{m('total_N_kg_ha'):5.0f}  {m('total_W_mm'):5.0f}  "
              f"{m('R_yield'):+8.4f}  {m('R_ane'):+8.4f}  "
              f"{m('R_water_eff'):+8.4f}")
    print("=" * 78)

    with open(EVAL_LOG, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✓ Eval results → {EVAL_LOG}")


# ================================================================== #
# Main                                                                 #
# ================================================================== #
if __name__ == '__main__':
    try:
        model_path, vecnorm_path = train()
        evaluate(model_path, vecnorm_path)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise
