"""
PPO — NITROGEN MINIMISATION objective.

Reward tuned to reduce total N application while accepting slight yield loss.
  N_EXCESS 200→100 kg/ha, penalty split 30% water / 70% N,
  W_RESOURCE 0.15→0.30, W_FERT 0.30→0.20, W_ANE 0.30→0.35

Run inside Docker:
    cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
    /opt/gym_dssat_pdi/bin/python3 -u 04c_ppo_min_nitrogen.py
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
PCSmartFarmEnv = import_module('pc_env_nitrogen').PCSmartFarmEnv


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

MODEL_PATH      = '/tmp/ppo_min_nitrogen_v3_1M'
VECNORM_PATH    = '/tmp/ppo_min_nitrogen_v3_1M_vecnorm.pkl'
EPISODE_LOG     = '/workspace/episode_log_nitrogen_v3.csv'
METRICS_LOG     = '/workspace/training_metrics_nitrogen_v3.csv'
EVAL_LOG        = '/workspace/eval_results_nitrogen_v3.csv'
PLOT_TRAINING   = '/workspace/plot_training_nitrogen_v3.png'
PLOT_LOSSES     = '/workspace/plot_ppo_losses_nitrogen_v3.png'

EVAL_EPISODES   = 5
DSSAT_SEED      = 123
ENABLE_FAULTS   = False
RANDOM_WEATHER  = True

NORM_OBS        = True
NORM_REWARD     = True
CLIP_OBS        = 5.0
CLIP_REWARD     = 10.0

ROLLING_WINDOW  = 100   # episodes for rolling mean


# ================================================================== #
# Env factory                                                          #
# ================================================================== #
def _make_env(dssat_seed):
    def _init():
        return PCSmartFarmEnv(
            mode='all',
            dssat_seed=dssat_seed,
            run_dssat_location='run_dssat',
            enable_faults=ENABLE_FAULTS,
        )
    return _init


# ================================================================== #
# Hyperparameter print                                                 #
# ================================================================== #
def _print_config():
    W = 78
    print("=" * W)
    print("SIMPLE PPO — CONFIGURATION")
    print("=" * W)
    print(f"  {'Environment'}")
    print(f"    {'mode':<20} all")
    print(f"    {'random_weather':<20} {RANDOM_WEATHER}")
    print(f"    {'enable_faults':<20} {ENABLE_FAULTS}")
    print(f"    {'dssat_seed':<20} {DSSAT_SEED}")
    print()
    print(f"  {'PPO Hyperparameters'}")
    print(f"    {'total_timesteps':<20} {TOTAL_TIMESTEPS:,}")
    print(f"    {'n_steps':<20} {N_STEPS:,}")
    print(f"    {'batch_size':<20} {BATCH_SIZE}")
    print(f"    {'n_epochs':<20} {N_EPOCHS}")
    print(f"    {'learning_rate':<20} {LEARNING_RATE}")
    print(f"    {'gamma':<20} {GAMMA}")
    print(f"    {'gae_lambda':<20} {GAE_LAMBDA}")
    print(f"    {'clip_range':<20} {CLIP_RANGE}")
    print(f"    {'ent_coef':<20} {ENT_COEF}")
    print(f"    {'max_grad_norm':<20} {MAX_GRAD_NORM}")
    print(f"    {'device':<20} cpu")
    print()
    print(f"  {'Normalisation (VecNormalize)'}")
    print(f"    {'norm_obs':<20} {NORM_OBS}")
    print(f"    {'norm_reward':<20} {NORM_REWARD}")
    print(f"    {'clip_obs':<20} {CLIP_OBS}")
    print(f"    {'clip_reward':<20} {CLIP_REWARD}")
    print()
    print(f"  {'Reward Weights (seasonal)'}")
    print(f"    {'W_yield':<20} 0.40")
    print(f"    {'W_hiad':<20} 0.10")
    print(f"    {'W_ane':<20} 0.30")
    print(f"    {'W_penalty':<20} 0.20")
    print("=" * W)
    print()


# ================================================================== #
# Logging callback                                                     #
# ================================================================== #
class TrainingLogger(BaseCallback):

    EP_FIELDS = [
        'timestep', 'episode', 'ep_length', 'cum_reward',
        'R_water', 'R_fert', 'R_resource', 'R_losses', 'R_daily_mean',
        'R_seasonal', 'R_yield', 'R_hiad', 'R_ane', 'R_penalty',
        'yield_kg_ha', 'total_N_kg_ha', 'total_W_mm', 'total_rain_mm',
        'energy_budget', 'comm_quality',
    ]

    MT_FIELDS = [
        'timestep', 'n_updates', 'ep_rew_mean', 'ep_len_mean',
        'policy_loss', 'value_loss', 'entropy_loss',
        'explained_variance', 'clip_fraction', 'approx_kl', 'fps',
    ]

    def __init__(self, episode_log_path, metrics_log_path, verbose=1):
        super().__init__(verbose)
        self.episode_log_path = episode_log_path
        self.metrics_log_path = metrics_log_path

        self._ep_count      = 0
        self._ep_cum_reward = 0.0
        self._ep_length     = 0
        self._ep_daily_r    = []

        # rolling window for progress summaries
        self._recent = deque(maxlen=ROLLING_WINDOW)

        self._ep_file = self._mt_file = None
        self._ep_writer = self._mt_writer = None

    # -------------------------------------------------------------- #
    def _on_training_start(self):
        self._ep_file   = open(self.episode_log_path, 'w', newline='', buffering=1)
        self._ep_writer = csv.DictWriter(self._ep_file, fieldnames=self.EP_FIELDS)
        self._ep_writer.writeheader()

        self._mt_file   = open(self.metrics_log_path, 'w', newline='', buffering=1)
        self._mt_writer = csv.DictWriter(self._mt_file, fieldnames=self.MT_FIELDS)
        self._mt_writer.writeheader()

        print(f"  Episode log  → {self.episode_log_path}")
        print(f"  Metrics log  → {self.metrics_log_path}\n")
        print(f"  {'ep':>5}  {'step':>8}  {'len':>4}  {'cum_R':>7}  "
              f"{'yield':>6}  {'N':>5}  {'W':>5}  "
              f"{'R_seas':>7}  {'R_yld':>7}  {'R_hiad':>7}  {'R_ane':>7}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*4}  {'-'*7}  "
              f"{'-'*6}  {'-'*5}  {'-'*5}  "
              f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    # -------------------------------------------------------------- #
    def _on_step(self):
        dones   = self.locals.get('dones',   [False])
        infos   = self.locals.get('infos',   [{}])
        rewards = self.locals.get('rewards', [0.0])

        self._ep_cum_reward += float(rewards[0])
        self._ep_length     += 1

        c = infos[0].get('reward_components', {})
        self._ep_daily_r.append(float(c.get('R_daily', 0.0)))

        if dones[0]:
            sos = infos[0].get('sos_state', {})
            self._ep_count += 1

            row = {
                'timestep':     self.num_timesteps,
                'episode':      self._ep_count,
                'ep_length':    self._ep_length,
                'cum_reward':   round(self._ep_cum_reward, 6),
                'R_water':      round(float(c.get('R_water',    0.0)), 6),
                'R_fert':       round(float(c.get('R_fert',     0.0)), 6),
                'R_resource':   round(float(c.get('R_resource', 0.0)), 6),
                'R_losses':     round(float(c.get('R_losses',   0.0)), 6),
                'R_daily_mean': round(float(np.mean(self._ep_daily_r)), 8),
                'R_seasonal':   round(float(c.get('R_seasonal', 0.0)), 6),
                'R_yield':      round(float(c.get('R_yield',    0.0)), 6),
                'R_hiad':       round(float(c.get('R_hiad',     0.0)), 6),
                'R_ane':        round(float(c.get('R_ane',      0.0)), 6),
                'R_penalty':    round(float(c.get('R_penalty',  0.0)), 6),
                'yield_kg_ha':  round(float(sos.get('grnwt',         0.0)), 1),
                'total_N_kg_ha':round(float(sos.get('total_nitrogen', 0.0)), 1),
                'total_W_mm':   round(float(sos.get('total_water',    0.0)), 1),
                'total_rain_mm':round(float(sos.get('total_rain',     0.0)), 1),
                'energy_budget':round(float(sos.get('energy_budget',  0.0)), 3),
                'comm_quality': round(float(sos.get('comm_quality',   0.0)), 3),
            }
            self._ep_writer.writerow(row)
            self._ep_file.flush()
            self._recent.append(row)

            if self.verbose >= 1:
                print(
                    f"  {self._ep_count:5d}  {self.num_timesteps:8d}  "
                    f"{row['ep_length']:4d}  "
                    f"{row['cum_reward']:+7.3f}  "
                    f"{row['yield_kg_ha']:6.0f}  "
                    f"{row['total_N_kg_ha']:5.0f}  "
                    f"{row['total_W_mm']:5.0f}  "
                    f"{row['R_seasonal']:+7.4f}  "
                    f"{row['R_yield']:+7.4f}  "
                    f"{row['R_hiad']:+7.4f}  "
                    f"{row['R_ane']:+7.4f}"
                )

            # Rolling mean summary every ROLLING_WINDOW episodes
            if self._ep_count % ROLLING_WINDOW == 0 and len(self._recent) > 0:
                def rm(k): return np.mean([r[k] for r in self._recent])
                print(
                    f"\n  ── rolling mean (last {len(self._recent)} ep) ──  "
                    f"cum_R={rm('cum_reward'):+.3f}  "
                    f"yield={rm('yield_kg_ha'):.0f} kg/ha  "
                    f"N={rm('total_N_kg_ha'):.0f}  "
                    f"W={rm('total_W_mm'):.0f} mm  "
                    f"R_seas={rm('R_seasonal'):+.4f}  "
                    f"R_ane={rm('R_ane'):+.4f}\n"
                )

            self._ep_cum_reward = 0.0
            self._ep_length     = 0
            self._ep_daily_r    = []

        return True

    # -------------------------------------------------------------- #
    def _on_rollout_end(self):
        lv = self.model.logger.name_to_value

        row = {
            'timestep':           self.num_timesteps,
            'n_updates':          int(lv.get('train/n_updates',                0)),
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

    # -------------------------------------------------------------- #
    def _on_training_end(self):
        for f in (self._ep_file, self._mt_file):
            if f:
                f.close()
        print(f"\n✓ Episode log  saved → {self.episode_log_path}")
        print(f"✓ Metrics log  saved → {self.metrics_log_path}")


# ================================================================== #
# Plot training curves                                                 #
# ================================================================== #
def _rolling(arr, w=50):
    """Compute rolling mean with window w."""
    arr = np.array(arr, dtype=float)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode='valid')


def _plot_training_curves(ep_log, mt_log):
    """Read CSVs and save two PNG figures to /workspace/."""
    # ---- load episode log ----
    ep_rows = []
    try:
        with open(ep_log) as f:
            ep_rows = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"  ⚠  {ep_log} not found — skipping plots")
        return

    episodes   = [int(r['episode'])       for r in ep_rows]
    cum_r      = [float(r['cum_reward'])   for r in ep_rows]
    yield_v    = [float(r['yield_kg_ha'])  for r in ep_rows]
    n_v        = [float(r['total_N_kg_ha'])for r in ep_rows]
    w_v        = [float(r['total_W_mm'])   for r in ep_rows]
    r_seas     = [float(r['R_seasonal'])   for r in ep_rows]
    r_ane      = [float(r['R_ane'])        for r in ep_rows]

    W = 50   # rolling window for plots

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('PPO Training Curves — SmartFarm (random weather, 1M steps)',
                 fontsize=13, fontweight='bold')

    panels = [
        (axes[0, 0], episodes, cum_r,   'Cumulative Reward',  'Episode', 'cum_reward',  'tab:blue'),
        (axes[0, 1], episodes, yield_v, 'Yield (kg/ha)',       'Episode', 'kg/ha',       'tab:green'),
        (axes[0, 2], episodes, n_v,     'Total N Applied (kg/ha)', 'Episode', 'kg/ha',  'tab:orange'),
        (axes[1, 0], episodes, w_v,     'Total Water (mm)',    'Episode', 'mm',          'tab:cyan'),
        (axes[1, 1], episodes, r_seas,  'R_seasonal',         'Episode', 'R_seasonal',  'tab:purple'),
        (axes[1, 2], episodes, r_ane,   'R_ane (N efficiency)','Episode', 'R_ane',      'tab:red'),
    ]

    for ax, x, y, title, xlabel, ylabel, color in panels:
        ax.plot(x, y, alpha=0.25, color=color, linewidth=0.6)
        roll = _rolling(y, W)
        x_roll = x[W - 1:] if len(x) >= W else x
        ax.plot(x_roll, roll, color=color, linewidth=1.8,
                label=f'rolling mean ({W} ep)')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_TRAINING, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Training plot  saved → {PLOT_TRAINING}")

    # ---- load metrics log ----
    mt_rows = []
    try:
        with open(mt_log) as f:
            mt_rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return

    if not mt_rows:
        return

    updates     = [int(r['n_updates'])         for r in mt_rows]
    pol_loss    = [float(r['policy_loss'])      for r in mt_rows]
    val_loss    = [float(r['value_loss'])       for r in mt_rows]
    ent_loss    = [float(r['entropy_loss'])     for r in mt_rows]

    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
    fig2.suptitle('PPO Loss Curves', fontsize=12, fontweight='bold')

    loss_panels = [
        (axes2[0], updates, pol_loss, 'Policy Loss',  'tab:blue'),
        (axes2[1], updates, val_loss, 'Value Loss',   'tab:orange'),
        (axes2[2], updates, ent_loss, 'Entropy Loss', 'tab:green'),
    ]
    for ax, x, y, title, color in loss_panels:
        ax.plot(x, y, color=color, linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Updates', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_LOSSES, dpi=120, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Loss plot      saved → {PLOT_LOSSES}")


# ================================================================== #
# Training                                                             #
# ================================================================== #
def train():
    _print_config()

    print("=" * 78)
    print(f"TRAINING — {TOTAL_TIMESTEPS:,} timesteps  |  n_steps={N_STEPS}")
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

    callback = TrainingLogger(
        episode_log_path=EPISODE_LOG,
        metrics_log_path=METRICS_LOG,
        verbose=1,
    )

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)
    elapsed = time.time() - t0

    model.save(MODEL_PATH)
    vec_env.save(VECNORM_PATH)
    vec_env.close()

    print(f"\n✓ Training complete in {elapsed / 60:.1f} min")
    print(f"✓ Model    saved → {MODEL_PATH}.zip")
    print(f"✓ VecNorm  saved → {VECNORM_PATH}")

    _plot_training_curves(EPISODE_LOG, METRICS_LOG)

    return MODEL_PATH, VECNORM_PATH


# ================================================================== #
# Evaluation                                                           #
# ================================================================== #
def evaluate(model_path, vecnorm_path, n_episodes=EVAL_EPISODES):
    print("\n" + "=" * 78)
    print(f"EVAL — {n_episodes} deterministic episodes  |  model: {os.path.basename(model_path)}")
    print("=" * 78)

    model = PPO.load(model_path, device='cpu')

    EVAL_FIELDS = [
        'episode', 'cum_reward',
        'R_seasonal', 'R_yield', 'R_hiad', 'R_ane', 'R_penalty',
        'yield_kg_ha', 'total_N_kg_ha', 'total_W_mm', 'total_rain_mm',
        'ep_length',
    ]

    rows = []
    print(f"\n  {'ep':>4}  {'cum_R':>7}  {'yield':>6}  "
          f"{'N':>5}  {'W':>5}  {'R_seas':>7}  {'R_yld':>7}  {'R_hiad':>7}  {'R_ane':>7}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*6}  "
          f"{'-'*5}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    for ep in range(n_episodes):
        eval_env = DummyVecEnv([_make_env(dssat_seed=2000 + ep)])
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
        eval_env.training    = False
        eval_env.norm_reward = False

        obs       = eval_env.reset()
        cum_r     = 0.0
        ep_len    = 0
        done      = False
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done_arr, info_arr = eval_env.step(action)
            cum_r    += float(r[0])
            ep_len   += 1
            done      = bool(done_arr[0])
            last_info = info_arr[0]

        eval_env.close()

        sos = last_info.get('sos_state', {})
        c   = last_info.get('reward_components', {})

        row = {
            'episode':      ep + 1,
            'cum_reward':   round(cum_r, 6),
            'R_seasonal':   round(float(c.get('R_seasonal', 0.0)), 6),
            'R_yield':      round(float(c.get('R_yield',    0.0)), 6),
            'R_hiad':       round(float(c.get('R_hiad',     0.0)), 6),
            'R_ane':        round(float(c.get('R_ane',      0.0)), 6),
            'R_penalty':    round(float(c.get('R_penalty',  0.0)), 6),
            'yield_kg_ha':  round(float(sos.get('grnwt',         0.0)), 1),
            'total_N_kg_ha':round(float(sos.get('total_nitrogen', 0.0)), 1),
            'total_W_mm':   round(float(sos.get('total_water',    0.0)), 1),
            'total_rain_mm':round(float(sos.get('total_rain',     0.0)), 1),
            'ep_length':    ep_len,
        }
        rows.append(row)

        print(
            f"  {ep+1:4d}  {cum_r:+7.3f}  "
            f"{row['yield_kg_ha']:6.0f}  "
            f"{row['total_N_kg_ha']:5.0f}  "
            f"{row['total_W_mm']:5.0f}  "
            f"{row['R_seasonal']:+7.4f}  "
            f"{row['R_yield']:+7.4f}  "
            f"{row['R_hiad']:+7.4f}  "
            f"{row['R_ane']:+7.4f}"
        )

    def _ms(key):
        vals = [r[key] for r in rows]
        return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

    y_m,  y_s,  y_lo, y_hi  = _ms('yield_kg_ha')
    n_m,  n_s,  n_lo, n_hi  = _ms('total_N_kg_ha')
    w_m,  w_s,  w_lo, w_hi  = _ms('total_W_mm')
    rs_m, *_                 = _ms('R_seasonal')
    ry_m, _,    _,    _      = _ms('R_yield')
    rh_m, _,    _,    _      = _ms('R_hiad')
    ra_m, _,    _,    _      = _ms('R_ane')
    rp_m, _,    _,    _      = _ms('R_penalty')
    cr_m, cr_s, _,    _      = _ms('cum_reward')

    print("\n" + "=" * 78)
    print(f"  {'yield':<12}  mean={y_m:6.0f}  max={y_hi:6.0f}  "
          f"min={y_lo:6.0f}  std={y_s:5.0f}  kg/ha")
    print(f"  {'nitrogen':<12}  mean={n_m:6.0f}  max={n_hi:6.0f}  "
          f"min={n_lo:6.0f}  std={n_s:5.0f}  kg/ha")
    print(f"  {'water':<12}  mean={w_m:6.0f}  max={w_hi:6.0f}  "
          f"min={w_lo:6.0f}  std={w_s:5.0f}  mm")
    print(f"  {'R_seasonal':<12}  mean={rs_m:+.4f}  "
          f"R_yield={ry_m:+.4f}  R_hiad={rh_m:+.4f}  "
          f"R_ane={ra_m:+.4f}  R_penalty={rp_m:+.4f}")
    print(f"  {'cum_R':<12}  mean={cr_m:+.4f}  std={cr_s:.4f}")
    print("=" * 78)

    summary = {k: '' for k in EVAL_FIELDS}
    summary['episode'] = 'MEAN'
    for k in [k for k in EVAL_FIELDS if k != 'episode']:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        summary[k] = round(float(np.mean(vals)), 4) if vals else ''

    with open(EVAL_LOG, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(summary)

    print(f"\n✓ Eval results saved → {EVAL_LOG}")


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
