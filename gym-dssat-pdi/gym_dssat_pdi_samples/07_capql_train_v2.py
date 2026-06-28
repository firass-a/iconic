"""
CAPQL v2 — canonical CAPQL training script.

Uses capql_env_v2.CAPQLEnv + 02_smart_farm_env_pcppo rewards.
Observation layout matches pc_env.py (farmer-realistic 11-D state + w).
Existing models/capql_v2/actor.pt is incompatible — retrain after this change.
20% of episodes use a pure preference corner (yield / N-eff / water).

Outputs (separate from v1 for easy comparison):
  models/capql_v2/actor.pt          ← persisted on host via /workspace mount
  /workspace/capql_v2_episode_log.csv
  /workspace/capql_v2_eval_results.csv
  /workspace/capql_v2_plot_training.png
  /workspace/capql_v2_plot_losses.png

Run inside Docker (mount iconic → /workspace):
    cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
    /opt/gym_dssat_pdi/bin/python3 -u 07_capql_train_v2.py 2>&1 | tee /workspace/train_capql_v2.log

Or from Windows:  .\\train_capql_v2_docker.ps1
"""
import csv
import os
import sys
import time
import numpy as np
from collections import deque
from importlib import import_module

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

CAPQLEnv = import_module('capql_env_v2').CAPQLEnv


# ================================================================== #
# Config                                                               #
# ================================================================== #
TOTAL_STEPS    = 1_000_000
WARMUP_STEPS   = 5_000
BUFFER_SIZE    = 200_000
BATCH_SIZE     = 256
LR_ACTOR       = 3e-4
LR_CRITIC      = 3e-4
GAMMA          = 0.99
TAU            = 0.005
POLICY_FREQ    = 2
GRAD_CLIP      = 1.0

EXPL_NOISE_START = 0.3
EXPL_NOISE_END   = 0.05
EXPL_NOISE_STEPS = 800_000

TP_NOISE_STD  = np.array([8.0, 2.0], dtype=np.float32)
TP_NOISE_CLIP = np.array([20.0, 5.0], dtype=np.float32)

AUG_LAMBDA    = 0.5
AUG_EPS       = 0.1
USE_RELABEL   = True

N_OBJECTIVES  = 3
OBS_DIM       = 14
ACTION_DIM    = 2
HIDDEN        = 256

ACT_LOW   = np.array([0.0,   0.0], dtype=np.float32)
ACT_HIGH  = np.array([200.0, 50.0], dtype=np.float32)
ACT_RANGE = ACT_HIGH - ACT_LOW


def _default_model_dir() -> str:
    """Save weights on the mounted workspace (survives container stop)."""
    if os.path.isdir('/workspace'):
        return '/workspace/models/capql_v2'
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, '..', '..', 'models', 'capql_v2'))


MODEL_DIR    = _default_model_dir()
os.makedirs(MODEL_DIR, exist_ok=True)
EPISODE_LOG  = '/workspace/capql_v2_episode_log.csv'
EVAL_LOG     = '/workspace/capql_v2_eval_results.csv'
PLOT_TRAIN   = '/workspace/capql_v2_plot_training.png'
PLOT_LOSSES  = '/workspace/capql_v2_plot_losses.png'

DSSAT_SEED      = 123
EVAL_EPISODES   = 5
LOG_INTERVAL    = 10
ROLLING_WINDOW  = 50

EVAL_CORNERS = {
    'yield':    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    'n_eff':    np.array([0.0, 1.0, 0.0], dtype=np.float32),
    'water':    np.array([0.0, 0.0, 1.0], dtype=np.float32),
    'balanced': np.array([1/3, 1/3, 1/3], dtype=np.float32),
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_ACT_MID  = torch.FloatTensor((ACT_HIGH + ACT_LOW) / 2.0).to(DEVICE)
_ACT_HALF = torch.FloatTensor(ACT_RANGE / 2.0).to(DEVICE)
_ACT_LOW  = torch.FloatTensor(ACT_LOW).to(DEVICE)
_ACT_HIGH = torch.FloatTensor(ACT_HIGH).to(DEVICE)


# ================================================================== #
# Networks                                                             #
# ================================================================== #
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN,  HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN,  ACTION_DIM), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs) * _ACT_HALF + _ACT_MID


class TwinCritic(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = OBS_DIM + ACTION_DIM
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN,  HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN,  1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN,  HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN,  1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs, action):
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


# ================================================================== #
# Replay buffer                                                        #
# ================================================================== #
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.ptr = self.size = 0

        self.crop_obs      = np.zeros((capacity, 11),           dtype=np.float32)
        self.next_crop_obs = np.zeros((capacity, 11),           dtype=np.float32)
        self.action        = np.zeros((capacity, ACTION_DIM),   dtype=np.float32)
        self.r_daily       = np.zeros((capacity, 1),            dtype=np.float32)
        self.reward_vec    = np.zeros((capacity, N_OBJECTIVES), dtype=np.float32)
        self.done          = np.zeros((capacity, 1),            dtype=np.float32)
        self.w_stored      = np.zeros((capacity, N_OBJECTIVES), dtype=np.float32)

    def add(self, crop_obs, action, r_daily, reward_vec, next_crop_obs, done, w):
        i = self.ptr
        self.crop_obs[i]      = crop_obs
        self.next_crop_obs[i] = next_crop_obs
        self.action[i]        = action
        self.r_daily[i]       = float(r_daily)
        self.reward_vec[i]    = reward_vec
        self.done[i]          = float(done)
        self.w_stored[i]      = w
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        crop_obs      = self.crop_obs[idx]
        next_crop_obs = self.next_crop_obs[idx]
        action        = self.action[idx]
        r_daily       = self.r_daily[idx]
        reward_vec    = self.reward_vec[idx]
        done          = self.done[idx]

        if USE_RELABEL:
            w_batch = np.random.dirichlet(
                np.ones(N_OBJECTIVES), size=batch_size
            ).astype(np.float32)
        else:
            w_batch = self.w_stored[idx]

        obs      = np.concatenate([crop_obs,      w_batch], axis=1)
        next_obs = np.concatenate([next_crop_obs, w_batch], axis=1)

        return (
            torch.FloatTensor(obs).to(DEVICE),
            torch.FloatTensor(action).to(DEVICE),
            torch.FloatTensor(r_daily).to(DEVICE),
            torch.FloatTensor(reward_vec).to(DEVICE),
            torch.FloatTensor(next_obs).to(DEVICE),
            torch.FloatTensor(done).to(DEVICE),
            torch.FloatTensor(w_batch).to(DEVICE),
        )


# ================================================================== #
# Concave augmentation                                                 #
# ================================================================== #
def augmented_reward(r_daily, reward_vec, done, w):
    scalarized = (w * reward_vec).sum(dim=-1, keepdim=True)
    concave    = AUG_LAMBDA * torch.mean(
        torch.log(reward_vec + 1.0 + AUG_EPS), dim=-1, keepdim=True
    )
    return r_daily + done * (scalarized + concave)


# ================================================================== #
# CAPQL Agent                                                          #
# ================================================================== #
class CAPQLAgent:

    def __init__(self):
        self.actor  = Actor().to(DEVICE)
        self.critic = TwinCritic().to(DEVICE)

        self.actor_tgt  = Actor().to(DEVICE)
        self.critic_tgt = TwinCritic().to(DEVICE)
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self._update_count = 0

    def select_action(self, obs, step, deterministic=False):
        with torch.no_grad():
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            action = self.actor(obs_t).squeeze(0).cpu().numpy()

        if not deterministic:
            frac  = min(1.0, step / EXPL_NOISE_STEPS)
            sigma = EXPL_NOISE_START + frac * (EXPL_NOISE_END - EXPL_NOISE_START)
            noise = np.random.normal(0.0, 1.0, size=ACTION_DIM) * sigma * ACT_RANGE
            action = np.clip(action + noise, ACT_LOW, ACT_HIGH)

        return action.astype(np.float32)

    def update(self, buffer):
        obs, action, r_daily, reward_vec, next_obs, done, w = buffer.sample(BATCH_SIZE)

        r_aug = augmented_reward(r_daily, reward_vec, done, w)

        with torch.no_grad():
            next_action = self.actor_tgt(next_obs)

            tp_noise = torch.zeros_like(next_action)
            tp_noise[:, 0] = (torch.randn(BATCH_SIZE, device=DEVICE)
                              * TP_NOISE_STD[0]).clamp(-TP_NOISE_CLIP[0], TP_NOISE_CLIP[0])
            tp_noise[:, 1] = (torch.randn(BATCH_SIZE, device=DEVICE)
                              * TP_NOISE_STD[1]).clamp(-TP_NOISE_CLIP[1], TP_NOISE_CLIP[1])

            next_action = (next_action + tp_noise).clamp(_ACT_LOW, _ACT_HIGH)

            tq1, tq2   = self.critic_tgt(next_obs, next_action)
            target_q   = r_aug + GAMMA * (1.0 - done) * torch.min(tq1, tq2)

        q1, q2      = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.critic_opt.step()

        self._update_count += 1
        actor_loss_val = float('nan')

        if self._update_count % POLICY_FREQ == 0:
            actor_loss = -self.critic.q_min(obs, self.actor(obs)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
            self.actor_opt.step()

            actor_loss_val = actor_loss.item()

            for p, tp in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                tp.data.copy_(TAU * p.data + (1.0 - TAU) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                tp.data.copy_(TAU * p.data + (1.0 - TAU) * tp.data)

        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss_val}

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(),  os.path.join(directory, 'actor.pt'))
        torch.save(self.critic.state_dict(), os.path.join(directory, 'critic.pt'))
        print(f"✓ Model saved → {directory}")

    def load(self, directory):
        self.actor.load_state_dict(
            torch.load(os.path.join(directory, 'actor.pt'), map_location=DEVICE))
        self.critic.load_state_dict(
            torch.load(os.path.join(directory, 'critic.pt'), map_location=DEVICE))
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())


# ================================================================== #
# Training                                                             #
# ================================================================== #
def train():
    _print_config()

    env    = CAPQLEnv(mode='all', dssat_seed=DSSAT_SEED,
                      run_dssat_location='run_dssat')
    buffer = ReplayBuffer(BUFFER_SIZE)
    agent  = CAPQLAgent()

    EP_FIELDS = [
        'episode', 'timestep', 'ep_length', 'cum_reward',
        'w_yield', 'w_neff', 'w_water',
        'R_yield', 'R_ane', 'R_water_eff',
        'yield_kg_ha', 'total_N_kg_ha', 'total_W_mm', 'total_rain_mm',
        'critic_loss', 'actor_loss',
    ]
    ep_file   = open(EPISODE_LOG, 'w', newline='', buffering=1)
    ep_writer = csv.DictWriter(ep_file, fieldnames=EP_FIELDS)
    ep_writer.writeheader()

    obs, info  = env.reset()
    crop_obs   = obs[:11].copy()
    w          = info['preference_w'].copy()

    ep_count  = 0
    ep_daily  = 0.0
    ep_length = 0
    recent    = deque(maxlen=ROLLING_WINDOW)
    last_losses = {'critic_loss': float('nan'), 'actor_loss': float('nan')}

    t_start = time.time()
    _print_header()

    for step in range(1, TOTAL_STEPS + 1):

        if step < WARMUP_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, step)

        next_obs, r_daily, done, _, step_info = env.step(action)
        next_crop_obs = next_obs[:11].copy()
        reward_vec    = step_info.get('reward_vec', np.zeros(N_OBJECTIVES, dtype=np.float32))

        buffer.add(crop_obs, action, r_daily, reward_vec, next_crop_obs, done, w)

        ep_daily  += r_daily
        ep_length += 1

        if step >= WARMUP_STEPS and buffer.size >= BATCH_SIZE:
            last_losses = agent.update(buffer)

        if done:
            ep_count += 1
            sos = step_info.get('sos_state', {})
            rv  = reward_vec

            cum_reward = ep_daily + float(np.dot(w, rv))

            row = {
                'episode':       ep_count,
                'timestep':      step,
                'ep_length':     ep_length,
                'cum_reward':    round(cum_reward, 4),
                'w_yield':       round(float(w[0]), 4),
                'w_neff':        round(float(w[1]), 4),
                'w_water':       round(float(w[2]), 4),
                'R_yield':       round(float(rv[0]), 4),
                'R_ane':         round(float(rv[1]), 4),
                'R_water_eff':   round(float(rv[2]), 4),
                'yield_kg_ha':   round(float(sos.get('grnwt',         0.0)), 1),
                'total_N_kg_ha': round(float(sos.get('total_nitrogen', 0.0)), 1),
                'total_W_mm':    round(float(sos.get('total_water',    0.0)), 1),
                'total_rain_mm': round(float(sos.get('total_rain',     0.0)), 1),
                'critic_loss':   round(last_losses['critic_loss'], 6),
                'actor_loss':    round(last_losses['actor_loss'],  6)
                                 if not np.isnan(last_losses['actor_loss']) else '',
            }
            ep_writer.writerow(row)
            ep_file.flush()
            recent.append(row)

            if ep_count % LOG_INTERVAL == 0:
                _print_row(ep_count, step, row)

            if ep_count % ROLLING_WINDOW == 0 and len(recent) > 0:
                _print_rolling(recent)

            obs, info  = env.reset()
            crop_obs   = obs[:11].copy()
            w          = info['preference_w'].copy()
            ep_daily   = 0.0
            ep_length  = 0
        else:
            obs      = next_obs
            crop_obs = next_crop_obs

    elapsed = time.time() - t_start
    print(f"\n✓ Training complete in {elapsed / 60:.1f} min")

    agent.save(MODEL_DIR)
    ep_file.close()
    print(f"✓ Episode log → {EPISODE_LOG}")

    _plot_training(EPISODE_LOG)
    return agent


# ================================================================== #
# Evaluation                                                           #
# ================================================================== #
def evaluate(agent):
    print("\n" + "=" * 78)
    print("EVAL — CAPQL v2 preference corners")
    print("=" * 78)

    EVAL_FIELDS = [
        'corner', 'episode',
        'w_yield', 'w_neff', 'w_water',
        'cum_reward', 'R_yield', 'R_ane', 'R_water_eff',
        'yield_kg_ha', 'total_N_kg_ha', 'total_W_mm', 'total_rain_mm',
        'ep_length',
    ]
    all_rows = []

    for corner_name, w_corner in EVAL_CORNERS.items():
        print(f"\n  ── {corner_name}  w={w_corner.round(3)} ──")
        print(f"  {'ep':>4}  {'cum_R':>7}  {'yield':>6}  {'N':>5}  {'W':>5}  "
              f"{'R_yld':>7}  {'R_ane':>7}  {'R_wef':>7}")
        print(f"  {'-'*4}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*5}  "
              f"{'-'*7}  {'-'*7}  {'-'*7}")

        corner_rows = []
        for ep in range(EVAL_EPISODES):
            eval_env = CAPQLEnv(mode='all', dssat_seed=3000 + ep,
                                run_dssat_location='run_dssat')
            obs, _ = eval_env.reset()

            eval_env.current_w = w_corner.copy()
            obs[-3:] = w_corner

            cum_daily = 0.0
            ep_len    = 0
            done      = False
            last_info = {}

            while not done:
                action = agent.select_action(obs, step=TOTAL_STEPS, deterministic=True)
                obs, r_daily, done, _, last_info = eval_env.step(action)
                obs[-3:]  = w_corner
                cum_daily += r_daily
                ep_len    += 1

            eval_env.close()

            sos = last_info.get('sos_state', {})
            rv  = last_info.get('reward_vec', np.zeros(N_OBJECTIVES))
            cum_reward = cum_daily + float(np.dot(w_corner, rv))

            row = {
                'corner':        corner_name,
                'episode':       ep + 1,
                'w_yield':       round(float(w_corner[0]), 4),
                'w_neff':        round(float(w_corner[1]), 4),
                'w_water':       round(float(w_corner[2]), 4),
                'cum_reward':    round(cum_reward, 4),
                'R_yield':       round(float(rv[0]), 4),
                'R_ane':         round(float(rv[1]), 4),
                'R_water_eff':   round(float(rv[2]), 4),
                'yield_kg_ha':   round(float(sos.get('grnwt',         0.0)), 1),
                'total_N_kg_ha': round(float(sos.get('total_nitrogen', 0.0)), 1),
                'total_W_mm':    round(float(sos.get('total_water',    0.0)), 1),
                'total_rain_mm': round(float(sos.get('total_rain',     0.0)), 1),
                'ep_length':     ep_len,
            }
            corner_rows.append(row)
            all_rows.append(row)

            print(f"  {ep+1:4d}  {cum_reward:+7.3f}  "
                  f"{row['yield_kg_ha']:6.0f}  {row['total_N_kg_ha']:5.0f}  "
                  f"{row['total_W_mm']:5.0f}  "
                  f"{row['R_yield']:+7.4f}  {row['R_ane']:+7.4f}  "
                  f"{row['R_water_eff']:+7.4f}")

        def _m(k): return np.mean([r[k] for r in corner_rows])
        print(f"\n  mean → yield={_m('yield_kg_ha'):.0f} kg/ha  "
              f"N={_m('total_N_kg_ha'):.0f}  W={_m('total_W_mm'):.0f} mm  "
              f"R_yld={_m('R_yield'):+.3f}  R_ane={_m('R_ane'):+.3f}  "
              f"R_wef={_m('R_water_eff'):+.3f}")

    print("\n" + "=" * 78)
    print(f"  {'Corner':<10}  {'yield':>7}  {'N':>5}  {'W':>5}  "
          f"{'R_yield':>8}  {'R_ane':>8}  {'R_water':>8}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*5}  {'-'*5}  "
          f"{'-'*8}  {'-'*8}  {'-'*8}")
    for corner_name in EVAL_CORNERS:
        cr = [r for r in all_rows if r['corner'] == corner_name]
        def _m(k): return np.mean([r[k] for r in cr])
        print(f"  {corner_name:<10}  {_m('yield_kg_ha'):7.0f}  "
              f"{_m('total_N_kg_ha'):5.0f}  {_m('total_W_mm'):5.0f}  "
              f"{_m('R_yield'):+8.4f}  {_m('R_ane'):+8.4f}  "
              f"{_m('R_water_eff'):+8.4f}")
    print("=" * 78)

    with open(EVAL_LOG, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n✓ Eval results → {EVAL_LOG}")


# ================================================================== #
# Plotting                                                             #
# ================================================================== #
def _rolling(arr, w=30):
    arr = np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(w) / w, mode='valid') if len(arr) >= w else arr


def _plot_training(ep_log):
    try:
        with open(ep_log) as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"  ⚠ {ep_log} not found — skipping plots")
        return
    if not rows:
        return

    eps   = [int(r['episode'])         for r in rows]
    cum_r = [float(r['cum_reward'])    for r in rows]
    yld   = [float(r['yield_kg_ha'])   for r in rows]
    n_v   = [float(r['total_N_kg_ha']) for r in rows]
    w_v   = [float(r['total_W_mm'])    for r in rows]
    r_yld = [float(r['R_yield'])       for r in rows]
    r_ane = [float(r['R_ane'])         for r in rows]

    W   = 30
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        'CAPQL v2 Training Curves — Corner Injection (20% pure corners)\n'
        f'(λ={AUG_LAMBDA}, hindsight relabeling={USE_RELABEL}, 1M steps)',
        fontsize=11, fontweight='bold'
    )

    panels = [
        (axes[0, 0], eps, cum_r, 'Cumulative Reward',  'tab:blue'),
        (axes[0, 1], eps, yld,   'Yield (kg/ha)',       'tab:green'),
        (axes[0, 2], eps, n_v,   'Total N (kg/ha)',     'tab:orange'),
        (axes[1, 0], eps, w_v,   'Total Water (mm)',    'tab:cyan'),
        (axes[1, 1], eps, r_yld, 'R_yield',             'tab:purple'),
        (axes[1, 2], eps, r_ane, 'R_ane',               'tab:red'),
    ]
    for ax, x, y, title, color in panels:
        ax.plot(x, y, alpha=0.2, color=color, linewidth=0.5)
        roll  = _rolling(y, W)
        x_r   = x[W - 1:] if len(x) >= W else x
        ax.plot(x_r, roll, color=color, linewidth=1.8, label=f'rolling({W})')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Episode', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_TRAIN, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Training plot → {PLOT_TRAIN}")

    c_loss = [float(r['critic_loss']) for r in rows if r['critic_loss']]
    a_loss = [float(r['actor_loss'])  for r in rows
              if r['actor_loss'] not in ('', 'nan')]
    if c_loss:
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
        fig2.suptitle('CAPQL v2 Loss Curves', fontsize=12, fontweight='bold')
        axes2[0].plot(c_loss, color='tab:blue',   linewidth=0.8, alpha=0.7)
        axes2[0].set_title('Critic Loss', fontsize=10)
        axes2[0].set_xlabel('Episode', fontsize=8)
        axes2[0].grid(True, alpha=0.3)
        axes2[1].plot(a_loss, color='tab:orange', linewidth=0.8, alpha=0.7)
        axes2[1].set_title('Actor Loss', fontsize=10)
        axes2[1].set_xlabel('Episode', fontsize=8)
        axes2[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOT_LOSSES, dpi=120, bbox_inches='tight')
        plt.close(fig2)
        print(f"✓ Loss plot     → {PLOT_LOSSES}")


# ================================================================== #
# Print helpers                                                        #
# ================================================================== #
def _print_config():
    W = 78
    print("=" * W)
    print("CAPQL v2 — Concave-augmented Pareto Q-Learning  (corner injection)")
    print("=" * W)
    print(f"  {'Backbone':<26} TD3  (Twin Delayed Deep Deterministic PG)")
    print(f"  {'Objectives':<26} [R_yield, R_ane, R_water_eff]")
    print(f"  {'Preference dist.':<26} 20% pure corners + 80% Dirichlet([1,1,1])")
    print(f"  {'Corner injection':<26} 6.7% yield / 6.7% n_eff / 6.7% water")
    print(f"  {'Augmentation':<26} λ·mean(log(Rᵢ + 1 + ε))")
    print(f"  {'AUG_LAMBDA  λ':<26} {AUG_LAMBDA}")
    print(f"  {'AUG_EPS     ε':<26} {AUG_EPS}")
    print(f"  {'Hindsight relabeling':<26} {USE_RELABEL}")
    print()
    print(f"  {'total_steps':<26} {TOTAL_STEPS:,}")
    print(f"  {'warmup_steps':<26} {WARMUP_STEPS:,}")
    print(f"  {'buffer_size':<26} {BUFFER_SIZE:,}")
    print(f"  {'batch_size':<26} {BATCH_SIZE}")
    print(f"  {'lr_actor / lr_critic':<26} {LR_ACTOR} / {LR_CRITIC}")
    print(f"  {'gamma':<26} {GAMMA}")
    print(f"  {'tau':<26} {TAU}")
    print(f"  {'policy_freq':<26} {POLICY_FREQ}")
    print(f"  {'hidden_size':<26} {HIDDEN}")
    print(f"  {'device':<26} {DEVICE}")
    print(f"  {'model_dir':<26} {MODEL_DIR}")
    print("=" * W + "\n")


def _print_header():
    print(f"  {'ep':>5}  {'step':>8}  {'len':>4}  {'cum_R':>7}  "
          f"{'w':>14}  {'yield':>6}  {'N':>5}  {'W':>5}  "
          f"{'R_yld':>7}  {'R_ane':>7}  {'R_wef':>7}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*4}  {'-'*7}  "
          f"{'-'*14}  {'-'*6}  {'-'*5}  {'-'*5}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}")


def _print_row(ep_count, step, row):
    w_str = f"[{row['w_yield']:.2f},{row['w_neff']:.2f},{row['w_water']:.2f}]"
    print(f"  {ep_count:5d}  {step:8d}  {row['ep_length']:4d}  "
          f"{row['cum_reward']:+7.3f}  {w_str:>14}  "
          f"{row['yield_kg_ha']:6.0f}  {row['total_N_kg_ha']:5.0f}  "
          f"{row['total_W_mm']:5.0f}  "
          f"{row['R_yield']:+7.4f}  {row['R_ane']:+7.4f}  "
          f"{row['R_water_eff']:+7.4f}")


def _print_rolling(recent):
    def rm(k): return np.mean([r[k] for r in recent])
    print(f"\n  ── rolling mean (last {len(recent)} ep) ──  "
          f"cum_R={rm('cum_reward'):+.3f}  "
          f"yield={rm('yield_kg_ha'):.0f} kg/ha  "
          f"N={rm('total_N_kg_ha'):.0f}  W={rm('total_W_mm'):.0f} mm  "
          f"R_yld={rm('R_yield'):+.3f}  "
          f"R_ane={rm('R_ane'):+.3f}  "
          f"R_wef={rm('R_water_eff'):+.3f}\n")


# ================================================================== #
# Main                                                                 #
# ================================================================== #
if __name__ == '__main__':
    try:
        agent = train()
        evaluate(agent)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise
