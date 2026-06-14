"""
14_capql_train_robust_v2.py  —  CAPQL-Robust LSTM (TD3 + episode replay + BPTT).

Architecture
────────────
  Actor   : LSTM(16→256) + head(259→256→2)      forget-gate bias = 1.0
  Critic  : twin LSTMs(16→256) + heads(259+2→256→1)
  Obs input to LSTM : crop+mask (16-dim, without w)
  w injected at heads only → hindsight relabeling without re-running LSTM

Episode Replay Buffer
─────────────────────
  Stores full episodes (T+1, 16). At sample time:
    - Resample w ~ Dirichlet(1,1,1) per episode
    - Compute augmented_reward at terminal step
    - Return (B, T+1, 16) sequences for full BPTT

Curriculum
──────────
  Phase 1 (0–500K)  : 80% clean, mild faults
  Phase 2 (500K–1.5M): 50% clean, all fault types
  Phase 3 (1.5M–3M) : 30% clean, full severity + combos

Monitoring
──────────
  CSV logs : episode_log, eval_corners_log, eval_fault_log
  Plots    : training_curves, corner_divergence, fault_resilience, lstm_diag
  Mechanistic: single-episode Fig D (stuck sensor + LSTM response)

Run inside Docker:
    cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
    /opt/gym_dssat_pdi/bin/python3 -u 14_capql_train_robust_v2.py 2>&1 | tee /workspace/train_lstm_v2.log
"""

import csv, os, sys, time, random
from collections import deque
import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from importlib import import_module

CAPQLRobustEnvV2 = import_module('capql_env_robust_v2').CAPQLRobustEnvV2

# ══════════════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════════════
TOTAL_STEPS      = 3_000_000
WARMUP_EPISODES  = 50          # random actions for first N episodes
BUFFER_MAX_EP    = 5_000       # max episodes in replay buffer
BATCH_SIZE       = 32          # episodes per gradient update
UPDATE_INTERVAL  = 4           # update every N episodes
N_UPDATES        = 4           # gradient steps per update call

LR_ACTOR   = 3e-4
LR_CRITIC  = 3e-4
GAMMA      = 0.99
TAU        = 0.005
POLICY_FREQ = 2
GRAD_CLIP  = 0.5
AUG_LAMBDA = 0.5
AUG_EPS    = 0.1

NOISE_START      = 0.30
NOISE_END        = 0.05
NOISE_DECAY_STEPS= 2_000_000
TP_STD  = np.array([8.0,  2.0], dtype=np.float32)
TP_CLIP = np.array([20.0, 5.0], dtype=np.float32)

CROP_MASK_DIM = 16
N_OBJ         = 3
OBS_DIM       = 19
ACTION_DIM    = 2
HIDDEN        = 256

ACT_LOW   = np.array([0.0,   0.0], dtype=np.float32)
ACT_HIGH  = np.array([200.0, 50.0], dtype=np.float32)
ACT_RANGE = ACT_HIGH - ACT_LOW

EVAL_INTERVAL = 20_000   # steps between eval runs
PLOT_INTERVAL = 20_000   # steps between plot refreshes
LOG_INTERVAL  = 10       # episodes between console prints
N_EVAL_EPS    = 5        # eval episodes per corner

MODEL_DIR  = '/workspace/capql_robust_lstm'
EP_LOG     = '/workspace/capql_lstm_episode_log.csv'
EVAL_COR   = '/workspace/capql_lstm_eval_corners.csv'
EVAL_FLT   = '/workspace/capql_lstm_eval_faults.csv'

DEVICE = torch.device('cpu')

EVAL_CORNERS = {
    'yield':    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    'n_eff':    np.array([0.0, 1.0, 0.0], dtype=np.float32),
    'water':    np.array([0.0, 0.0, 1.0], dtype=np.float32),
    'balanced': np.array([1/3, 1/3, 1/3], dtype=np.float32),
}

EVAL_FAULTS = [
    ('clean',           {}),
    ('noise',           {'noise_std': 0.08}),
    ('stuck_s0',        {'stuck_sensors': [0], 'stuck_day': 30}),
    ('dropout_s0',      {'dropout_rate': 0.15}),
    ('offset_high',     {'sensor_bias': {0: 0.20}}),
    ('gain_high',       {'sensor_gain': {0: 1.4}}),
    ('actuator_eff_N',  {'N_efficiency': 0.50}),
    ('actuator_eff_W',  {'W_efficiency': 0.50}),
]


# ══════════════════════════════════════════════════════════════════════
# Episode Replay Buffer
# ══════════════════════════════════════════════════════════════════════
class EpisodeBuffer:
    """
    Stores full episodes as (T+1, 16) observation sequences.
    At sample time: resample w ~ Dirichlet and compute augmented reward.
    """

    def __init__(self, max_episodes=BUFFER_MAX_EP):
        self.buffer = deque(maxlen=max_episodes)

    def push(self, obs16_full, actions, r_daily, rv, dones, fault_type='clean'):
        """
        obs16_full : (T+1, 16) — obs at t=0..T (includes terminal next_obs)
        actions    : (T, 2)
        r_daily    : (T,)
        rv         : (3,)  terminal reward vec
        dones      : (T,)  1.0 only at t=T-1
        """
        self.buffer.append({
            'obs16_full': np.asarray(obs16_full, dtype=np.float32),
            'actions':    np.asarray(actions,    dtype=np.float32),
            'r_daily':    np.asarray(r_daily,    dtype=np.float32),
            'rv':         np.asarray(rv,          dtype=np.float32),
            'dones':      np.asarray(dones,       dtype=np.float32),
            'fault_type': fault_type,
        })

    def sample(self, batch_size):
        eps = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        B   = len(eps)

        # Truncate all episodes to the shortest in the batch (avoids stack shape mismatch)
        min_len = min(e['obs16_full'].shape[0] for e in eps)   # T+1 of shortest
        T = min_len - 1

        obs16_full = np.stack([e['obs16_full'][:min_len] for e in eps])  # (B, T+1, 16)
        actions    = np.stack([e['actions'][:T]          for e in eps])  # (B, T, 2)
        r_daily    = np.stack([e['r_daily'][:T]          for e in eps])  # (B, T)
        rv         = np.stack([e['rv']                   for e in eps])  # (B, 3)
        dones      = np.stack([e['dones'][:T]            for e in eps])  # (B, T)

        # Hindsight preference relabeling: resample w per episode
        w = np.random.dirichlet(np.ones(N_OBJ), size=B).astype(np.float32)  # (B, 3)

        # Augmented reward: r_daily everywhere, terminal bonus on each episode's
        # last VALID step (identified by dones==1.0, not just the batch's last column)
        r_aug = r_daily.copy()
        scalarized = (w * rv).sum(axis=1)                              # (B,)
        concave    = AUG_LAMBDA * np.mean(
            np.log(np.clip(rv, 0.0, None) + 1.0 + AUG_EPS), axis=1)  # (B,)
        terminal_bonus = scalarized + concave                          # (B,)
        # Find the terminal step for each episode (first done=1)
        for b in range(B):
            term_idx = int(np.argmax(dones[b]))   # index of done=1
            r_aug[b, term_idx] += terminal_bonus[b]

        def tt(x): return torch.FloatTensor(x).to(DEVICE)
        return tt(obs16_full), tt(actions), tt(r_aug), tt(dones), tt(w)

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════
# Networks
# ══════════════════════════════════════════════════════════════════════
def _init_forget_gate(lstm_module, hidden):
    """Set LSTM forget-gate bias = 1 to help long-episode gradient flow."""
    for name, p in lstm_module.named_parameters():
        if 'bias' in name:
            n = p.size(0)
            nn.init.constant_(p[n // 4: n // 2], 1.0)


class LSTMActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(CROP_MASK_DIM, HIDDEN, batch_first=True)
        _init_forget_gate(self.lstm, HIDDEN)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN + N_OBJ, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, ACTION_DIM),     nn.Tanh(),
        )
        self.register_buffer('act_mid',  torch.FloatTensor((ACT_HIGH + ACT_LOW) / 2.0))
        self.register_buffer('act_half', torch.FloatTensor(ACT_RANGE / 2.0))

    def forward_sequence(self, obs16_full, w):
        """
        obs16_full : (B, T+1, 16)
        w          : (B, 3)
        Returns actions (B, T, 2) and h_full (B, T+1, H).
        Actions at step t use hidden state computed after seeing obs[t].
        """
        h_full, _ = self.lstm(obs16_full)            # (B, T+1, H)
        T = obs16_full.shape[1] - 1
        h_curr = h_full[:, :T, :]                    # (B, T, H)
        w_exp  = w.unsqueeze(1).expand(-1, T, -1)    # (B, T, 3)
        raw    = self.head(torch.cat([h_curr, w_exp], dim=-1))  # (B, T, 2)
        return raw * self.act_half + self.act_mid, h_full

    def step(self, obs16_np, w_np, h, c):
        """Single env step — maintains (h, c) across steps."""
        with torch.no_grad():
            x = torch.FloatTensor(obs16_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
            w = torch.FloatTensor(w_np).unsqueeze(0).to(DEVICE)
            lstm_out, (hn, cn) = self.lstm(x, (h, c))
            combined = torch.cat([lstm_out, w.unsqueeze(1)], dim=-1)
            raw   = self.head(combined).squeeze()
            action = (raw * self.act_half + self.act_mid).cpu().numpy()
        return action.astype(np.float32), hn, cn


class LSTMTwinCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(CROP_MASK_DIM, HIDDEN, batch_first=True)
        self.lstm2 = nn.LSTM(CROP_MASK_DIM, HIDDEN, batch_first=True)
        _init_forget_gate(self.lstm1, HIDDEN)
        _init_forget_gate(self.lstm2, HIDDEN)
        in_dim = HIDDEN + N_OBJ + ACTION_DIM
        self.q1_head = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)
        )
        self.q2_head = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)
        )

    def forward_sequence(self, obs16_full, w, actions):
        """
        obs16_full : (B, T+1, 16)  — full sequence including terminal obs
        w          : (B, 3)
        actions    : (B, T, 2)     — actions taken (current for critic, target for bootstrap)
        Returns (q1, q2) each (B, T, 1) using hidden states at current steps.
        """
        h1_full, _ = self.lstm1(obs16_full)   # (B, T+1, H)
        h2_full, _ = self.lstm2(obs16_full)
        T  = actions.shape[1]
        h1 = h1_full[:, :T, :]
        h2 = h2_full[:, :T, :]
        w_exp = w.unsqueeze(1).expand(-1, T, -1)
        c1 = torch.cat([h1, w_exp, actions], dim=-1)
        c2 = torch.cat([h2, w_exp, actions], dim=-1)
        return self.q1_head(c1), self.q2_head(c2)

    def next_q(self, obs16_full, w, next_actions):
        """
        Compute Q at NEXT steps using h_{t+1} hidden states.
        next_actions : (B, T, 2)  — target actor's actions at s_{t+1}
        """
        h1_full, _ = self.lstm1(obs16_full)
        h2_full, _ = self.lstm2(obs16_full)
        T = next_actions.shape[1]
        h1_next = h1_full[:, 1:, :]   # h after seeing obs[1], ..., obs[T]
        h2_next = h2_full[:, 1:, :]
        w_exp   = w.unsqueeze(1).expand(-1, T, -1)
        c1 = torch.cat([h1_next, w_exp, next_actions], dim=-1)
        c2 = torch.cat([h2_next, w_exp, next_actions], dim=-1)
        return self.q1_head(c1), self.q2_head(c2)


def soft_update(net, tgt, tau=TAU):
    for p, tp in zip(net.parameters(), tgt.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


# ══════════════════════════════════════════════════════════════════════
# TD3 Update (full BPTT over episode)
# ══════════════════════════════════════════════════════════════════════
_act_low_t  = torch.FloatTensor(ACT_LOW).to(DEVICE)
_act_high_t = torch.FloatTensor(ACT_HIGH).to(DEVICE)


def update(batch, actor, actor_tgt, critic, critic_tgt,
           actor_opt, critic_opt, update_count):

    obs16_full, actions, r_aug, dones, w = batch
    B, Tp1, _ = obs16_full.shape
    T = Tp1 - 1

    # ── Critic update ──────────────────────────────────────────────
    with torch.no_grad():
        # Target actor: get next actions at each step
        next_acts_raw, _ = actor_tgt.forward_sequence(obs16_full, w)
        # next_acts_raw is (B, T, 2) based on h_curr (h at obs[0]..obs[T-1])
        # We want actions at s_{t+1}, so use next-step hidden states
        # Re-run target actor using next-step hidden states:
        h_full_tgt, _ = actor_tgt.lstm(obs16_full)   # (B, T+1, H)
        h_next_tgt    = h_full_tgt[:, 1:, :]          # (B, T, H)
        w_exp = w.unsqueeze(1).expand(-1, T, -1)
        raw_next = actor_tgt.head(torch.cat([h_next_tgt, w_exp], -1))
        next_acts = raw_next * actor_tgt.act_half + actor_tgt.act_mid

        # TD3 target policy smoothing
        n = torch.zeros_like(next_acts)
        n[:, :, 0] = (torch.randn(B, T, device=DEVICE)
                      * TP_STD[0]).clamp(-TP_CLIP[0], TP_CLIP[0])
        n[:, :, 1] = (torch.randn(B, T, device=DEVICE)
                      * TP_STD[1]).clamp(-TP_CLIP[1], TP_CLIP[1])
        next_acts = (next_acts + n).clamp(_act_low_t, _act_high_t)

        # Target Q at next steps
        tq1, tq2 = critic_tgt.next_q(obs16_full, w, next_acts)
        tq = torch.min(tq1, tq2).squeeze(-1)           # (B, T)
        target_q = r_aug + GAMMA * (1.0 - dones) * tq  # (B, T)

    q1, q2 = critic.forward_sequence(obs16_full, w, actions)  # (B, T, 1)
    critic_loss = (F.mse_loss(q1.squeeze(-1), target_q) +
                   F.mse_loss(q2.squeeze(-1), target_q))
    critic_opt.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
    critic_opt.step()

    # ── Actor update (delayed) ─────────────────────────────────────
    actor_loss_val = None
    if update_count % POLICY_FREQ == 0:
        curr_acts, h_full = actor.forward_sequence(obs16_full, w)   # (B, T, 2)
        h_curr = h_full[:, :T, :]
        w_exp2 = w.unsqueeze(1).expand(-1, T, -1)
        q_actor = critic.q1_head(
            torch.cat([h_curr, w_exp2, curr_acts], dim=-1)
        )
        actor_loss_val = -q_actor.mean()
        actor_opt.zero_grad()
        actor_loss_val.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
        actor_opt.step()
        soft_update(actor, actor_tgt)
        soft_update(critic, critic_tgt)
        actor_loss_val = actor_loss_val.item()

    return critic_loss.item(), actor_loss_val


# ══════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════
def _run_eval_episode(actor, env, w_corner, fault_kwargs=None):
    """Run one deterministic episode. Returns (rv, sos_info)."""
    from capql_env_robust_v2 import _FaultStateV3

    env.current_w  = w_corner.copy()
    env._max_anfer = float(np.clip(200.0 * (1.0 - w_corner[1])**2, 2.0, 200.0))
    env._max_amir  = float(np.clip( 50.0 * (1.0 - w_corner[2])**2, 3.0,  50.0))

    # Override fault state
    if fault_kwargs is not None:
        env._fault_state = _FaultStateV3(fault_kwargs)

    sos_obs = env._sos_env.reset()
    raw_obs = env._encode(sos_obs)
    obs19   = env._fault_state.corrupt(raw_obs) if env._fault_state else raw_obs
    obs19[-N_OBJ:] = w_corner

    h = torch.zeros(1, 1, HIDDEN, device=DEVICE)
    c = torch.zeros(1, 1, HIDDEN, device=DEVICE)
    done = False
    last_info = {}

    while not done:
        obs16 = obs19[:CROP_MASK_DIM]
        action, h, c = actor.step(obs16, w_corner, h, c)
        action = np.clip(action, [0, 0], [env._max_anfer, env._max_amir])
        obs19, _, done, _, last_info = env.step(action)
        obs19[-N_OBJ:] = w_corner

    rv  = np.array(last_info.get('reward_vec', [0, 0, 0]), dtype=np.float32)
    sos = last_info.get('sos_state', {})
    return rv, sos


def eval_corners(actor, total_steps):
    """Return list of result dicts for all corners (clean episodes)."""
    results = []
    env = CAPQLRobustEnvV2(dssat_seed=9999)
    env.curriculum_phase = 3   # ensure env exists without auto-reset fault sampling
    env.reset()

    for corner_name, w_corner in EVAL_CORNERS.items():
        for seed in range(9000, 9000 + N_EVAL_EPS):
            env._sos_env._seed = seed
            from capql_env_robust_v2 import _FaultStateV3
            env._fault_state = _FaultStateV3({})   # clean
            rv, sos = _run_eval_episode(actor, env, w_corner)
            results.append({
                'total_steps':   total_steps,
                'corner':        corner_name,
                'R_yield':       float(rv[0]),
                'R_ane':         float(rv[1]),
                'R_water_eff':   float(rv[2]),
                'yield_kg_ha':   float(sos.get('grnwt', 0)),
                'total_N_kg_ha': float(sos.get('total_nitrogen', 0)),
                'total_W_mm':    float(sos.get('total_water', 0)),
            })
    env.close()
    return results


def eval_faults(actor, total_steps):
    """Return list of result dicts for each fault type × balanced corner."""
    results = []
    w_balanced = EVAL_CORNERS['balanced']
    env = CAPQLRobustEnvV2(dssat_seed=9998)
    env.curriculum_phase = 3
    env.reset()

    for fault_name, fault_kwargs in EVAL_FAULTS:
        for seed in range(8000, 8000 + N_EVAL_EPS):
            env._sos_env._seed = seed
            from capql_env_robust_v2 import _FaultStateV3
            env._fault_state = _FaultStateV3(fault_kwargs)
            rv, sos = _run_eval_episode(actor, env, w_balanced, fault_kwargs)
            results.append({
                'total_steps': total_steps,
                'fault':       fault_name,
                'R_yield':     float(rv[0]),
                'R_ane':       float(rv[1]),
                'R_water_eff': float(rv[2]),
                'yield_kg_ha': float(sos.get('grnwt', 0)),
            })
    env.close()
    return results


# ══════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════
def plot_training(ep_rows, out='/workspace/plot_lstm_training.png'):
    if len(ep_rows) < 5:
        return
    clean  = [r for r in ep_rows if r['is_clean']]
    faulted= [r for r in ep_rows if not r['is_clean']]

    def smooth(vals, k=50):
        out_ = []
        for i in range(len(vals)):
            out_.append(np.mean(vals[max(0, i - k):i + 1]))
        return out_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, rows, title in [
        (axes[0], clean,   'Clean episodes'),
        (axes[1], faulted, 'Faulted episodes'),
    ]:
        if not rows:
            ax.set_title(f'{title} — no data')
            continue
        steps  = [r['total_steps'] for r in rows]
        ry     = [r['R_yield']     for r in rows]
        ra     = [r['R_ane']       for r in rows]
        rw     = [r['R_water_eff'] for r in rows]
        ax.plot(steps, smooth(ry), label='R_yield',     color='#27ae60', lw=1.5)
        ax.plot(steps, smooth(ra), label='R_ane',       color='#e67e22', lw=1.5)
        ax.plot(steps, smooth(rw), label='R_water',     color='#2980b9', lw=1.5)
        ax.axhline(0, color='#aaa', lw=0.7, ls='--')
        ax.set_xlabel('Total steps')
        ax.set_ylabel('Reward component (smoothed)')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        # Shade curriculum phases
        for boundary, label, color in [
            (500_000,   'P1→P2', '#f39c12'),
            (1_500_000, 'P2→P3', '#c0392b'),
        ]:
            ax.axvline(boundary, color=color, lw=1.0, ls=':', alpha=0.7)
            ax.text(boundary, ax.get_ylim()[0], label,
                    fontsize=7, color=color, va='bottom', ha='right')

    fig.suptitle('CAPQL-Robust LSTM — Training Curves', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_corners(corner_rows, out='/workspace/plot_lstm_corners.png'):
    if not corner_rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['R_yield', 'R_ane', 'R_water_eff']
    colors  = {'yield': '#27ae60', 'n_eff': '#e67e22',
               'water': '#2980b9', 'balanced': '#8e44ad'}

    for ax, metric in zip(axes, metrics):
        for corner, clr in colors.items():
            rows = [r for r in corner_rows if r['corner'] == corner]
            if not rows:
                continue
            steps = sorted(set(r['total_steps'] for r in rows))
            means = []
            for s in steps:
                sub = [r[metric] for r in rows if r['total_steps'] == s]
                means.append(np.mean(sub))
            ax.plot(steps, means, color=clr, lw=2.0, marker='o', ms=4,
                    label=corner)
        ax.axhline(0, color='#aaa', lw=0.7, ls='--')
        ax.set_xlabel('Total steps')
        ax.set_ylabel(metric)
        ax.set_title(f'Corner divergence — {metric}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle('CAPQL-Robust LSTM — Preference Corner Divergence\n'
                 '(corners should stay separated — collapse = multi-objective failure)',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_faults(fault_rows, out='/workspace/plot_lstm_faults.png'):
    if not fault_rows:
        return
    fault_names = list(dict.fromkeys(r['fault'] for r in fault_rows))
    fig, ax = plt.subplots(figsize=(14, 5))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(fault_names)))

    for i, fname in enumerate(fault_names):
        rows  = [r for r in fault_rows if r['fault'] == fname]
        steps = sorted(set(r['total_steps'] for r in rows))
        means = [np.mean([r['R_yield'] for r in rows if r['total_steps'] == s])
                 for s in steps]
        ls = '-' if fname == 'clean' else '--'
        lw = 2.5 if fname == 'clean' else 1.5
        ax.plot(steps, means, color=colors[i], lw=lw, ls=ls,
                marker='o', ms=4, label=fname)

    ax.axhline(0, color='#aaa', lw=0.7, ls=':')
    ax.set_xlabel('Total steps')
    ax.set_ylabel('R_yield (mean over 5 eps)')
    ax.set_title('CAPQL-Robust LSTM — Fault Resilience Progress\n'
                 '(balanced corner, medium severity faults)',
                 fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_lstm_diag(diag_rows, out='/workspace/plot_lstm_diag.png'):
    if not diag_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    steps  = [r['total_steps'] for r in diag_rows]
    h_norm = [r['h_norm']      for r in diag_rows]
    ep_len = [r['ep_len']      for r in diag_rows]

    axes[0].plot(steps, h_norm, color='#c0392b', lw=1.2, alpha=0.6)
    axes[0].set_xlabel('Total steps')
    axes[0].set_ylabel('||h_T|| (mean last hidden state norm)')
    axes[0].set_title('LSTM hidden state norm')
    axes[0].grid(alpha=0.2)

    axes[1].plot(steps, ep_len, color='#2980b9', lw=1.2, alpha=0.6)
    axes[1].set_xlabel('Total steps')
    axes[1].set_ylabel('Episode length (steps)')
    axes[1].set_title('Episode length over training')
    axes[1].grid(alpha=0.2)

    fig.suptitle('CAPQL-Robust LSTM — Diagnostics', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_mechanistic(actor, out='/workspace/plot_lstm_mechanistic.png'):
    """
    Figure D: single episode with stuck sensor at day 45.
    Plots true vs faulted obs[7], LSTM h_norm, irrigation action, grnwt.
    """
    from capql_env_robust_v2 import _FaultStateV3

    env = CAPQLRobustEnvV2(dssat_seed=7777)
    env.curriculum_phase = 3
    env.reset()
    w_corner = EVAL_CORNERS['balanced']

    # Episode with stuck soil moisture sensor at day 45
    env._fault_state = _FaultStateV3({'stuck_sensors': [0], 'stuck_day': 45})
    env.current_w = w_corner.copy()

    sos_obs = env._sos_env.reset()
    raw_obs = env._encode(sos_obs)
    obs19   = env._fault_state.corrupt(raw_obs)
    obs19[-N_OBJ:] = w_corner

    h = torch.zeros(1, 1, HIDDEN, device=DEVICE)
    c = torch.zeros(1, 1, HIDDEN, device=DEVICE)

    days, sw_true, sw_obs, amir_hist, grnwt_hist, h_norm_hist = \
        [], [], [], [], [], []
    done = False
    day  = 0

    while not done:
        days.append(day)
        obs16 = obs19[:CROP_MASK_DIM]
        sw_obs.append(float(obs16[7]))

        action, h, c = actor.step(obs16, w_corner, h, c)
        action = np.clip(action, [0, 0],
                         [env._max_anfer, env._max_amir])

        h_norm_hist.append(torch.norm(h).item())
        amir_hist.append(float(action[1]))

        obs19, _, done, _, info = env.step(action)
        obs19[-N_OBJ:] = w_corner

        sos = info.get('sos_state', {})
        grnwt_hist.append(float(sos.get('grnwt', 0)))

        # True sw_mean from info if available, else use obs before fault
        sw_true.append(float(sos.get('sw_mean', obs19[7])))
        day += 1

    env.close()

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(days, sw_true, color='#2980b9', lw=2, label='True sw_mean')
    axes[0].plot(days, sw_obs,  color='#e74c3c', lw=1.5, ls='--',
                 label='Observed (faulted)')
    axes[0].axvline(45, color='#c0392b', lw=1.5, ls=':', label='Stuck at day 45')
    axes[0].set_ylabel('Soil moisture (norm.)')
    axes[0].set_title('(a) True vs Observed Soil Moisture', fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].plot(days, h_norm_hist, color='#8e44ad', lw=1.5)
    axes[1].axvline(45, color='#c0392b', lw=1.5, ls=':')
    axes[1].set_ylabel('||h_t||')
    axes[1].set_title('(b) LSTM Hidden State Norm — should change pattern after stuck')
    axes[1].grid(alpha=0.2)

    axes[2].bar(days, amir_hist, color='#3498db', alpha=0.7, width=0.8)
    axes[2].axvline(45, color='#c0392b', lw=1.5, ls=':')
    axes[2].set_ylabel('Irrigation (mm)')
    axes[2].set_title('(c) Irrigation Commands — should stabilise to stage-based after stuck')
    axes[2].grid(alpha=0.2, axis='y')

    axes[3].plot(days, grnwt_hist, color='#27ae60', lw=2)
    axes[3].axvline(45, color='#c0392b', lw=1.5, ls=':')
    axes[3].set_xlabel('Day of season')
    axes[3].set_ylabel('Grain weight (norm.)')
    axes[3].set_title('(d) Cumulative Grain Weight — resilient agent limits damage')
    axes[3].grid(alpha=0.2)

    fig.suptitle(
        'Figure D — LSTM Response to Stuck Sensor (sensor_0 = sw_mean, stuck at day 45)\n'
        'Dashed red = fault active | vertical line = fault onset',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓  Mechanistic plot → {out}')


# ══════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print('=' * 72)
    print('14_capql_train_robust_v2.py  —  CAPQL-Robust LSTM')
    print('=' * 72)
    print(f'  Actor   : LSTM({CROP_MASK_DIM}→{HIDDEN}) + head({HIDDEN}+{N_OBJ}→{HIDDEN}→{ACTION_DIM})')
    print(f'  Critic  : twin LSTMs + heads')
    print(f'  Buffer  : {BUFFER_MAX_EP} episodes  |  batch {BATCH_SIZE} eps')
    print(f'  Steps   : {TOTAL_STEPS:,}  |  curriculum 3-phase')
    print(f'  BPTT    : full episode (T≈160), grad_clip={GRAD_CLIP}')
    print(f'  Hindsight w ~ Dirichlet resampled at sample time')
    print('=' * 72 + '\n')

    actor      = LSTMActor().to(DEVICE)
    actor_tgt  = LSTMActor().to(DEVICE)
    actor_tgt.load_state_dict(actor.state_dict())
    critic     = LSTMTwinCritic().to(DEVICE)
    critic_tgt = LSTMTwinCritic().to(DEVICE)
    critic_tgt.load_state_dict(critic.state_dict())

    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=LR_ACTOR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    buffer = EpisodeBuffer(BUFFER_MAX_EP)
    env    = CAPQLRobustEnvV2(dssat_seed=1)

    # CSV logging
    ep_log_f  = open(EP_LOG,   'w', newline='', buffering=1)
    cor_log_f = open(EVAL_COR, 'w', newline='', buffering=1)
    flt_log_f = open(EVAL_FLT, 'w', newline='', buffering=1)

    ep_fields  = ['episode','total_steps','phase','fault_type','is_clean',
                  'R_yield','R_ane','R_water_eff','yield_kg_ha',
                  'total_N_kg_ha','total_W_mm','w_yield','w_neff','w_water',
                  'ep_len','h_norm']
    cor_fields = ['total_steps','corner','R_yield','R_ane','R_water_eff',
                  'yield_kg_ha','total_N_kg_ha','total_W_mm']
    flt_fields = ['total_steps','fault','R_yield','R_ane','R_water_eff','yield_kg_ha']

    ep_w  = csv.DictWriter(ep_log_f,  fieldnames=ep_fields);  ep_w.writeheader()
    cor_w = csv.DictWriter(cor_log_f, fieldnames=cor_fields); cor_w.writeheader()
    flt_w = csv.DictWriter(flt_log_f, fieldnames=flt_fields); flt_w.writeheader()

    # In-memory lists for periodic plotting
    ep_rows, corner_rows, fault_rows, diag_rows = [], [], [], []

    total_steps  = 0
    ep_count     = 0
    update_count = 0
    t0           = time.time()
    last_eval    = 0
    last_plot    = 0

    print(f"  {'ep':>5}  {'step':>9}  {'phase':>5}  {'fault_type':<18}"
          f"  {'R_yld':>7}  {'R_ane':>7}  {'R_wef':>7}  {'yield':>7}  ETA")
    print('  ' + '─' * 80)

    obs, info = env.reset()
    env.curriculum_phase = 1

    while total_steps < TOTAL_STEPS:
        phase = (1 if total_steps < 500_000 else
                 2 if total_steps < 1_500_000 else 3)
        env.curriculum_phase = phase

        w           = info['preference_w'].copy()
        fault_type  = info.get('fault_type', 'clean')

        # LSTM hidden state reset at episode start
        h = torch.zeros(1, 1, HIDDEN, device=DEVICE)
        c = torch.zeros(1, 1, HIDDEN, device=DEVICE)

        obs16 = obs[:CROP_MASK_DIM]

        ep_obs16  = [obs16.copy()]
        ep_acts   = []
        ep_rdaily = []
        ep_dones  = []
        ep_rv     = np.zeros(N_OBJ, dtype=np.float32)
        done      = False
        ep_steps  = 0

        while not done:
            if ep_count < WARMUP_EPISODES:
                action_np = env.action_space.sample()
            else:
                action_np, h, c = actor.step(obs16, w, h, c)
                frac  = min(1.0, total_steps / NOISE_DECAY_STEPS)
                sigma = NOISE_START + frac * (NOISE_END - NOISE_START)
                noise = np.random.normal(0, sigma, ACTION_DIM) * ACT_RANGE
                action_np = np.clip(action_np + noise, ACT_LOW, ACT_HIGH)

            next_obs, r_daily, done, _, step_info = env.step(action_np)
            reward_vec = np.array(step_info.get('reward_vec', np.zeros(N_OBJ)),
                                  dtype=np.float32)

            next_obs16 = next_obs[:CROP_MASK_DIM]
            ep_obs16.append(next_obs16.copy())
            ep_acts.append(action_np.copy())
            ep_rdaily.append(float(r_daily))
            ep_dones.append(float(done))
            if done:
                ep_rv = reward_vec

            obs     = next_obs
            obs16   = next_obs16
            total_steps += 1
            ep_steps    += 1

        h_norm = float(torch.norm(h).item())
        buffer.push(
            np.array(ep_obs16),
            np.array(ep_acts),
            np.array(ep_rdaily),
            ep_rv,
            np.array(ep_dones),
            fault_type,
        )
        ep_count += 1

        # Gradient updates
        if len(buffer) >= BATCH_SIZE and ep_count % UPDATE_INTERVAL == 0:
            for _ in range(N_UPDATES):
                batch = buffer.sample(BATCH_SIZE)
                c_loss, a_loss = update(
                    batch, actor, actor_tgt, critic, critic_tgt,
                    actor_opt, critic_opt, update_count
                )
                update_count += 1

        # Logging
        sos = step_info.get('sos_state', {})
        row = {
            'episode':      ep_count,
            'total_steps':  total_steps,
            'phase':        phase,
            'fault_type':   fault_type,
            'is_clean':     int(fault_type == 'clean'),
            'R_yield':      float(ep_rv[0]),
            'R_ane':        float(ep_rv[1]),
            'R_water_eff':  float(ep_rv[2]),
            'yield_kg_ha':  float(sos.get('grnwt', 0)),
            'total_N_kg_ha':float(sos.get('total_nitrogen', 0)),
            'total_W_mm':   float(sos.get('total_water', 0)),
            'w_yield':      float(w[0]),
            'w_neff':       float(w[1]),
            'w_water':      float(w[2]),
            'ep_len':       ep_steps,
            'h_norm':       h_norm,
        }
        ep_w.writerow(row)
        ep_rows.append(row)
        diag_rows.append({'total_steps': total_steps,
                          'h_norm': h_norm, 'ep_len': ep_steps})

        if ep_count % LOG_INTERVAL == 0:
            elapsed = (time.time() - t0) / 60.0
            eta     = elapsed / max(total_steps, 1) * (TOTAL_STEPS - total_steps)
            print(
                f"  {ep_count:>5}  {total_steps:>9,}  P{phase:>1}  "
                f"  {fault_type:<18}"
                f"  {ep_rv[0]:>+7.3f}  {ep_rv[1]:>+7.3f}  {ep_rv[2]:>+7.3f}"
                f"  {float(sos.get('grnwt',0)):>7.0f}"
                f"  ETA {eta:.1f}m",
                flush=True,
            )

        # Periodic evaluation
        if total_steps - last_eval >= EVAL_INTERVAL and ep_count >= WARMUP_EPISODES:
            last_eval = total_steps
            print(f'\n  ── Eval at step {total_steps:,} ──')
            try:
                c_res = eval_corners(actor, total_steps)
                for r in c_res:
                    cor_w.writerow(r)
                corner_rows.extend(c_res)
                # Print corner summary
                for cn in EVAL_CORNERS:
                    sub = [r for r in c_res if r['corner'] == cn]
                    ry = np.mean([r['R_yield'] for r in sub])
                    ra = np.mean([r['R_ane']   for r in sub])
                    rw = np.mean([r['R_water_eff'] for r in sub])
                    print(f'    {cn:<10} R_yield={ry:+.3f} R_ane={ra:+.3f}'
                          f' R_water={rw:+.3f}')

                f_res = eval_faults(actor, total_steps)
                for r in f_res:
                    flt_w.writerow(r)
                fault_rows.extend(f_res)
                print(f'  ── Fault eval done ──\n')
            except Exception as ex:
                print(f'  [eval error] {ex}', flush=True)

        # Periodic plots
        if total_steps - last_plot >= PLOT_INTERVAL:
            last_plot = total_steps
            try:
                plot_training(ep_rows)
                plot_corners(corner_rows)
                plot_faults(fault_rows)
                plot_lstm_diag(diag_rows)
                print(f'  ✓  Plots refreshed at step {total_steps:,}', flush=True)
            except Exception as ex:
                print(f'  [plot error] {ex}', flush=True)

        # Reset for next episode
        obs, info = env.reset()

    # ── End of training ────────────────────────────────────────────
    env.close()
    ep_log_f.close()
    cor_log_f.close()
    flt_log_f.close()

    torch.save(actor.state_dict(),  os.path.join(MODEL_DIR, 'actor.pt'))
    torch.save(critic.state_dict(), os.path.join(MODEL_DIR, 'critic.pt'))
    print(f'\n  ✓  Model saved → {MODEL_DIR}')
    print(f'  ✓  Training time: {(time.time()-t0)/60:.1f} min')

    # Final plots
    plot_training(ep_rows)
    plot_corners(corner_rows)
    plot_faults(fault_rows)
    plot_lstm_diag(diag_rows)

    # Mechanistic Fig D
    try:
        plot_mechanistic(actor)
    except Exception as ex:
        print(f'  [mechanistic plot error] {ex}')

    return actor


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    try:
        train()
        print('\n' + '=' * 72)
    except Exception as e:
        import traceback
        print(f'\n  ERROR: {type(e).__name__}: {e}', file=sys.stderr)
        traceback.print_exc()
        raise
