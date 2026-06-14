"""
16_capql_lstm_v2.py  —  CAPQL-LSTM v2 (truncated BPTT K=20, w at every step)

Architecture
────────────
  Actor  : LSTM(19→256) + head(259→256→2)
  Critic : twin LSTM(19→256) + heads(261→256→1)
  LSTM input = [obs16(16), w(3)] at every timestep
  → LSTM learns preference-conditioned temporal fault patterns
  → hindsight w resampled at train time, LSTM re-run with new w (K=20, cheap)

Truncated BPTT K=20
────────────────────
  Transition buffer stores K=20-step obs16 window per transition.
  Update every 4 env steps × 4 grad steps = ~3M gradient steps total.
  (Previous LSTM: 75K steps → 40× improvement)

2-fault curriculum
──────────────────
  Phase 1 (0–1M steps)  : 85% clean, 15% faulted (mild severity)
  Phase 2 (1M–3M steps) : 70% clean, 30% faulted (full severity)
  Faults: sensor_noise (σ=0.02–0.12) | sensor_stuck (sensors 0–2, day 10–120)

Preference conditioning
───────────────────────
  CORNER_PROB = 0.50 → 50% of episodes use one of 4 fixed corners
  4 corners: yield / n_eff / water / balanced

Run inside Docker:
    cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
    /opt/gym_dssat_pdi/bin/python3 -u 16_capql_lstm_v2.py 2>&1 | tee /workspace/train_lstm_v2.log
"""

import copy, csv, os, sys, time, random
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

# ── imports from existing modules ─────────────────────────────────────────
_robust_mod   = import_module('capql_env_robust_v2')
CAPQLRobustEnvV2 = _robust_mod.CAPQLRobustEnvV2
_FaultStateV3    = _robust_mod._FaultStateV3

# ══════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════
_WS = '/workspace' if os.path.isdir('/workspace') else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'capql_lstm_v2_out')

MODEL_DIR = os.path.join(_WS, 'capql_lstm_v2')
EP_LOG    = os.path.join(_WS, 'capql_lstm_v2_episodes.csv')
EVAL_LOG  = os.path.join(_WS, 'capql_lstm_v2_eval.csv')
PLOT_DIR  = _WS

os.makedirs(MODEL_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════════════
TOTAL_STEPS     = 3_000_000
K_HISTORY       = 20          # truncated BPTT window
BATCH_SIZE      = 256
BUFFER_SIZE     = 200_000
UPDATE_INTERVAL = 4            # env steps between updates
N_UPDATES       = 4            # gradient steps per update call
WARMUP_STEPS    = 5_000

CORNER_PROB     = 0.50         # fraction of episodes using fixed corners

LR_ACTOR        = 3e-4
LR_CRITIC       = 3e-4
GAMMA           = 0.99
TAU             = 0.005
ACTOR_UPDATE_FREQ = 2
GRAD_CLIP       = 0.5
AUG_LAMBDA      = 0.5
AUG_EPS         = 0.1

NOISE_START     = 0.30
NOISE_END       = 0.05
NOISE_DECAY     = 2_000_000

# TD3 target policy smoothing (absolute action units)
TP_STD  = np.array([8.0,  2.0], dtype=np.float32)
TP_CLIP = np.array([20.0, 5.0], dtype=np.float32)

OBS16_DIM = 16    # crop(11) + mask(5)
N_OBJ     = 3
LSTM_IN   = OBS16_DIM + N_OBJ   # 19  (w concat at every step)
HIDDEN    = 256
ACT_DIM   = 2

ACT_LOW   = np.array([0.0,    0.0], dtype=np.float32)
ACT_HIGH  = np.array([200.0, 50.0], dtype=np.float32)
ACT_MID   = (ACT_HIGH + ACT_LOW) / 2.0
ACT_HALF  = (ACT_HIGH - ACT_LOW) / 2.0

EVAL_INTERVAL = 50_000
N_EVAL_EPS    = 5
LOG_INTERVAL  = 20     # episodes between console prints

DEVICE = torch.device('cpu')

EVAL_CORNERS = {
    'yield':    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    'n_eff':    np.array([0.0, 1.0, 0.0], dtype=np.float32),
    'water':    np.array([0.0, 0.0, 1.0], dtype=np.float32),
    'balanced': np.array([1/3, 1/3, 1/3], dtype=np.float32),
}

_CORNERS_LIST = [
    np.array([1.0, 0.0, 0.0], dtype=np.float32),  # yield
    np.array([0.0, 1.0, 0.0], dtype=np.float32),  # n_eff
    np.array([0.0, 0.0, 1.0], dtype=np.float32),  # water
    np.array([1/3, 1/3, 1/3], dtype=np.float32),  # balanced
]


# ══════════════════════════════════════════════════════════════════════
# 2-fault environment (subclass of CAPQLRobustEnvV2)
# ══════════════════════════════════════════════════════════════════════
class TwoFaultEnv(CAPQLRobustEnvV2):
    """
    CAPQLRobustEnvV2 with only sensor_noise + sensor_stuck.
    2-phase curriculum: Phase 1 mild (0–1M), Phase 2 full (1M–3M).
    50% of episodes use a fixed preference corner.
    """

    def reset(self, *, seed=None, options=None):
        phase = 1 if self.total_steps < 1_000_000 else 2

        # Preference weight: 50% fixed corner, 50% random Dirichlet
        if np.random.random() < CORNER_PROB:
            self.current_w = random.choice(_CORNERS_LIST).copy()
        else:
            self.current_w = np.random.dirichlet(
                np.ones(N_OBJ)).astype(np.float32)

        # Per-episode action caps (inherited from parent)
        w_neff  = float(self.current_w[1])
        w_water = float(self.current_w[2])
        self._max_anfer = float(np.clip(200.0 * (1.0 - w_neff)**2,  2.0, 200.0))
        self._max_amir  = float(np.clip( 50.0 * (1.0 - w_water)**2, 3.0,  50.0))

        fault_type, fault_kwargs = self._sample_two_faults(phase)
        self._fault_state     = _FaultStateV3(fault_kwargs)
        self._last_fault_type = fault_type

        sos_obs = self._sos_env.reset()
        raw_obs = self._encode(sos_obs)               # (14,): [crop(11), w(3)]
        obs19   = self._fault_state.corrupt(raw_obs)  # (19,): [crop(11), mask(5), w(3)]
        obs19[-N_OBJ:] = self.current_w              # ensure w is current

        return obs19, {
            'preference_w': self.current_w.copy(),
            'fault_type':   fault_type,
            'phase':        phase,
        }

    def _sample_two_faults(self, phase):
        """Return (fault_type, kwargs) from {clean, noise, stuck}."""
        sev        = 0.4 if phase == 1 else 1.0
        clean_prob = 0.85 if phase == 1 else 0.70

        if np.random.random() < clean_prob:
            return 'clean', {}

        if np.random.random() < 0.5:
            # sensor_noise: Gaussian σ on all crop sensors
            std = float(np.random.uniform(0.02, 0.12 * sev))
            return 'noise', {'noise_std': std}
        else:
            # sensor_stuck: one sensor freezes after stuck_day
            s   = int(np.random.choice([0, 1, 2]))  # soil/canopy/grain
            day = int(np.random.uniform(10, max(11, int(120 * sev))))
            return 'stuck', {'stuck_sensors': [s], 'stuck_day': day}


# ══════════════════════════════════════════════════════════════════════
# Transition buffer  (K=20 obs window, hindsight w at sample time)
# ══════════════════════════════════════════════════════════════════════
class TransitionBuffer:
    """
    Stores per-step transitions with a K-step obs16 history window.
    At sample time: resample w ~ Dirichlet → inject into LSTM input.
    """

    def __init__(self, max_size=BUFFER_SIZE, K=K_HISTORY):
        self.K      = K
        self.buffer = deque(maxlen=max_size)
        self._hist  = deque(maxlen=K)   # rolling obs16 window during collection

    def reset_episode(self):
        self._hist.clear()

    def push(self, obs16, obs16_next, action, r_daily, r_vec, done,
             fault_type='clean'):
        """
        obs16      : (16,) current observation
        obs16_next : (16,) next observation
        r_vec      : (3,) terminal reward vector, or None for non-terminal steps
        """
        self._hist.append(obs16.copy())
        seq = list(self._hist)
        # Zero-pad at episode start if history not full
        pad      = [np.zeros(OBS16_DIM, dtype=np.float32)] * (self.K - len(seq))
        seq_full = np.stack(pad + seq, axis=0)  # (K, 16)

        self.buffer.append({
            'obs_seq':  seq_full,
            'obs_next': obs16_next.copy(),
            'action':   action.copy(),
            'r_daily':  float(r_daily),
            'r_vec':    r_vec.copy() if r_vec is not None else None,
            'done':     bool(done),
            'fault':    fault_type,
        })

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        B = len(batch)

        obs_seq  = np.stack([b['obs_seq']  for b in batch])  # (B, K, 16)
        obs_next = np.stack([b['obs_next'] for b in batch])  # (B, 16)
        actions  = np.stack([b['action']   for b in batch])  # (B, 2)
        r_daily  = np.array([b['r_daily']  for b in batch], dtype=np.float32)
        dones    = np.array([b['done']     for b in batch], dtype=np.float32)

        # Hindsight preference relabeling: resample w per transition
        w = np.random.dirichlet(np.ones(N_OBJ), size=B).astype(np.float32)

        # Augmented reward: step reward + terminal scalarized bonus
        r_aug = r_daily.copy()
        for i, b in enumerate(batch):
            if b['done'] and b['r_vec'] is not None:
                rv     = b['r_vec']
                scal   = float(np.dot(w[i], rv))
                concav = AUG_LAMBDA * float(
                    np.mean(np.log(np.clip(rv, 0.0, None) + 1.0 + AUG_EPS)))
                r_aug[i] += scal + concav

        # Next obs window: shift right by 1, append obs_next
        obs_seq_next = np.concatenate([
            obs_seq[:, 1:, :],            # (B, K-1, 16)
            obs_next[:, np.newaxis, :],   # (B,   1, 16)
        ], axis=1)                        # (B, K, 16)

        def tt(x): return torch.FloatTensor(x).to(DEVICE)
        return (tt(obs_seq), tt(obs_seq_next),
                tt(actions), tt(r_aug), tt(dones), tt(w))

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════
# Networks
# ══════════════════════════════════════════════════════════════════════
def _set_forget_bias(lstm, val=1.0):
    """Initialise LSTM forget-gate bias to val (helps gradient flow)."""
    for name, p in lstm.named_parameters():
        if 'bias' in name:
            n = p.size(0)
            nn.init.constant_(p[n // 4: n // 2], val)


class LSTMActor(nn.Module):
    """
    LSTM actor with preference vector w injected at every timestep.

    LSTM input  : [obs16(16), w(3)] = 19-dim per step
    LSTM output : hidden state h_t (256-dim)
    Head input  : [h_t(256), w(3)] = 259-dim   (w redundant but reinforces signal)
    Head output : [anfer, amir]
    """

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(LSTM_IN, HIDDEN, batch_first=True)
        _set_forget_bias(self.lstm)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN + N_OBJ, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, ACT_DIM),        nn.Tanh(),
        )
        self.register_buffer('act_mid',  torch.FloatTensor(ACT_MID))
        self.register_buffer('act_half', torch.FloatTensor(ACT_HALF))

    def forward(self, obs_seq, w):
        """
        obs_seq : (B, K, 16)
        w       : (B, 3)
        Returns : actions (B, 2), h_last (B, HIDDEN)
        """
        B, K, _ = obs_seq.shape
        w_exp    = w.unsqueeze(1).expand(-1, K, -1)           # (B, K, 3)
        lstm_in  = torch.cat([obs_seq, w_exp], dim=-1)        # (B, K, 19)
        h_seq, _ = self.lstm(lstm_in)                         # (B, K, HIDDEN)
        h_last   = h_seq[:, -1, :]                            # (B, HIDDEN)
        raw      = self.head(torch.cat([h_last, w], dim=-1))  # (B, 2)
        return raw * self.act_half + self.act_mid, h_last

    @torch.no_grad()
    def step(self, obs16_np, w_np, h, c):
        """Single env step — maintains (h, c) across episode steps."""
        x   = torch.FloatTensor(obs16_np).view(1, 1, OBS16_DIM).to(DEVICE)
        wt  = torch.FloatTensor(w_np).view(1, 1, N_OBJ).to(DEVICE)
        inp = torch.cat([x, wt], dim=-1)
        h_out, (hn, cn) = self.lstm(inp, (h, c))
        raw    = self.head(torch.cat([h_out.squeeze(1),
                                       wt.squeeze(1)], dim=-1))
        action = (raw * self.act_half + self.act_mid).cpu().numpy().flatten()
        return action.astype(np.float32), hn, cn


class LSTMTwinCritic(nn.Module):
    """
    Twin LSTM critic with w injected at every LSTM step.

    LSTM input  : [obs16(16), w(3)] = 19-dim per step
    Head input  : [h_t(256), action(2), w(3)] = 261-dim
    Head output : Q-value (scalar)
    """

    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(LSTM_IN, HIDDEN, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_IN, HIDDEN, batch_first=True)
        _set_forget_bias(self.lstm1)
        _set_forget_bias(self.lstm2)
        head_in = HIDDEN + ACT_DIM + N_OBJ   # 261
        self.q1 = nn.Sequential(
            nn.Linear(head_in, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(head_in, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)
        )

    def _hidden(self, obs_seq, w):
        B, K, _ = obs_seq.shape
        w_exp   = w.unsqueeze(1).expand(-1, K, -1)
        inp     = torch.cat([obs_seq, w_exp], dim=-1)
        h1, _   = self.lstm1(inp)
        h2, _   = self.lstm2(inp)
        return h1[:, -1, :], h2[:, -1, :]   # (B, HIDDEN) each

    def forward(self, obs_seq, w, actions):
        h1, h2 = self._hidden(obs_seq, w)
        c1 = torch.cat([h1, actions, w], dim=-1)
        c2 = torch.cat([h2, actions, w], dim=-1)
        return self.q1(c1), self.q2(c2)

    def Q1(self, obs_seq, w, actions):
        h1, _ = self._hidden(obs_seq, w)
        return self.q1(torch.cat([h1, actions, w], dim=-1))


def soft_update(net, tgt, tau=TAU):
    for p, tp in zip(net.parameters(), tgt.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


# ══════════════════════════════════════════════════════════════════════
# TD3 update (truncated BPTT K=20, hindsight w)
# ══════════════════════════════════════════════════════════════════════
_act_low_t  = torch.FloatTensor(ACT_LOW).to(DEVICE)
_act_high_t = torch.FloatTensor(ACT_HIGH).to(DEVICE)
_tp_std_t   = torch.FloatTensor(TP_STD).to(DEVICE)
_tp_clip_t  = torch.FloatTensor(TP_CLIP).to(DEVICE)


def update(batch, actor, actor_tgt, critic, critic_tgt,
           actor_opt, critic_opt, update_count):
    obs_seq, obs_seq_next, actions, r_aug, dones, w = batch
    # obs_seq / obs_seq_next : (B, K, 16)
    # actions                 : (B, 2)
    # r_aug, dones            : (B,)
    # w                       : (B, 3)   — resampled hindsight preference

    # ── Critic update ──────────────────────────────────────────────
    with torch.no_grad():
        next_acts, _ = actor_tgt.forward(obs_seq_next, w)     # (B, 2)

        # TD3 target policy smoothing
        noise = torch.randn_like(next_acts) * _tp_std_t
        noise = noise.clamp(-_tp_clip_t, _tp_clip_t)
        next_acts = (next_acts + noise).clamp(_act_low_t, _act_high_t)

        tq1, tq2  = critic_tgt.forward(obs_seq_next, w, next_acts)
        tq        = torch.min(tq1, tq2).squeeze(-1)            # (B,)
        target_q  = r_aug + GAMMA * (1.0 - dones) * tq        # (B,)

    q1, q2   = critic.forward(obs_seq, w, actions)
    c_loss   = (F.mse_loss(q1.squeeze(-1), target_q) +
                F.mse_loss(q2.squeeze(-1), target_q))
    critic_opt.zero_grad()
    c_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
    critic_opt.step()

    # ── Actor update (delayed, every ACTOR_UPDATE_FREQ critic updates) ─
    a_loss_val = None
    if update_count % ACTOR_UPDATE_FREQ == 0:
        new_acts, _ = actor.forward(obs_seq, w)
        q_pi        = critic.Q1(obs_seq, w, new_acts)
        a_loss      = -q_pi.mean()
        actor_opt.zero_grad()
        a_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
        actor_opt.step()
        soft_update(actor, actor_tgt)
        soft_update(critic, critic_tgt)
        a_loss_val = a_loss.item()

    return c_loss.item(), a_loss_val


# ══════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════
def _run_eval_ep(actor, env, w_corner, fault_kwargs=None):
    """Run one deterministic episode. Returns (rv, sos_info_last)."""
    env.current_w   = w_corner.copy()
    env._max_anfer  = float(np.clip(200.0 * (1.0 - w_corner[1])**2, 2.0, 200.0))
    env._max_amir   = float(np.clip( 50.0 * (1.0 - w_corner[2])**2, 3.0,  50.0))

    if fault_kwargs is not None:
        env._fault_state = _FaultStateV3(fault_kwargs)
    else:
        env._fault_state = _FaultStateV3({})

    sos_obs = env._sos_env.reset()
    raw_obs = env._encode(sos_obs)
    obs19   = env._fault_state.corrupt(raw_obs)
    obs19[-N_OBJ:] = w_corner

    h = torch.zeros(1, 1, HIDDEN, device=DEVICE)
    c = torch.zeros(1, 1, HIDDEN, device=DEVICE)
    done, last_info = False, {}

    while not done:
        obs16  = obs19[:OBS16_DIM]
        action, h, c = actor.step(obs16, w_corner, h, c)
        action = np.clip(action, [0, 0], [env._max_anfer, env._max_amir])
        obs19, _, done, _, last_info = env.step(action)
        obs19[-N_OBJ:] = w_corner

    rv  = np.array(last_info.get('reward_vec', [0, 0, 0]), dtype=np.float32)
    sos = last_info.get('sos_state', {})
    return rv, sos


def eval_all(actor, total_steps, env):
    """Evaluate all 4 corners × {clean, noise, stuck} fault scenarios."""
    results = []

    fault_scenarios = [
        ('clean', {}),
        ('noise_mid',  {'noise_std': 0.08}),
        ('stuck_s0',   {'stuck_sensors': [0], 'stuck_day': 40}),
    ]

    for corner_name, w_corner in EVAL_CORNERS.items():
        for fault_name, fault_kwargs in fault_scenarios:
            for seed in range(7000, 7000 + N_EVAL_EPS):
                env._sos_env._seed = seed
                rv, sos = _run_eval_ep(actor, env, w_corner,
                                       fault_kwargs if fault_name != 'clean' else None)
                results.append({
                    'total_steps':   total_steps,
                    'corner':        corner_name,
                    'fault':         fault_name,
                    'R_yield':       float(rv[0]),
                    'R_ane':         float(rv[1]),
                    'R_water_eff':   float(rv[2]),
                    'yield_kg_ha':   float(sos.get('grnwt', 0)),
                    'total_N_kg_ha': float(sos.get('total_nitrogen', 0)),
                    'total_W_mm':    float(sos.get('total_water', 0)),
                })

    return results


# ══════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════
def plot_training(ep_rows):
    if len(ep_rows) < 5:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    def smooth(vals, k=60):
        out = []
        for i in range(len(vals)):
            out.append(np.mean(vals[max(0, i - k):i + 1]))
        return out

    for ax, key, title in [
        (axes[0], 'clean',  'Clean episodes'),
        (axes[1], 'faulted', 'Faulted episodes'),
    ]:
        rows = [r for r in ep_rows
                if (key == 'clean') == r['is_clean']]
        if not rows:
            ax.set_title(f'{title} — no data')
            continue
        steps = [r['total_steps'] for r in rows]
        ry    = [r['R_yield']     for r in rows]
        ra    = [r['R_ane']       for r in rows]
        rw    = [r['R_water_eff'] for r in rows]
        ax.plot(steps, smooth(ry), label='R_yield',   color='#27ae60', lw=1.5)
        ax.plot(steps, smooth(ra), label='R_ane',     color='#e67e22', lw=1.5)
        ax.plot(steps, smooth(rw), label='R_water',   color='#2980b9', lw=1.5)
        ax.axhline(0, color='#aaa', lw=0.7, ls='--')
        ax.axvline(1_000_000, color='#c0392b', lw=1, ls=':', alpha=0.7)
        ax.text(1_000_000, ax.get_ylim()[0], 'P1→P2',
                fontsize=7, color='#c0392b', va='bottom', ha='right')
        ax.set_xlabel('Total steps')
        ax.set_ylabel('Reward component (smoothed)')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle('CAPQL-LSTM v2 — Training Curves', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'fig_lstm_v2_training.png'),
                dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_corners(eval_rows):
    """4-panel bar chart: R_yield per corner × fault scenario."""
    if not eval_rows:
        return

    corners = list(EVAL_CORNERS.keys())
    faults  = ['clean', 'noise_mid', 'stuck_s0']
    colors  = ['#27ae60', '#e67e22', '#e74c3c']

    # Take latest eval checkpoint
    latest = max(r['total_steps'] for r in eval_rows)
    rows   = [r for r in eval_rows if r['total_steps'] == latest]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['R_yield', 'R_ane', 'R_water_eff']
    labels  = ['R_yield', 'R_ane', 'R_water_eff']

    for ax, met, lab in zip(axes, metrics, labels):
        x = np.arange(len(corners))
        w = 0.25
        for fi, (fault, col) in enumerate(zip(faults, colors)):
            vals = []
            for corner in corners:
                subset = [r[met] for r in rows
                          if r['corner'] == corner and r['fault'] == fault]
                vals.append(float(np.mean(subset)) if subset else 0.0)
            ax.bar(x + fi * w, vals, width=w, label=fault, color=col, alpha=0.85)
        ax.set_xticks(x + w)
        ax.set_xticklabels(corners, rotation=15, fontsize=9)
        ax.axhline(0, color='#aaa', lw=0.7, ls='--')
        ax.set_ylabel(lab)
        ax.set_title(lab)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle(f'CAPQL-LSTM v2 — Corner Evaluation @ {latest:,} steps',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'fig_lstm_v2_corners.png'),
                dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_mechanistic(actor, env):
    """
    Run one episode with a stuck sensor (sensor 0 freezes at day 40).
    Plot: obs sensor 0, LSTM hidden-state norm, actions over time.
    This is the 'mechanistic' figure for the thesis.
    """
    w_corner = EVAL_CORNERS['yield']
    env.current_w   = w_corner.copy()
    env._max_anfer  = 200.0
    env._max_amir   = 50.0
    env._fault_state = _FaultStateV3({'stuck_sensors': [0], 'stuck_day': 40})

    sos_obs = env._sos_env.reset()
    raw_obs = env._encode(sos_obs)
    obs19   = env._fault_state.corrupt(raw_obs)
    obs19[-N_OBJ:] = w_corner

    h = torch.zeros(1, 1, HIDDEN, device=DEVICE)
    c = torch.zeros(1, 1, HIDDEN, device=DEVICE)

    days, sw_vals, h_norms, anfets, amirs = [], [], [], [], []
    done, day = False, 0

    while not done:
        obs16 = obs19[:OBS16_DIM]
        sw_vals.append(float(obs16[0]))   # sw_mean (first crop feature)

        with torch.no_grad():
            x  = torch.FloatTensor(obs16).view(1, 1, OBS16_DIM).to(DEVICE)
            wt = torch.FloatTensor(w_corner).view(1, 1, N_OBJ).to(DEVICE)
            inp = torch.cat([x, wt], dim=-1)
            h_out, (h, c) = actor.lstm(inp, (h, c))
            h_norm = float(h.norm().item())
            raw    = actor.head(torch.cat([h_out.squeeze(1),
                                            wt.squeeze(1)], dim=-1))
            action = (raw * actor.act_half + actor.act_mid).cpu().numpy().flatten()

        h_norms.append(h_norm)
        action = np.clip(action, [0, 0], [200.0, 50.0])
        anfets.append(float(action[0]))
        amirs.append(float(action[1]))
        days.append(day)

        obs19, _, done, _, _ = env.step(action)
        obs19[-N_OBJ:] = w_corner
        day += 1

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(days, sw_vals, color='#2980b9', lw=1.5)
    axes[0].axvline(40, color='red', ls='--', lw=1, label='Sensor stuck @ day 40')
    axes[0].set_ylabel('sw_mean (obs[0])')
    axes[0].set_title('Sensor 0 (soil water) — frozen after day 40')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].plot(days, h_norms, color='#8e44ad', lw=1.5)
    axes[1].axvline(40, color='red', ls='--', lw=1)
    axes[1].set_ylabel('||h_t||  (LSTM hidden norm)')
    axes[1].set_title('LSTM hidden-state norm — shifts when fault occurs')
    axes[1].grid(alpha=0.2)

    axes[2].plot(days, anfets, color='#e67e22', lw=1.2, label='N applied (kg/ha)')
    axes[2].plot(days, amirs,  color='#27ae60', lw=1.2, label='Water applied (mm)')
    axes[2].axvline(40, color='red', ls='--', lw=1)
    axes[2].set_ylabel('Action')
    axes[2].set_xlabel('Day of season')
    axes[2].set_title('Actions — model adapts after stuck-sensor onset')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)

    fig.suptitle('CAPQL-LSTM v2 — Mechanistic Response to Stuck Sensor',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'fig_lstm_v2_mechanistic.png'),
                dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] mechanistic figure saved.')


# ══════════════════════════════════════════════════════════════════════
# CSV helpers
# ══════════════════════════════════════════════════════════════════════
def _open_csv(path, fieldnames):
    f = open(path, 'w', newline='')
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    return f, w


# ══════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════
def main():
    print(f'Device: {DEVICE}')
    print(f'Total steps: {TOTAL_STEPS:,}')
    print(f'K_HISTORY: {K_HISTORY}  BATCH: {BATCH_SIZE}  '
          f'UPDATE_INTERVAL: {UPDATE_INTERVAL}  N_UPDATES: {N_UPDATES}')
    print(f'Expected gradient steps: ~{(TOTAL_STEPS - WARMUP_STEPS) // UPDATE_INTERVAL * N_UPDATES:,}')
    print(f'CORNER_PROB: {CORNER_PROB}  LSTM_IN: {LSTM_IN}')
    print()

    # ── Networks ──────────────────────────────────────────────────────
    actor      = LSTMActor().to(DEVICE)
    actor_tgt  = copy.deepcopy(actor)
    critic     = LSTMTwinCritic().to(DEVICE)
    critic_tgt = copy.deepcopy(critic)

    for net in [actor_tgt, critic_tgt]:
        for p in net.parameters():
            p.requires_grad_(False)

    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=LR_ACTOR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    buffer = TransitionBuffer()

    # ── Environments ──────────────────────────────────────────────────
    train_env = TwoFaultEnv(dssat_seed=42)
    eval_env  = TwoFaultEnv(dssat_seed=9999)

    # ── CSV logs ──────────────────────────────────────────────────────
    ep_fields = ['episode', 'total_steps', 'fault_type', 'phase', 'is_clean',
                 'R_yield', 'R_ane', 'R_water_eff', 'ep_len', 'critic_loss']
    ev_fields = ['total_steps', 'corner', 'fault', 'R_yield', 'R_ane',
                 'R_water_eff', 'yield_kg_ha', 'total_N_kg_ha', 'total_W_mm']

    ep_f, ep_w = _open_csv(EP_LOG, ep_fields)
    ev_f, ev_w = _open_csv(EVAL_LOG, ev_fields)

    # ── Training state ────────────────────────────────────────────────
    total_steps  = 0
    episode      = 0
    update_count = 0
    ep_rows      = []
    eval_rows    = []
    t0           = time.time()

    # ── Main loop ─────────────────────────────────────────────────────
    while total_steps < TOTAL_STEPS:
        train_env.total_steps = total_steps
        obs19, info = train_env.reset()

        obs16      = obs19[:OBS16_DIM]
        w_ep       = info['preference_w']
        fault_type = info['fault_type']
        phase      = info['phase']
        is_clean   = (fault_type == 'clean')

        h = torch.zeros(1, 1, HIDDEN, device=DEVICE)
        c = torch.zeros(1, 1, HIDDEN, device=DEVICE)
        buffer.reset_episode()

        done, ep_len = False, 0
        last_rv = np.zeros(N_OBJ, dtype=np.float32)
        ep_c_loss_sum = 0.0
        ep_c_loss_cnt = 0

        # Exploration noise magnitude
        noise_scale = max(
            NOISE_END,
            NOISE_START * (1.0 - total_steps / NOISE_DECAY)
        )

        while not done:
            # ── Select action ─────────────────────────────────────
            if total_steps < WARMUP_STEPS:
                action = train_env.action_space.sample()
            else:
                action, h, c = actor.step(obs16, w_ep, h, c)
                noise = np.random.normal(0, noise_scale, size=2) * ACT_HIGH
                action = np.clip(action + noise, ACT_LOW, ACT_HIGH)

            # ── Env step ──────────────────────────────────────────
            obs19_next, r_daily, done, _, step_info = train_env.step(action)
            obs16_next = obs19_next[:OBS16_DIM]

            r_vec = None
            if done:
                r_vec   = np.array(step_info.get('reward_vec', [0, 0, 0]),
                                    dtype=np.float32)
                last_rv = r_vec

            buffer.push(obs16, obs16_next, action, r_daily, r_vec,
                        done, fault_type)

            obs16        = obs16_next
            total_steps += 1
            ep_len      += 1

            # ── Gradient updates ──────────────────────────────────
            if (total_steps > WARMUP_STEPS and
                    total_steps % UPDATE_INTERVAL == 0 and
                    len(buffer) >= BATCH_SIZE):
                for _ in range(N_UPDATES):
                    batch = buffer.sample(BATCH_SIZE)
                    c_loss, _ = update(
                        batch, actor, actor_tgt, critic, critic_tgt,
                        actor_opt, critic_opt, update_count)
                    ep_c_loss_sum += c_loss
                    ep_c_loss_cnt += 1
                    update_count  += 1

        # ── Episode logging ───────────────────────────────────────
        episode += 1
        avg_closs = ep_c_loss_sum / max(1, ep_c_loss_cnt)
        row = {
            'episode':     episode,
            'total_steps': total_steps,
            'fault_type':  fault_type,
            'phase':       phase,
            'is_clean':    int(is_clean),
            'R_yield':     round(float(last_rv[0]), 4),
            'R_ane':       round(float(last_rv[1]), 4),
            'R_water_eff': round(float(last_rv[2]), 4),
            'ep_len':      ep_len,
            'critic_loss': round(avg_closs, 6),
        }
        ep_w.writerow(row)
        ep_f.flush()
        ep_rows.append({**row, 'is_clean': is_clean})

        if episode % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            print(f'[{total_steps:>9,}] ep={episode:>5}  '
                  f'phase={phase}  fault={fault_type:<12}  '
                  f'Ry={last_rv[0]:+.3f}  Ra={last_rv[1]:+.3f}  '
                  f'Rw={last_rv[2]:+.3f}  '
                  f'noise={noise_scale:.3f}  '
                  f'c_loss={avg_closs:.4f}  '
                  f't={elapsed/60:.1f}min')

        # ── Periodic evaluation ───────────────────────────────────
        if total_steps % EVAL_INTERVAL < ep_len:
            print(f'\n--- Eval @ {total_steps:,} steps ---')
            eval_env.total_steps = total_steps
            results = eval_all(actor, total_steps, eval_env)
            for r in results:
                ev_w.writerow(r)
                eval_rows.append(r)
            ev_f.flush()

            # Print summary
            for corner in EVAL_CORNERS:
                clean_res = [r for r in results
                             if r['corner'] == corner and r['fault'] == 'clean']
                if clean_res:
                    ry = np.mean([r['R_yield']     for r in clean_res])
                    rw = np.mean([r['R_water_eff'] for r in clean_res])
                    ya = np.mean([r['yield_kg_ha'] for r in clean_res])
                    wm = np.mean([r['total_W_mm']  for r in clean_res])
                    print(f'  {corner:<10}  Ry={ry:+.3f}  Rw={rw:+.3f}  '
                          f'yield={ya:.0f}kg/ha  W={wm:.0f}mm')
            print()

            # Refresh plots
            plot_training(ep_rows)
            plot_corners(eval_rows)

        # ── Checkpoint ────────────────────────────────────────────
        if total_steps % 500_000 < ep_len:
            ckpt = os.path.join(MODEL_DIR,
                                f'ckpt_{total_steps // 1000}k.pt')
            torch.save({
                'actor':      actor.state_dict(),
                'critic':     critic.state_dict(),
                'total_steps': total_steps,
            }, ckpt)
            print(f'[ckpt] saved → {ckpt}')

    # ── Final save ────────────────────────────────────────────────────
    torch.save({
        'actor':       actor.state_dict(),
        'critic':      critic.state_dict(),
        'actor_tgt':   actor_tgt.state_dict(),
        'critic_tgt':  critic_tgt.state_dict(),
        'total_steps': total_steps,
    }, os.path.join(MODEL_DIR, 'final.pt'))
    print(f'\nModel saved → {MODEL_DIR}/final.pt')

    # ── Final evaluation + mechanistic plot ───────────────────────────
    eval_env.total_steps = total_steps
    final_results = eval_all(actor, total_steps, eval_env)
    for r in final_results:
        ev_w.writerow(r)
    ev_f.flush()

    plot_training(ep_rows)
    plot_corners(final_results)
    plot_mechanistic(actor, eval_env)

    ep_f.close()
    ev_f.close()
    train_env.close()
    eval_env.close()

    elapsed = time.time() - t0
    print(f'\nTotal training time: {elapsed/60:.1f} min  ({elapsed/3600:.2f} h)')


if __name__ == '__main__':
    main()
