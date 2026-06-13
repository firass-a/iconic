"""
12_capql_train_robust.py  —  CAPQL-Robust (MLP actor + fault-augmented training).

Same architecture as CAPQL v2 (07_capql_train_v2.py) with two additions:
  1. 19-dim obs: faulted_crop(11) + mask_flags(5) + preference_w(3)
     mask_flags[i] = 0 → Type 1 fault (dropout / packet loss, agent knows)
     mask_flags[i] = 1 → Type 2 fault (stuck / bias, looks valid)
  2. Fault-augmented training via CAPQLRobustEnv:
     60% clean | 10% dropout | 10% packet | 10% stuck | 5% N-act | 5% W-act

The replay buffer stores crop+mask (16-dim) and w SEPARATELY so that
hindsight preference relabeling can resample w at every batch call —
exactly as in v2.  This is the critical fix: every sampled transition
has a w-dependent terminal reward, so Q-learning bootstrapping propagates
the preference signal back through all timesteps (not just the ~10% that
fall in a 16-step window as in the sequence-buffer GRU approach).

Outputs:
  /workspace/capql_robust/actor.pt
  /workspace/capql_robust/critic.pt
  /workspace/capql_robust_episode_log.csv
  /workspace/capql_robust_eval_results.csv

Run inside Docker:
    cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
    /opt/gym_dssat_pdi/bin/python3 -u 12_capql_train_robust.py 2>&1 | tee /workspace/gym-dssat-pdi/robust_train_v3.log
"""

import csv, os, sys, time
import numpy as np
from importlib import import_module

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import torch
import torch.nn as nn
import torch.nn.functional as F

CAPQLRobustEnv = import_module('capql_env_robust').CAPQLRobustEnv

# ══════════════════════════════════════════════════════════════════════
# Hyper-parameters  (match v2 exactly wherever possible)
# ══════════════════════════════════════════════════════════════════════
TOTAL_STEPS   = 1_000_000
WARMUP_STEPS  = 5_000
BUFFER_SIZE   = 200_000
BATCH_SIZE    = 256
LR_ACTOR      = 3e-4
LR_CRITIC     = 3e-4
GAMMA         = 0.99
TAU           = 0.005
POLICY_FREQ   = 2
GRAD_CLIP     = 1.0
AUG_LAMBDA    = 0.5
AUG_EPS       = 0.1

EXPL_NOISE_START = 0.3
EXPL_NOISE_END   = 0.05
EXPL_NOISE_STEPS = 800_000

TP_NOISE_STD  = np.array([8.0,  2.0], dtype=np.float32)
TP_NOISE_CLIP = np.array([20.0, 5.0], dtype=np.float32)

N_OBJECTIVES  = 3
CROP_MASK_DIM = 16    # crop(11) + mask(5) — stored without w for hindsight
OBS_DIM       = 19    # CROP_MASK_DIM + N_OBJECTIVES
ACTION_DIM    = 2
HIDDEN        = 256

ACT_LOW   = np.array([0.0,   0.0], dtype=np.float32)
ACT_HIGH  = np.array([200.0, 50.0], dtype=np.float32)
ACT_RANGE = ACT_HIGH - ACT_LOW

MODEL_DIR  = '/workspace/capql_robust'
EP_LOG     = '/workspace/capql_robust_episode_log.csv'
EVAL_CSV   = '/workspace/capql_robust_eval_results.csv'
DEVICE     = torch.device('cpu')

EVAL_CORNERS = {
    'yield':    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    'n_eff':    np.array([0.0, 1.0, 0.0], dtype=np.float32),
    'water':    np.array([0.0, 0.0, 1.0], dtype=np.float32),
    'balanced': np.array([1/3, 1/3, 1/3], dtype=np.float32),
}
N_EVAL_EPS   = 5
LOG_INTERVAL = 10

_ACT_MID  = torch.FloatTensor((ACT_HIGH + ACT_LOW) / 2.0).to(DEVICE)
_ACT_HALF = torch.FloatTensor(ACT_RANGE / 2.0).to(DEVICE)
_ACT_LOW  = torch.FloatTensor(ACT_LOW).to(DEVICE)
_ACT_HIGH = torch.FloatTensor(ACT_HIGH).to(DEVICE)


# ══════════════════════════════════════════════════════════════════════
# Networks  (identical to v2 — only input dim differs: 19 vs 14)
# ══════════════════════════════════════════════════════════════════════
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

    def act(self, obs, step=None, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.forward(obs_t).squeeze(0).cpu().numpy()
        if not deterministic and step is not None:
            frac  = min(1.0, step / EXPL_NOISE_STEPS)
            sigma = EXPL_NOISE_START + frac * (EXPL_NOISE_END - EXPL_NOISE_START)
            noise = np.random.normal(0.0, 1.0, size=ACTION_DIM) * sigma * ACT_RANGE
            action = np.clip(action + noise, ACT_LOW, ACT_HIGH)
        return action.astype(np.float32)


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
        return torch.min(*self.forward(obs, action))


# ══════════════════════════════════════════════════════════════════════
# Replay buffer  (crop+mask and w stored separately for hindsight)
# ══════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    """
    Stores crop+mask (16-dim) and w separately so that at sample time
    w can be resampled from Dirichlet(1,1,1) for hindsight relabeling.

    This ensures EVERY batch has a w-dependent terminal reward signal,
    which is what makes CAPQL's preference conditioning work.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.ptr = self.size = 0
        self.crop_mask      = np.zeros((capacity, CROP_MASK_DIM), dtype=np.float32)
        self.next_crop_mask = np.zeros((capacity, CROP_MASK_DIM), dtype=np.float32)
        self.action         = np.zeros((capacity, ACTION_DIM),    dtype=np.float32)
        self.r_daily        = np.zeros((capacity, 1),             dtype=np.float32)
        self.reward_vec     = np.zeros((capacity, N_OBJECTIVES),  dtype=np.float32)
        self.done           = np.zeros((capacity, 1),             dtype=np.float32)

    def add(self, obs19, action, r_daily, reward_vec, next_obs19, done):
        i = self.ptr
        self.crop_mask[i]      = obs19[:CROP_MASK_DIM]       # obs[0:16]
        self.next_crop_mask[i] = next_obs19[:CROP_MASK_DIM]  # obs[0:16]
        self.action[i]         = action
        self.r_daily[i]        = float(r_daily)
        self.reward_vec[i]     = reward_vec
        self.done[i]           = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        # Resample w ~ Dirichlet(1,1,1) for every transition in the batch
        w_hl = np.random.dirichlet(
            np.ones(N_OBJECTIVES), size=batch_size
        ).astype(np.float32)

        # Reconstruct 19-dim obs with hindsight w
        obs      = np.concatenate([self.crop_mask[idx],      w_hl], axis=1)
        next_obs = np.concatenate([self.next_crop_mask[idx], w_hl], axis=1)

        def tt(x): return torch.FloatTensor(x).to(DEVICE)
        return (
            tt(obs),
            tt(self.action[idx]),
            tt(self.r_daily[idx]),
            tt(self.reward_vec[idx]),
            tt(next_obs),
            tt(self.done[idx]),
            tt(w_hl),
        )

    def __len__(self):
        return self.size


# ══════════════════════════════════════════════════════════════════════
# Concave-augmented scalarization  (same as v2)
# ══════════════════════════════════════════════════════════════════════
def augmented_reward(r_daily, reward_vec, done, w):
    """r_daily + done × (w·rv + λ·mean(log(rv + 1 + ε)))"""
    scalarized = (w * reward_vec).sum(dim=-1, keepdim=True)
    concave    = AUG_LAMBDA * torch.mean(
        torch.log(reward_vec + 1.0 + AUG_EPS), dim=-1, keepdim=True
    )
    return r_daily + done * (scalarized + concave)


def soft_update(net, tgt, tau):
    for p, tp in zip(net.parameters(), tgt.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════
def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print('=' * 68)
    print('12_capql_train_robust.py  —  CAPQL-Robust (MLP + fault training)')
    print('=' * 68)
    print(f'  Obs dim  : {OBS_DIM}  (crop(11) + mask(5) + w(3))')
    print(f'  Actor    : MLP  ({OBS_DIM}→{HIDDEN}→{HIDDEN}→{ACTION_DIM})')
    print(f'  Critic   : MLP  ({OBS_DIM+ACTION_DIM}→{HIDDEN}→{HIDDEN}→1)')
    print(f'  Steps    : {TOTAL_STEPS:,}  |  warmup {WARMUP_STEPS:,}  |  batch {BATCH_SIZE}')
    print(f'  Hindsight: w ~ Dirichlet(1,1,1) resampled at every batch')
    print(f'  Aug λ={AUG_LAMBDA}  ε={AUG_EPS}')
    print('  Fault mix: 60% clean | 10% dropout | 10% packet | 10% stuck'
          ' | 5% N-act | 5% W-act')
    print('=' * 68 + '\n')

    actor      = Actor().to(DEVICE)
    actor_tgt  = Actor().to(DEVICE)
    actor_tgt.load_state_dict(actor.state_dict())
    critic     = TwinCritic().to(DEVICE)
    critic_tgt = TwinCritic().to(DEVICE)
    critic_tgt.load_state_dict(critic.state_dict())
    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=LR_ACTOR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    buffer = ReplayBuffer(BUFFER_SIZE)
    env    = CAPQLRobustEnv(dssat_seed=1)

    ep_log_f = open(EP_LOG, 'w', newline='', buffering=1)
    ep_log_w = csv.DictWriter(ep_log_f, fieldnames=[
        'episode', 'timestep', 'R_yield', 'R_ane', 'R_water_eff',
        'yield_kg_ha', 'total_N_kg_ha', 'total_W_mm',
        'w_yield', 'w_neff', 'w_water',
    ])
    ep_log_w.writeheader()

    obs, info    = env.reset()
    w            = info['preference_w'].copy()
    ep_count     = 0
    update_count = 0
    t0           = time.time()

    print(f"  {'ep':>5}  {'step':>8}  {'R_yld':>7}  {'R_ane':>7}"
          f"  {'R_wef':>7}  {'yield':>7}  {'w':>18}  ETA")
    print('  ' + '─' * 76)
    print('  Collecting warmup steps...\n', flush=True)

    for step in range(1, TOTAL_STEPS + 1):

        if step < WARMUP_STEPS:
            action = env.action_space.sample()
        else:
            action = actor.act(obs, step=step)

        next_obs, r_daily, done, _, step_info = env.step(action)
        reward_vec = np.array(
            step_info.get('reward_vec', np.zeros(N_OBJECTIVES)),
            dtype=np.float32,
        )

        buffer.add(obs, action, r_daily, reward_vec, next_obs, done)

        # One gradient step per env step (same as v2)
        if step >= WARMUP_STEPS and len(buffer) >= BATCH_SIZE:
            obs_b, act_b, r_b, rv_b, nobs_b, done_b, w_b = buffer.sample(BATCH_SIZE)

            r_aug = augmented_reward(r_b, rv_b, done_b, w_b)

            with torch.no_grad():
                na = actor_tgt(nobs_b)
                tp = torch.zeros_like(na)
                tp[:, 0] = (torch.randn(BATCH_SIZE, device=DEVICE)
                            * TP_NOISE_STD[0]).clamp(-TP_NOISE_CLIP[0], TP_NOISE_CLIP[0])
                tp[:, 1] = (torch.randn(BATCH_SIZE, device=DEVICE)
                            * TP_NOISE_STD[1]).clamp(-TP_NOISE_CLIP[1], TP_NOISE_CLIP[1])
                na        = (na + tp).clamp(_ACT_LOW, _ACT_HIGH)
                tq1, tq2  = critic_tgt(nobs_b, na)
                target_q  = r_aug + GAMMA * (1.0 - done_b) * torch.min(tq1, tq2)

            q1, q2      = critic(obs_b, act_b)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
            critic_opt.step()

            update_count += 1
            if update_count % POLICY_FREQ == 0:
                actor_loss = -critic.q_min(obs_b, actor(obs_b)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
                actor_opt.step()
                soft_update(actor, actor_tgt, TAU)
                soft_update(critic, critic_tgt, TAU)

        if done:
            ep_count += 1
            sos = step_info.get('sos_state', {})
            rv  = reward_vec

            ep_log_w.writerow({
                'episode':       ep_count,
                'timestep':      step,
                'R_yield':       float(rv[0]),
                'R_ane':         float(rv[1]),
                'R_water_eff':   float(rv[2]),
                'yield_kg_ha':   float(sos.get('grnwt', 0.0)),
                'total_N_kg_ha': float(sos.get('total_nitrogen', 0.0)),
                'total_W_mm':    float(sos.get('total_water', 0.0)),
                'w_yield':       float(w[0]),
                'w_neff':        float(w[1]),
                'w_water':       float(w[2]),
            })

            if ep_count % LOG_INTERVAL == 0:
                elapsed = (time.time() - t0) / 60.0
                eta     = elapsed / step * (TOTAL_STEPS - step)
                print(
                    f"  {ep_count:>5}  {step:>8,}"
                    f"  {rv[0]:>+7.3f}  {rv[1]:>+7.3f}  {rv[2]:>+7.3f}"
                    f"  {sos.get('grnwt', 0):>7.0f}"
                    f"  [{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]"
                    f"  ETA {eta:.1f}m",
                    flush=True,
                )

            obs, info = env.reset()
            w         = info['preference_w'].copy()
        else:
            obs = next_obs

    env.close()
    ep_log_f.close()

    torch.save(actor.state_dict(),  os.path.join(MODEL_DIR, 'actor.pt'))
    torch.save(critic.state_dict(), os.path.join(MODEL_DIR, 'critic.pt'))
    print(f'\n  ✓  Model saved to {MODEL_DIR}')
    print(f'  ✓  Training time: {(time.time()-t0)/60:.1f} min')
    return actor


# ══════════════════════════════════════════════════════════════════════
# Evaluation  (clean episodes, correct action caps per corner)
# ══════════════════════════════════════════════════════════════════════
def evaluate(actor):
    print('\n  ── Evaluating on clean corners ──')
    results = []

    for corner_name, w_corner in EVAL_CORNERS.items():
        for seed in range(9000, 9000 + N_EVAL_EPS):
            env = CAPQLRobustEnv(dssat_seed=seed)
            obs, _ = env.reset()

            # Pin preference and set correct action caps for this corner
            env.current_w  = w_corner.copy()
            obs[-3:]       = w_corner
            w_neff         = float(w_corner[1])
            w_water        = float(w_corner[2])
            env._max_anfer = float(np.clip(200.0 * (1.0 - w_neff) ** 2,  2.0, 200.0))
            env._max_amir  = float(np.clip( 50.0 * (1.0 - w_water) ** 2, 3.0,  50.0))

            # Force clean episode
            if env._faulty is not None:
                env._faulty.N_efficiency     = 1.0
                env._faulty.W_efficiency     = 1.0
                env._faulty.dropout_rate     = 0.0
                env._faulty.packet_loss_prob = 0.0
                env._faulty.stuck_sensors    = set()

            done = False
            last_info = {}
            while not done:
                action = actor.act(obs, deterministic=True)
                obs, _, done, _, last_info = env.step(action)
                obs[-3:] = w_corner   # pin w every step
            env.close()

            rv  = last_info.get('reward_vec', np.zeros(3))
            sos = last_info.get('sos_state', {})
            results.append({
                'corner':        corner_name,
                'R_yield':       float(rv[0]),
                'R_ane':         float(rv[1]),
                'R_water_eff':   float(rv[2]),
                'yield_kg_ha':   float(sos.get('grnwt', 0.0)),
                'total_N_kg_ha': float(sos.get('total_nitrogen', 0.0)),
                'total_W_mm':    float(sos.get('total_water', 0.0)),
            })

    with open(EVAL_CSV, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        wr.writeheader()
        wr.writerows(results)
    print(f'  ✓  Eval → {EVAL_CSV}')

    for corner_name in EVAL_CORNERS:
        sub = [r for r in results if r['corner'] == corner_name]
        rY  = np.mean([r['R_yield']     for r in sub])
        rN  = np.mean([r['R_ane']       for r in sub])
        rW  = np.mean([r['R_water_eff'] for r in sub])
        yld = np.mean([r['yield_kg_ha'] for r in sub])
        print(f'  {corner_name:<12} R_yield={rY:+.3f}  R_ane={rN:+.3f}'
              f'  R_water={rW:+.3f}  yield={yld:.0f} kg/ha')


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    try:
        actor = train()
        evaluate(actor)
        print('\n' + '=' * 68)
    except Exception as e:
        print(f'\n  ERROR: {type(e).__name__}: {e}', file=sys.stderr)
        raise
