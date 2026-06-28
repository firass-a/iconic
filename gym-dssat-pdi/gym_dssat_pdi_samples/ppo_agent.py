"""
Custom PPO agent: actor-critic + all loss terms.

Components (explicit):
  1. Rollout collection        → RolloutBuffer
  2. GAE + returns             → RolloutBuffer.compute_returns_and_advantages
  3. PPO clipped policy loss   → PPOAgent.policy_loss
  4. Value function loss       → PPOAgent.value_loss
  5. Entropy bonus             → PPOAgent.entropy_bonus
  6. Gradient update           → PPOAgent.update
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from ppo_rollout_buffer import RolloutBuffer


def _mlp(sizes, activation=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), activation()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Shared-body actor-critic for continuous Box actions."""

    def __init__(self, obs_dim, act_dim, hidden=(128, 128),
                 action_low=None, action_high=None):
        super().__init__()
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32)

        self.backbone = _mlp([obs_dim, *hidden])
        feat = hidden[-1]
        self.mu_head = nn.Linear(feat, act_dim)
        self.v_head = nn.Linear(feat, 1)

        # State-independent log std (one per action dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        h = self.backbone(obs)
        return self.mu_head(h), self.v_head(h).squeeze(-1)

    def _distribution(self, obs):
        mu, value = self.forward(obs)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std), value

    def act(self, obs):
        """Sample action for rollout collection (numpy in/out)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, value = self._distribution(obs_t)
            raw = dist.sample()
            log_prob = dist.log_prob(raw).sum(-1)
            action = torch.clamp(raw, self.action_low, self.action_high)

        return (
            action.squeeze(0).cpu().numpy(),
            float(value.item()),
            float(log_prob.item()),
        )

    def evaluate_actions(self, obs, actions):
        """Re-evaluate log_prob, entropy, value for PPO update."""
        dist, values = self._distribution(obs)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy, values


class PPOAgent:
    """
    Full PPO trainer with explicit loss decomposition.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        action_low,
        action_high,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        lr=3e-4,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        device='cpu',
        seed=0,
    ):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.net = ActorCritic(
            obs_dim, act_dim,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer(
            n_steps, obs_dim, act_dim, gamma=gamma, gae_lambda=gae_lambda,
        )

        self._last_policy_loss = 0.0
        self._last_value_loss = 0.0
        self._last_entropy = 0.0
        self._last_total_loss = 0.0

    # ------------------------------------------------------------------
    # Loss components (each explicit for thesis / debugging)
    # ------------------------------------------------------------------
    @staticmethod
    def policy_loss(log_probs_new, log_probs_old, advantages, clip_range):
        """
        PPO clipped surrogate objective (maximize → minimize negative).

            ratio = π_θ(a|s) / π_θ_old(a|s)
            L_CLIP = E[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ]
        """
        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        return -torch.min(surr1, surr2).mean()

    @staticmethod
    def value_loss(values, returns):
        """MSE between predicted value and GAE return target."""
        return nn.functional.mse_loss(values, returns)

    @staticmethod
    def entropy_bonus(entropy):
        """Mean entropy of the policy (added to total loss as -coef * H)."""
        return entropy.mean()

    # ------------------------------------------------------------------
    # Rollout + update loop
    # ------------------------------------------------------------------
    def collect_rollout(self, env, n_pref_dims=3, episode_log=None):
        """Fill rollout buffer; return (buffer, rollout_stats).

        episode_log: optional dict with 'writer' (csv.DictWriter) and 'count' (int)
        to append one row per finished season (CAPQL-style dense training curves).
        """
        self.buffer.reset()
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_len = 0
        n_episodes = 0
        episode_returns = []
        r_vec_sum = np.zeros(3, dtype=np.float64)
        pref_sum = np.zeros(n_pref_dims, dtype=np.float64)
        harvest_yields = []
        last_rv = np.zeros(3, dtype=np.float64)

        for _ in range(self.n_steps):
            action, value, log_prob = self.net.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            self.buffer.add(obs, action, reward, value, log_prob, done)
            episode_reward += reward
            episode_len += 1

            if 'reward_vector' in info:
                last_rv = np.asarray(info['reward_vector'], dtype=np.float64)
                r_vec_sum += last_rv
            if len(obs) >= n_pref_dims:
                pref_sum += obs[-n_pref_dims:]

            obs = next_obs

            if done:
                n_episodes += 1
                episode_returns.append(episode_reward)
                fs = info.get('full_state', {})
                totals = info.get('totals', {})
                if fs.get('grnwt'):
                    harvest_yields.append(float(fs['grnwt']))
                if episode_log is not None and episode_log.get('writer') is not None:
                    episode_log['count'] = episode_log.get('count', 0) + 1
                    w = info.get('preference', obs[-n_pref_dims:] if len(obs) >= n_pref_dims else np.ones(3) / 3)
                    episode_log['writer'].writerow({
                        'episode': episode_log['count'],
                        'timestep': episode_log.get('timestep', 0),
                        'ep_length': episode_len,
                        'cum_reward': round(episode_reward, 6),
                        'w_yield': round(float(w[0]), 4),
                        'w_water': round(float(w[1]), 4),
                        'w_fert': round(float(w[2]), 4),
                        'R_yield': round(float(last_rv[0]), 6),
                        'R_water': round(float(last_rv[1]), 6),
                        'R_fert': round(float(last_rv[2]), 6),
                        'yield_kg_ha': round(float(fs.get('grnwt', 0.0) or 0.0), 1),
                        'total_N_kg_ha': round(float(totals.get('nitrogen', 0.0) or 0.0), 1),
                        'total_W_mm': round(float(totals.get('water', 0.0) or 0.0), 1),
                    })
                    episode_log.get('file') and episode_log['file'].flush()
                obs, _ = env.reset()
                episode_reward = 0.0
                episode_len = 0
                last_rv = np.zeros(3, dtype=np.float64)

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, last_value = self.net.forward(obs_t)
            last_value = float(last_value.item())

        self.buffer.compute_returns_and_advantages(last_value, done)

        n = max(self.buffer.ptr, 1)
        stats = {
            'n_episodes': n_episodes,
            'episode_returns': episode_returns[-5:],
            'mean_R_yield': float(r_vec_sum[0] / n),
            'mean_R_water': float(r_vec_sum[1] / n),
            'mean_R_fert': float(r_vec_sum[2] / n),
            'mean_pref': (pref_sum / n).tolist(),
            'harvest_yields': harvest_yields[-3:],
        }
        return self.buffer, stats

    def update(self, buffer=None):
        """
        Multi-epoch PPO gradient update on the collected rollout.

        Total loss:
            L = L_policy + vf_coef * L_value - ent_coef * L_entropy
        """
        if buffer is None:
            buffer = self.buffer

        policy_losses, value_losses, entropies = [], [], []

        for _ in range(self.n_epochs):
            for batch in buffer.get_batches(self.batch_size):
                obs = torch.as_tensor(batch['obs'], device=self.device)
                actions = torch.as_tensor(batch['actions'], device=self.device)
                log_probs_old = torch.as_tensor(batch['log_probs_old'], device=self.device)
                advantages = torch.as_tensor(batch['advantages'], device=self.device)
                returns = torch.as_tensor(batch['returns'], device=self.device)

                log_probs_new, entropy, values = self.net.evaluate_actions(obs, actions)

                pi_loss = self.policy_loss(
                    log_probs_new, log_probs_old, advantages, self.clip_range,
                )
                v_loss = self.value_loss(values, returns)
                ent = self.entropy_bonus(entropy)

                total_loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(pi_loss.item())
                value_losses.append(v_loss.item())
                entropies.append(ent.item())

        self._last_policy_loss = float(np.mean(policy_losses))
        self._last_value_loss = float(np.mean(value_losses))
        self._last_entropy = float(np.mean(entropies))
        self._last_total_loss = (
            self._last_policy_loss
            + self.vf_coef * self._last_value_loss
            - self.ent_coef * self._last_entropy
        )

        return {
            'policy_loss': self._last_policy_loss,
            'value_loss': self._last_value_loss,
            'entropy': self._last_entropy,
            'total_loss': self._last_total_loss,
        }

    def predict(self, obs, deterministic=True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.net.forward(obs_t)
            if deterministic:
                action = torch.clamp(mu, self.net.action_low.to(self.device),
                                     self.net.action_high.to(self.device))
            else:
                dist, _ = self.net._distribution(obs_t)
                action = torch.clamp(dist.sample(), self.net.action_low.to(self.device),
                                     self.net.action_high.to(self.device))
        return action.squeeze(0).cpu().numpy()

    def save(self, path):
        torch.save({
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt['net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
