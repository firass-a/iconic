"""
Rollout buffer + GAE advantage estimation + return computation.

Stores one on-policy batch collected from the environment, then computes
advantages and returns before the PPO update phase.
"""
import numpy as np


class RolloutBuffer:
    """
    Fixed-size buffer for one PPO rollout (n_steps transitions).

    After collection, call compute_returns_and_advantages(last_value, last_done)
    to fill `advantages` and `returns` using GAE(λ).
    """

    def __init__(self, buffer_size, obs_dim, act_dim, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        self.ptr = 0
        self.full = False

        self.observations = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)

        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)

    def add(self, obs, action, reward, value, log_prob, done):
        if self.ptr >= self.buffer_size:
            raise RuntimeError('RolloutBuffer is full — call reset() before new rollout')

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_value, last_done):
        """
        GAE(λ) advantage estimation and return computation.

            δ_t = r_t + γ V(s_{t+1}) (1 - done_t) - V(s_t)
            A_t = Σ_l (γλ)^l δ_{t+l}
            R_t = A_t + V(s_t)

        Args:
            last_value: V(s_T) bootstrap after final collected step
            last_done:  whether the episode ended on the last step
        """
        last_gae = 0.0
        size = self.ptr

        for t in reversed(range(size)):
            if t == size - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:size] = self.advantages[:size] + self.values[:size]

        # Normalize advantages (stabilizes PPO updates)
        adv = self.advantages[:size]
        self.advantages[:size] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def get_batches(self, batch_size):
        """Yield random mini-batches for multi-epoch PPO updates."""
        size = self.ptr
        indices = np.arange(size)
        np.random.shuffle(indices)

        for start in range(0, size, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield {
                'obs': self.observations[batch_idx],
                'actions': self.actions[batch_idx],
                'log_probs_old': self.log_probs[batch_idx],
                'advantages': self.advantages[batch_idx],
                'returns': self.returns[batch_idx],
            }
