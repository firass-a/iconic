"""
Unit test for custom PPO components — no DSSAT / Docker required.

Verifies:
  rollout buffer → GAE → returns → policy loss → value loss → entropy → grad step

Run locally:
    pip install torch numpy gymnasium
    python ppo_unit_test.py
"""
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from ppo_rollout_buffer import RolloutBuffer
from ppo_agent import PPOAgent


class MockFarmEnv(gym.Env):
    """Tiny stand-in for PCSmartFarmEnv."""

    def __init__(self, obs_dim=14, seed=0):
        super().__init__()
        self.observation_space = spaces.Box(-1, 1, (obs_dim,), np.float32)
        self.action_space = spaces.Box(
            np.array([0.0, 0.0], np.float32),
            np.array([200.0, 50.0], np.float32),
        )
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        return self.rng.random(self.observation_space.shape, np.float32), {}

    def step(self, action):
        self.step_count += 1
        obs = self.rng.random(self.observation_space.shape, np.float32)
        reward = float(self.rng.random() - 0.3)
        done = self.step_count >= 50
        return obs, reward, done, False, {}


def test_rollout_buffer_gae():
    buf = RolloutBuffer(10, obs_dim=4, act_dim=2, gamma=0.99, gae_lambda=0.95)
    for i in range(10):
        buf.add(
            obs=np.ones(4) * i,
            action=np.array([1.0, 2.0]),
            reward=1.0,
            value=0.5,
            log_prob=-0.5,
            done=(i == 9),
        )
    buf.compute_returns_and_advantages(last_value=0.0, last_done=True)
    assert buf.advantages.shape == (10,)
    assert buf.returns.shape == (10,)
    assert not np.allclose(buf.advantages, 0)
    print('  [OK] RolloutBuffer + GAE + returns')


def test_ppo_losses():
    log_new = torch.tensor([0.0, -0.1, 0.2])
    log_old = torch.tensor([0.0, 0.0, 0.0])
    adv = torch.tensor([1.0, -0.5, 2.0])
    pi = PPOAgent.policy_loss(log_new, log_old, adv, clip_range=0.2)
    v = PPOAgent.value_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 2.5]))
    ent = PPOAgent.entropy_bonus(torch.tensor([0.5, 0.6]))
    assert pi.ndim == 0 and v.ndim == 0
    print(f'  [OK] policy_loss={pi.item():.4f}  value_loss={v.item():.4f}  entropy={ent.item():.4f}')


def test_full_train_loop():
    env = MockFarmEnv(obs_dim=14)
    agent = PPOAgent(
        obs_dim=14, act_dim=2,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        n_steps=128, batch_size=32, n_epochs=4,
    )
    for _ in range(3):
        buf, _ = agent.collect_rollout(env)
        m = agent.update(buf)
    assert 'policy_loss' in m
    obs, _ = env.reset()
    action = agent.predict(obs)
    assert action.shape == (2,)
    print(f'  [OK] 3 PPO updates  L_pi={m["policy_loss"]:+.4f}  L_v={m["value_loss"]:.4f}')


if __name__ == '__main__':
    print('=' * 60)
    print('PPO unit test (no Docker)')
    print('=' * 60)
    test_rollout_buffer_gae()
    test_ppo_losses()
    test_full_train_loop()
    print('=' * 60)
    print('ALL PASSED — run 05_pc_ppo_custom_train.py in Docker when ready')
    print('=' * 60)
