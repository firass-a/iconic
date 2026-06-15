"""
PC-PPO adapter for SmartFarmSoSEnv (3-objective MORL).

Objectives (preference-conditioned):
    w[0] — yield
    w[1] — water efficiency / minimal irrigation
    w[2] — fertilizer efficiency / minimal N use

Scalar reward: r = w · R⃗   (linear scalarization)

Observation (14-dim):
    11 state features + 3 preference weights

Run smoke test in Docker:
    python3 pc_env.py
"""
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from importlib import import_module
SmartFarmSoSEnv = import_module('02_smart_farm_env').SmartFarmSoSEnv

NORM = {
    'dap': 200.0,
    'vstage': 18.0,
    'xlai': 7.0,
    'swfac': 1.0,
    'nstres': 1.0,
    'grnwt': 12000.0,
    'topwt': 20000.0,
    'cumsumfert': 300.0,
    'totir': 1000.0,
    'rain': 50.0,
    'tmax': 45.0,
}

N_STATE_FEATURES = 11
N_PREFERENCE_DIMS = 3
OBS_DIM = N_STATE_FEATURES + N_PREFERENCE_DIMS

ACTION_LOW = np.array([0.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.array([200.0, 50.0], dtype=np.float32)


class PCSmartFarmEnv(gym.Env):
    """Preference-conditioned Gymnasium adapter around SmartFarmSoSEnv."""

    metadata = {'render_modes': []}

    def __init__(self, mode='all', dssat_seed=123, run_dssat_location='run_dssat',
                 random_weather=True, enable_faults=False, fault_rate=0.02,
                 preference=None, rng_seed=None):
        super().__init__()

        self.sos_env = SmartFarmSoSEnv(
            mode=mode,
            seed=dssat_seed,
            run_dssat_location=run_dssat_location,
            random_weather=random_weather,
            enable_faults=enable_faults,
            fault_rate=fault_rate,
        )

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self._fixed_preference = (
            np.asarray(preference, dtype=np.float32) if preference is not None else None
        )
        self._current_preference = None
        self.rng = np.random.default_rng(rng_seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        sos_obs = self.sos_env.reset()

        if self._fixed_preference is not None:
            self._current_preference = self._fixed_preference.copy()
        else:
            self._current_preference = self._sample_preference()

        obs = self._encode_observation(sos_obs)
        info = {'preference': self._current_preference.copy()}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()
        anfer = float(np.clip(action[0], ACTION_LOW[0], ACTION_HIGH[0]))
        amir = float(np.clip(action[1], ACTION_LOW[1], ACTION_HIGH[1]))

        sos_obs, reward_vector, done, sos_info = self.sos_env.step({
            'anfer': anfer,
            'amir': amir,
        })

        scalar_reward = float(np.dot(self._current_preference, reward_vector))
        obs = self._encode_observation(sos_obs)

        info = dict(sos_info)
        info['reward_vector'] = np.asarray(reward_vector, dtype=np.float32)
        info['preference'] = self._current_preference.copy()
        info['action_applied'] = np.array([anfer, amir], dtype=np.float32)

        return obs, scalar_reward, bool(done), False, info

    def close(self):
        self.sos_env.close()

    def _sample_preference(self):
        w = self.rng.dirichlet(alpha=np.ones(N_PREFERENCE_DIMS))
        return w.astype(np.float32)

    def _encode_observation(self, sos_obs):
        def g(key, default=0.0):
            return float(sos_obs.get(key, default) or default)

        feats = np.array([
            g('crop_dap') / NORM['dap'],
            g('crop_vstage') / NORM['vstage'],
            g('crop_xlai') / NORM['xlai'],
            g('crop_swfac') / NORM['swfac'],
            g('crop_nstres') / NORM['nstres'],
            g('moisture_ratio'),
            g('crop_grnwt') / NORM['grnwt'],
            g('crop_topwt') / NORM['topwt'],
            g('crop_cumsumfert') / NORM['cumsumfert'],
            g('crop_totir') / NORM['totir'],
            g('crop_rain') / NORM['rain'],
        ], dtype=np.float32)

        obs = np.concatenate([feats, self._current_preference]).astype(np.float32)
        return obs


def _smoke_test():
    print('=' * 70)
    print('PC-PPO adapter (3 objectives) — smoke test')
    print('=' * 70)

    env = PCSmartFarmEnv(mode='all', dssat_seed=123, rng_seed=0)
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (OBS_DIM,)

    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert abs(obs[N_STATE_FEATURES:].sum() - 1.0) < 1e-3

    obs, r, term, trunc, info = env.step(np.array([20.0, 5.0], dtype=np.float32))
    assert len(info['reward_vector']) == 3
    print(f'  reward_vector = {info["reward_vector"].round(4)}')
    print(f'  scalar r      = {r:+.4f}')
    print(f'  moisture (obs)  = {obs[5]:.3f}')

    env.close()
    print('\nALL SMOKE TESTS PASSED.')


if __name__ == '__main__':
    try:
        _smoke_test()
    except Exception as e:
        print(f'\nSMOKE TEST FAILED: {e}', file=sys.stderr)
        raise
