"""
PCPPOEnv — Gymnasium adapter for Preference-Conditioned PPO.

Architecture:
    SB3 PPO
       ↓                              ↑
       action (Box, 2)                obs (Box, 14) + scalar reward
       ↓                              ↑
    PCPPOEnv  (this file)
       ↓                              ↑
       {'anfer', 'amir'}              dict obs + R_daily + info['reward_vec']
       ↓                              ↑
    SmartFarmSoSEnv (02_smart_farm_env_pcppo.py)
       ↓                              ↑
    gym-DSSAT (Fortran simulator)

Action space:
    Box(low=[0, 0], high=[200, 50], shape=(2,), float32)

Observation space (14 dims):
    [0-10]  same 11 crop+SoS features as pc_env.py
    [11]    w[0] — yield preference     ∈ [0, 1]
    [12]    w[1] — N efficiency pref    ∈ [0, 1]
    [13]    w[2] — water efficiency pref∈ [0, 1]

Reward:
    r = R_daily + dot(w, reward_vec)   (scalar, preference-scalarized)
    reward_vec = [R_yield, R_ane, R_water_eff]  (only non-zero at terminal)

Preference sampling:
    Each episode: w ~ Dirichlet([1, 1, 1])  (uniform over the simplex)
    Stored as self.current_w — logged in info['preference_w']
"""
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from importlib import import_module

SmartFarmSoSEnv = import_module('02_smart_farm_env_pcppo').SmartFarmSoSEnv

N_OBJECTIVES = 3   # [R_yield, R_ane, R_water_eff]
CROP_OBS_DIM = 11
OBS_DIM      = CROP_OBS_DIM + N_OBJECTIVES   # 14

NORM = {
    'topwt':      20000.0,
    'grnwt':      12000.0,
    'dap':          200.0,
    'vstage':        18.0,
    'xlai':           7.0,
    'cumsumfert':   300.0,
    'totir':       1000.0,
}

ACTION_LOW  = np.array([0.0,   0.0], dtype=np.float32)
ACTION_HIGH = np.array([200.0, 50.0], dtype=np.float32)


class PCPPOEnv(gym.Env):
    """
    Gymnasium adapter for PC-PPO.

    At each episode reset, samples w ~ Dirichlet(1,1,1) and appends it
    to the crop observation. The scalar reward is:
        r = R_daily + dot(w, reward_vec)
    where reward_vec = [R_yield, R_ane, R_water_eff] at the terminal step.
    """

    metadata = {'render_modes': []}

    def __init__(self, mode='all', dssat_seed=123,
                 run_dssat_location='run_dssat',
                 enable_faults=False, fault_rate=0.02):
        super().__init__()

        self.sos_env = SmartFarmSoSEnv(
            mode=mode,
            seed=dssat_seed,
            run_dssat_location=run_dssat_location,
            enable_faults=enable_faults,
            fault_rate=fault_rate,
        )

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self.current_w = np.ones(N_OBJECTIVES, dtype=np.float32) / N_OBJECTIVES

    # -------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):  # noqa: ARG002
        # Sample new preference for this episode
        self.current_w = np.random.dirichlet(
            np.ones(N_OBJECTIVES)
        ).astype(np.float32)

        sos_obs = self.sos_env.reset()
        obs     = self._encode_observation(sos_obs)
        return obs, {'preference_w': self.current_w.copy()}

    # -------------------------------------------------------------- #
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()
        anfer  = float(np.clip(action[0], ACTION_LOW[0], ACTION_HIGH[0]))
        amir   = float(np.clip(action[1], ACTION_LOW[1], ACTION_HIGH[1]))

        sos_obs, R_daily, done, sos_info = self.sos_env.step({
            'anfer': anfer,
            'amir':  amir,
        })

        # Scalarize seasonal reward with preference vector
        reward_vec  = sos_info.get('reward_vec', np.zeros(N_OBJECTIVES, dtype=np.float32))
        R_seasonal  = float(np.dot(self.current_w, reward_vec)) if done else 0.0
        scalar_reward = float(R_daily) + R_seasonal

        obs        = self._encode_observation(sos_obs)
        terminated = bool(done)
        truncated  = False

        info = dict(sos_info)
        info['preference_w']    = self.current_w.copy()
        info['R_seasonal']      = R_seasonal
        info['reward_vec']      = reward_vec
        info['action_applied']  = np.array([anfer, amir], dtype=np.float32)

        return obs, scalar_reward, terminated, truncated, info

    # -------------------------------------------------------------- #
    def close(self):
        self.sos_env.close()

    # -------------------------------------------------------------- #
    def _encode_observation(self, sos_obs):
        """Flatten crop+SoS state (11 dims) then append preference w (3 dims)."""
        topwt      = float(sos_obs.get('crop_topwt',      0.0) or 0.0)
        grnwt      = float(sos_obs.get('crop_grnwt',      0.0) or 0.0)
        dap        = float(sos_obs.get('crop_dap',         0.0) or 0.0)
        vstage     = float(sos_obs.get('crop_vstage',      0.0) or 0.0)
        xlai       = float(sos_obs.get('crop_xlai',        0.0) or 0.0)
        cumsumfert = float(sos_obs.get('crop_cumsumfert',  0.0) or 0.0)
        totir      = float(sos_obs.get('crop_totir',       0.0) or 0.0)

        sw_layers = sos_obs.get('crop_sw', None)
        sw_mean   = float(np.mean(np.asarray(sw_layers, dtype=np.float32))) \
                    if sw_layers is not None else 0.0

        energy_budget = float(sos_obs.get('energy_budget', 1.0) or 0.0)
        comm_quality  = float(sos_obs.get('comm_quality',  1.0) or 0.0)

        sensor_vals = [
            float(sos_obs.get(f'sensor_{i}', 1.0) or 0.0)
            for i in range(self.sos_env.n_sensors)
        ]
        sensors_alive_frac = float(np.mean(sensor_vals)) if sensor_vals else 1.0

        crop_feats = np.array([
            topwt      / NORM['topwt'],
            grnwt      / NORM['grnwt'],
            dap        / NORM['dap'],
            vstage     / NORM['vstage'],
            xlai       / NORM['xlai'],
            cumsumfert / NORM['cumsumfert'],
            totir      / NORM['totir'],
            sw_mean,
            energy_budget,
            comm_quality,
            sensors_alive_frac,
        ], dtype=np.float32)

        return np.concatenate([crop_feats, self.current_w])


# ================================================================== #
# Smoke test                                                           #
# ================================================================== #
def _smoke_test():
    print("=" * 70)
    print("PCPPOEnv — preference-conditioned smoke test")
    print("=" * 70)

    env = PCPPOEnv(mode='all', dssat_seed=123,
                   run_dssat_location='run_dssat',
                   enable_faults=False)
    print(f"  action_space      = {env.action_space}")
    print(f"  observation_space = {env.observation_space}")
    assert env.action_space.shape      == (2,)
    assert env.observation_space.shape == (OBS_DIM,)

    obs, info = env.reset()
    w = info['preference_w']
    print(f"\n  reset() — sampled w = {w.round(3)}  (sum={w.sum():.3f})")
    print(f"  obs shape = {obs.shape}  last 3 = {obs[-3:].round(3)}")
    assert obs.shape == (OBS_DIM,)
    assert np.allclose(obs[-3:], w, atol=1e-5), "preference not in obs"

    obs, r, term, trunc, info = env.step(np.array([50.0, 8.0]))
    print(f"\n  step() — scalar_reward = {r:+.4f}")
    assert obs.shape == (OBS_DIM,)

    # Corner preference test
    obs, info = env.reset()
    env.current_w = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    print(f"\n  corner w=[1,0,0] set manually ✓")

    env.close()
    print("\nALL SMOKE TESTS PASSED.")
    print("=" * 70)


if __name__ == '__main__':
    try:
        _smoke_test()
    except AssertionError as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise
