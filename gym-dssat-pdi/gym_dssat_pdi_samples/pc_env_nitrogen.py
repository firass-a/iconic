"""
SmartFarmGymEnv — Gymnasium adapter for simple PPO over SmartFarmSoSEnv.

Architecture:
    SB3 PPO
       ↓                              ↑
       action (Box, 2)                obs (Box, 11) + scalar reward
       ↓                              ↑
    PCSmartFarmEnv  (this file)
       ↓                              ↑
       {'anfer', 'amir'}              dict obs + scalar reward
       ↓                              ↑
    SmartFarmSoSEnv  (02_smart_farm_env.py)
       ↓                              ↑
       same                           same
       ↓                              ↑
    gym-DSSAT (Fortran simulator)

Action space:
    Box(low=[0, 0], high=[200, 50], shape=(2,), float32)
        index 0: anfer ∈ [0, 200] kg N/ha/day
        index 1: amir  ∈ [0,  50] mm/day

Observation space:
    Box(shape=(11,), float32) — state features normalized to roughly [0, 1]:

        0  topwt              biomass (kg/ha)        / 20000
        1  grnwt              grain (kg/ha)          / 12000
        2  dap                days after planting    / 200
        3  vstage             V-stage                / 18
        4  xlai               leaf area index        / 7
        5  cumsumfert         cumulative N (kg/ha)   / 300
        6  totir              cumulative irrig (mm)  / 1000
        7  sw_mean            mean soil water        (already [0,1])
        8  energy_budget      SoS energy             (already [0,1])
        9  comm_quality       link quality           (already [0,1])
       10  sensors_alive_frac fraction of live sensors (already [0,1])

Reward:
    Scalar from SmartFarmSoSEnv's hierarchical daily + seasonal design.
    VecNormalize (applied in the training script) normalizes it further.

Note — PC-PPO extension:
    When moving to PC-PPO, add N_PREFERENCE_DIMS back to OBS_DIM, sample
    w ~ Dirichlet each episode, concatenate to feats, and scalarize the
    reward vector as r = w · R⃗. This file is intentionally kept simple.

Run smoke test inside Docker:
    python3 /workspace/gym-dssat-pdi/gym_dssat_pdi_samples/pc_env.py
"""
import sys
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from importlib import import_module
SmartFarmSoSEnv = import_module('02_smart_farm_env_nitrogen').SmartFarmSoSEnv


# ================================================================== #
# Normalization constants                                              #
# ================================================================== #
NORM = {
    'topwt':      20000.0,
    'grnwt':      12000.0,
    'dap':          200.0,
    'vstage':        18.0,
    'xlai':           7.0,
    'cumsumfert':   300.0,
    'totir':       1000.0,
}

OBS_DIM     = 11
ACTION_LOW  = np.array([0.0,   0.0], dtype=np.float32)
ACTION_HIGH = np.array([200.0, 50.0], dtype=np.float32)


# ================================================================== #
# Adapter                                                              #
# ================================================================== #
class PCSmartFarmEnv(gym.Env):
    """
    Gymnasium adapter around SmartFarmSoSEnv for simple PPO.

    Flattens the dict observation to an 11-dim float32 vector and passes
    the scalar reward from SmartFarmSoSEnv directly to SB3.
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

    # -------------------------------------------------------------- #
    # Gymnasium API                                                    #
    # -------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):  # noqa: ARG002
        sos_obs = self.sos_env.reset()
        obs     = self._encode_observation(sos_obs)
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()
        anfer  = float(np.clip(action[0], ACTION_LOW[0], ACTION_HIGH[0]))
        amir   = float(np.clip(action[1], ACTION_LOW[1], ACTION_HIGH[1]))

        sos_obs, scalar_reward, done, sos_info = self.sos_env.step({
            'anfer': anfer,
            'amir':  amir,
        })

        obs        = self._encode_observation(sos_obs)
        terminated = bool(done)
        truncated  = False

        info = dict(sos_info)
        info['action_applied'] = np.array([anfer, amir], dtype=np.float32)

        return obs, float(scalar_reward), terminated, truncated, info

    def close(self):
        self.sos_env.close()

    # -------------------------------------------------------------- #
    # Observation encoding                                             #
    # -------------------------------------------------------------- #
    def _encode_observation(self, sos_obs):
        """Flatten and normalize the SoS dict into an 11-dim float32 vector."""
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

        return np.array([
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


# ================================================================== #
# Smoke test                                                           #
# ================================================================== #
def _smoke_test():
    print("=" * 70)
    print("SmartFarmGymEnv — simple PPO smoke test")
    print("=" * 70)

    # ---- 1. Construction ----
    print("\n[1] Constructing PCSmartFarmEnv...")
    env = PCSmartFarmEnv(mode='all', dssat_seed=123,
                         run_dssat_location='run_dssat',
                         enable_faults=False)
    print(f"    action_space      = {env.action_space}")
    print(f"    observation_space = {env.observation_space}")
    assert env.action_space.shape      == (2,),      "action_space wrong shape"
    assert env.observation_space.shape == (OBS_DIM,), "observation_space wrong shape"
    print("    ✓ spaces OK")

    # ---- 2. Reset ----
    print("\n[2] reset()...")
    obs, info = env.reset()
    print(f"    obs shape  = {obs.shape}")
    print(f"    obs dtype  = {obs.dtype}")
    print(f"    obs        = {obs.round(3)}")
    assert obs.shape == (OBS_DIM,),    "obs shape wrong"
    assert obs.dtype == np.float32,    "obs dtype wrong"
    print("    ✓ reset OK")

    # ---- 3. Single step ----
    print("\n[3] step() with fixed action [50 N, 8 mm]...")
    obs, r, term, trunc, info = env.step(np.array([50.0, 8.0], dtype=np.float32))
    print(f"    scalar reward = {r:+.6f}")
    print(f"    terminated    = {term}, truncated = {trunc}")
    print(f"    R_daily       = {info['reward_components']['R_daily']:+.6f}")
    assert obs.shape == (OBS_DIM,), "obs shape wrong after step"
    assert isinstance(r, float),    "reward must be scalar float"
    print("    ✓ step OK")

    # ---- 4. Full episode ----
    print("\n[4] Full random episode...")
    obs, _ = env.reset()
    cum_reward = 0.0
    steps      = 0
    while True:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        cum_reward += r
        steps      += 1
        if term or trunc:
            break

    sos = info['sos_state']
    c   = info['reward_components']
    print(f"    episode length    = {steps} days")
    print(f"    cumulative reward = {cum_reward:+.4f}")
    print(f"    R_seasonal        = {c['R_seasonal']:+.4f}")
    print(f"      R_yield  = {c.get('R_yield', float('nan')):+.4f}  "
          f"(grnwt = {sos['grnwt']:.0f} kg/ha)")
    print(f"      R_hiad   = {c.get('R_hiad',  float('nan')):+.4f}")
    print(f"      R_ane    = {c.get('R_ane',   float('nan')):+.4f}")
    print(f"    total N  = {sos['total_nitrogen']:.0f} kg/ha")
    print(f"    total W  = {sos['total_water']:.0f} mm")
    assert steps > 0, "episode must have at least one step"
    print("    ✓ episode OK")

    env.close()
    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS PASSED.")
    print("Next: update 04_pc_ppo_quick_train.py, then run training.")
    print("=" * 70)


if __name__ == '__main__':
    try:
        _smoke_test()
    except AssertionError as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise
