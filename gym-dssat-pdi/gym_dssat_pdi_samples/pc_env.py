"""
PC-PPO adapter for SmartFarmSoSEnv.

This file does NOT modify your existing wrapper. It adds a Gymnasium-
compatible layer on top so that Stable-Baselines3's PPO can train on your
4-objective SoS environment with preference conditioning.

Architecture:
    SB3 PPO
       ↓                                 ↑
       action (Box, 2)                   obs (Box, 15) + scalar reward
       ↓                                 ↑
    PCSmartFarmEnv  (this file)
       ↓                                 ↑
       {'anfer', 'amir'}                 dict obs + 4-vector reward
       ↓                                 ↑
    SmartFarmSoSEnv  (your wrapper)
       ↓                                 ↑
       same                              same
       ↓                                 ↑
    gym-DSSAT (Fortran simulator)

Action space:
    Box(low=[0, 0], high=[200, 50], shape=(2,), float32)
        index 0: anfer ∈ [0, 200] kg N/ha/day  (200 is a generous upper bound;
                  real values rarely exceed 80, but we give PPO room to learn
                  that very large values are wasteful)
        index 1: amir  ∈ [0, 50]  mm/day      (50 leaves headroom above the
                  typical 8-20 mm event)

Observation space:
    Box(shape=(15,), float32) — concatenation of:
        11 state features (normalized to roughly [0, 1] or [-1, 1])
         4 preference dims (the current episode's w, sums to 1)

State features (in order):
    0  topwt              biomass (kg/ha)        / 20000
    1  grnwt              grain (kg/ha)          / 12000
    2  dap                day after planting     / 200
    3  vstage             V-stage                / 18
    4  xlai               leaf area index        / 7
    5  cumsumfert         cumulative N (kg/ha)   / 300
    6  totir              cumulative irrig (mm)  / 1000
    7  sw_mean            mean soil water        (already 0-1 range)
    8  energy_budget      SoS energy 0-1          (wrapper already normalizes)
    9  comm_quality       link quality 0-1
   10  sensors_alive_frac fraction up 0-1

Preference dims (indices 11-14):
   11  w_yield
   12  w_wue
   13  w_energy
   14  w_resilience

Reward:
    Scalar = w · R⃗   (linear scalarization, the canonical PC-PPO form)

Dependencies (often missing in gym-dssat image; install once in container):
    pip install --no-cache-dir gymnasium numpy

Run smoke test inside Docker (mount repo root that contains gym-dssat-pdi):
    docker run --rm --entrypoint /bin/bash -v "$PWD:/workspace" \\
      gym-dssat:debian-bookworm -lc \\
      "pip install -q gymnasium numpy && python3 -u \\
       /workspace/gym-dssat-pdi/gym_dssat_pdi_samples/pc_env.py"
"""
import sys
import numpy as np

import gymnasium as gym
from gymnasium import spaces

# Import the existing SoS wrapper from the sibling module (same directory)
from importlib import import_module
SmartFarmSoSEnv = import_module('02_smart_farm_env').SmartFarmSoSEnv


# ============================================================
# Normalization constants for the 11 state features
# ============================================================
# Each entry is the divisor that maps raw values to roughly [0, 1].
# Picked from typical maize ranges. The policy will still learn even if
# values occasionally exceed 1 — these are scale guides, not hard caps.
NORM = {
    'topwt':      20000.0,
    'grnwt':      12000.0,
    'dap':          200.0,
    'vstage':        18.0,
    'xlai':           7.0,
    'cumsumfert':   300.0,
    'totir':       1000.0,
}

N_STATE_FEATURES   = 11
N_PREFERENCE_DIMS  = 4
OBS_DIM            = N_STATE_FEATURES + N_PREFERENCE_DIMS   # 15

# Action bounds
ACTION_LOW  = np.array([0.0,   0.0], dtype=np.float32)
ACTION_HIGH = np.array([200.0, 50.0], dtype=np.float32)


# ============================================================
# Adapter class
# ============================================================
class PCSmartFarmEnv(gym.Env):
    """Preference-conditioned Gymnasium adapter around SmartFarmSoSEnv."""

    metadata = {'render_modes': []}

    def __init__(self, mode='all', dssat_seed=123,
                 enable_faults=False, fault_rate=0.02,
                 preference=None, rng_seed=None):
        """
        Args:
            mode:           gym-DSSAT mode, must be 'all' for PC-PPO
                            (we control both fertilization and irrigation)
            dssat_seed:     seed forwarded to gym-DSSAT
            enable_faults:  whether the wrapper injects sensor/comm faults
            fault_rate:     daily probability of a sensor failure
            preference:     if given (np.array of shape (4,)), the env uses
                            this FIXED preference every episode. If None,
                            a fresh preference is sampled each reset(). Use
                            fixed=preference at evaluation, None at training.
            rng_seed:       seed for the adapter's own RNG (preference sampler)
        """
        super().__init__()

        # Wrap your existing SoS env (unchanged)
        self.sos_env = SmartFarmSoSEnv(
            mode=mode,
            seed=dssat_seed,
            enable_faults=enable_faults,
            fault_rate=fault_rate,
        )

        # Gymnasium spaces
        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Preference handling
        self._fixed_preference = (
            np.asarray(preference, dtype=np.float32) if preference is not None else None
        )
        self._current_preference = None
        self.rng = np.random.default_rng(rng_seed)

    # ----------------------------------------------------------
    # Gymnasium API
    # ----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """Reset env and sample a new preference for this episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 1. Reset underlying wrapper
        sos_obs = self.sos_env.reset()

        # 2. Sample or fix the preference for this episode
        if self._fixed_preference is not None:
            self._current_preference = self._fixed_preference.copy()
        else:
            self._current_preference = self._sample_preference()

        # 3. Build and return the flat obs
        obs = self._encode_observation(sos_obs)
        info = {'preference': self._current_preference.copy()}
        return obs, info

    def step(self, action):
        """One day step. action is a Box(2,) — [anfer, amir]."""
        # SB3 already clips to [low, high] before calling step(), but be defensive
        action = np.asarray(action, dtype=np.float32).flatten()
        anfer = float(np.clip(action[0], ACTION_LOW[0], ACTION_HIGH[0]))
        amir  = float(np.clip(action[1], ACTION_LOW[1], ACTION_HIGH[1]))

        # Forward to SoS wrapper using its dict-action API
        sos_obs, reward_vector, done, sos_info = self.sos_env.step({
            'anfer': anfer,
            'amir':  amir,
        })

        # Scalarize: r = w · R⃗
        scalar_reward = float(np.dot(self._current_preference, reward_vector))

        # Encode obs for SB3
        obs = self._encode_observation(sos_obs)

        # Gymnasium API distinguishes terminated (natural end) vs truncated
        # (cut short by a time limit). gym-DSSAT ends episodes naturally
        # at harvest, so this is always terminated, never truncated.
        terminated = bool(done)
        truncated  = False

        # Hand reward vector + preference to the caller for logging
        info = dict(sos_info)
        info['reward_vector'] = np.asarray(reward_vector, dtype=np.float32)
        info['preference']    = self._current_preference.copy()
        info['action_applied'] = np.array([anfer, amir], dtype=np.float32)

        return obs, scalar_reward, terminated, truncated, info

    def close(self):
        self.sos_env.close()

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _sample_preference(self):
        """Sample w ~ Dirichlet([1,1,1,1]) — uniform over the 4-simplex."""
        w = self.rng.dirichlet(alpha=np.ones(N_PREFERENCE_DIMS))
        return w.astype(np.float32)

    def _encode_observation(self, sos_obs):
        """Turn the wrapper's dict obs into a flat 15-dim float32 vector."""
        # --- state features ---
        topwt           = float(sos_obs.get('crop_topwt', 0.0) or 0.0)
        grnwt           = float(sos_obs.get('crop_grnwt', 0.0) or 0.0)
        dap             = float(sos_obs.get('crop_dap',   0.0) or 0.0)
        vstage          = float(sos_obs.get('crop_vstage', 0.0) or 0.0)
        xlai            = float(sos_obs.get('crop_xlai',   0.0) or 0.0)
        cumsumfert      = float(sos_obs.get('crop_cumsumfert', 0.0) or 0.0)
        totir           = float(sos_obs.get('crop_totir', 0.0) or 0.0)

        sw_layers = sos_obs.get('crop_sw', None)
        if sw_layers is None:
            sw_mean = 0.0
        else:
            sw_mean = float(np.mean(np.asarray(sw_layers, dtype=np.float32)))

        energy_budget   = float(sos_obs.get('energy_budget', 1.0) or 0.0)
        comm_quality    = float(sos_obs.get('comm_quality',  1.0) or 0.0)

        # sensors are stored as separate keys sensor_0, sensor_1, ...
        sensor_vals = [
            float(sos_obs.get(f'sensor_{i}', 1.0) or 0.0)
            for i in range(self.sos_env.n_sensors)
        ]
        sensors_alive_frac = float(np.mean(sensor_vals)) if sensor_vals else 1.0

        # Normalize
        feats = np.array([
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

        # Concat preference
        obs = np.concatenate([feats, self._current_preference]).astype(np.float32)
        return obs


# ============================================================
# Smoke test  —  run inside Docker:  python3 pc_env.py
# ============================================================
def _smoke_test():
    """Verify the adapter creates, resets, steps, and produces sane shapes."""
    print("=" * 70)
    print("PC-PPO adapter — smoke test")
    print("=" * 70)

    # ---- 1. Construction ----
    print("\n[1] Constructing PCSmartFarmEnv...")
    env = PCSmartFarmEnv(mode='all', dssat_seed=123,
                         enable_faults=False, rng_seed=0)
    print(f"    action_space      = {env.action_space}")
    print(f"    observation_space = {env.observation_space}")
    assert env.action_space.shape == (2,),  "action_space wrong shape"
    assert env.observation_space.shape == (15,), "observation_space wrong shape"
    print("    ✓ spaces OK")

    # ---- 2. Reset ----
    print("\n[2] reset()...")
    obs, info = env.reset(seed=0)
    print(f"    obs shape      = {obs.shape}")
    print(f"    obs dtype      = {obs.dtype}")
    print(f"    obs[:11]       = {obs[:11].round(3)}   (state features)")
    print(f"    obs[11:15]     = {obs[11:].round(3)}   (preference w)")
    print(f"    w sums to      = {obs[11:].sum():.4f}   (should be ~1.0)")
    assert obs.shape == (15,), "obs shape wrong"
    assert obs.dtype == np.float32, "obs dtype wrong"
    assert abs(obs[11:].sum() - 1.0) < 1e-3, "preference doesn't sum to 1"
    print("    ✓ reset OK")

    # ---- 3. Step with a fixed action ----
    print("\n[3] step() with a fixed mid-range action [50 N, 8 mm]...")
    fixed_action = np.array([50.0, 8.0], dtype=np.float32)
    obs, r, term, trunc, info = env.step(fixed_action)
    print(f"    scalar reward  = {r:+.4f}")
    print(f"    reward_vector  = {info['reward_vector'].round(4)}")
    print(f"    terminated     = {term}, truncated = {trunc}")
    print(f"    preference     = {info['preference'].round(3)}")
    assert obs.shape == (15,), "obs shape wrong after step"
    assert isinstance(r, float), "reward must be scalar float"
    print("    ✓ step OK")

    # ---- 4. Full episode with random actions ----
    print("\n[4] Full random episode...")
    obs, info = env.reset(seed=1)
    fixed_w = info['preference']
    cum_vector = np.zeros(4)
    cum_scalar = 0.0
    steps = 0
    while True:
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        cum_vector += info['reward_vector']
        cum_scalar += r
        steps += 1
        if term or trunc:
            break
    print(f"    episode length     = {steps} days")
    print(f"    preference (fixed) = {fixed_w.round(3)}")
    print(f"    cumulative R⃗       = {cum_vector.round(3)}")
    print(f"    cumulative scalar  = {cum_scalar:+.3f}")
    print(f"    sanity check       = w·R⃗ ≈ {np.dot(fixed_w, cum_vector):+.3f}")
    print("    ✓ episode OK")

    # ---- 5. Preference variation ----
    print("\n[5] Verify preference changes across resets...")
    seen = []
    for i in range(3):
        _, info = env.reset(seed=10 + i)
        seen.append(info['preference'].copy())
        print(f"    reset {i}: w = {info['preference'].round(3)}")
    distinct = len({tuple(w.round(3)) for w in seen})
    assert distinct == 3, "preferences should differ across resets"
    print(f"    ✓ {distinct} distinct preferences over 3 resets")

    # ---- 6. Fixed-preference mode (for evaluation) ----
    print("\n[6] Verify fixed-preference mode for evaluation...")
    eval_env = PCSmartFarmEnv(
        mode='all',
        dssat_seed=123,
        preference=np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32),
        rng_seed=0,
    )
    _, info1 = eval_env.reset(seed=20)
    _, info2 = eval_env.reset(seed=21)
    assert np.allclose(info1['preference'], info2['preference']), \
        "fixed preference should not change across resets"
    print(f"    preference (both resets) = {info1['preference'].round(3)}")
    print("    ✓ fixed-preference mode OK")
    eval_env.close()

    env.close()
    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS PASSED.")
    print("Next: run a 512-step SB3 sanity check, then 50k quick training.")
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
