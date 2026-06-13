"""
capql_env_robust.py  —  CAPQL-Robust training environment.

Identical reward structure and objectives to CAPQLEnv v2 (3 objectives:
yield, N-efficiency, water).  Two additions:

  1. Realistic fault injection during training (FaultyEnvV2)
     Each episode randomly samples a fault profile from the catalogue
     below.  60% of episodes are fault-free so the base policy is not
     degraded by always training under stress.

  2. Extended observation (19-dim):
     obs = faulted_crop_features(11) + mask_flags(5) + preference_w(3)

     mask_flags[i] = 1  sensor i alive / trusted
                   = 0  sensor i dead  / packet lost  (Type 1 only)
     Type 2 faults (stuck, bias, noise) leave mask = 1 — the agent
     must infer them from temporal patterns via the GRU.

Fault catalogue (one profile drawn per episode reset):
─────────────────────────────────────────────────────
  60%  clean           no faults
  10%  sensor_dropout  random sensor(s) dropout_rate ~ U(0.05, 0.30)
  10%  packet_loss     packet_loss_prob  ~ U(0.05, 0.30)
  10%  stuck_sensor    random sensor freezes at day ~ U(10, 100)
   5%  actuator_N      N_efficiency ~ U(0.20, 0.80)
   5%  actuator_W      W_efficiency ~ U(0.20, 0.80)

Observation space : Box(-2, 2, shape=(19,))
Action space      : Box([0,0], [200,50], shape=(2,))
"""

import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from importlib import import_module

SmartFarmSoSEnv = import_module('02_smart_farm_env_pcppo').SmartFarmSoSEnv
FaultyEnvV2     = import_module('faulty_env_v2').FaultyEnvV2

N_OBJECTIVES = 3
CROP_DIM     = 11
MASK_DIM     = 5           # one flag per sensor
OBS_DIM      = CROP_DIM + MASK_DIM + N_OBJECTIVES   # 19

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

_CORNERS = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)
CORNER_PROB = 0.20   # 20% corner injection (same as v2)

# ──────────────────────────────────────────────────────────────────────
# Fault catalogue probabilities (must sum to 1.0)
# ──────────────────────────────────────────────────────────────────────
FAULT_PROBS = {
    'clean':          0.60,
    'sensor_dropout': 0.10,
    'packet_loss':    0.10,
    'stuck_sensor':   0.10,
    'actuator_N':     0.05,
    'actuator_W':     0.05,
}


def _sample_fault_kwargs():
    """Draw a random fault profile for one episode."""
    r = np.random.random()
    cumulative = 0.0
    for fault_type, prob in FAULT_PROBS.items():
        cumulative += prob
        if r < cumulative:
            break

    if fault_type == 'clean':
        return {}

    if fault_type == 'sensor_dropout':
        rate = float(np.random.uniform(0.05, 0.30))
        # pick 1-3 random sensors to be affected
        n = np.random.randint(1, 4)
        sensors = list(np.random.choice(5, size=n, replace=False))
        # we model as overall dropout_rate applied to all
        # (individual sensor selection handled inside FaultyEnvV2 via the
        #  dropout_rate applying to every alive sensor each step)
        return {'dropout_rate': rate}

    if fault_type == 'packet_loss':
        prob_loss = float(np.random.uniform(0.05, 0.30))
        return {'packet_loss_prob': prob_loss}

    if fault_type == 'stuck_sensor':
        sensor_idx = int(np.random.randint(0, 5))
        day        = int(np.random.randint(10, 100))
        return {'stuck_sensors': [sensor_idx], 'stuck_day': day}

    if fault_type == 'actuator_N':
        eta = float(np.random.uniform(0.20, 0.80))
        return {'N_efficiency': eta}

    if fault_type == 'actuator_W':
        eta = float(np.random.uniform(0.20, 0.80))
        return {'W_efficiency': eta}

    return {}   # fallback: clean


# ══════════════════════════════════════════════════════════════════════
class CAPQLRobustEnv(gym.Env):
    """
    CAPQL-Robust training environment.

    Wraps SmartFarmSoSEnv inside FaultyEnvV2.  A new FaultyEnvV2 is
    created at every reset() call with a freshly-sampled fault profile,
    so each episode can have a different fault type and severity.

    The inner DSSAT env is created once and reused to avoid the overhead
    of spawning a new Fortran process every episode.
    """

    metadata = {'render_modes': []}

    def __init__(self, mode='all', dssat_seed=123,
                 run_dssat_location='run_dssat'):
        super().__init__()

        self._sos_env = SmartFarmSoSEnv(
            mode=mode,
            seed=dssat_seed,
            run_dssat_location=run_dssat_location,
            enable_faults=False,   # all faults handled by FaultyEnvV2
        )

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(2,), dtype=np.float32
        )

        self.current_w  = np.full(N_OBJECTIVES, 1.0 / N_OBJECTIVES,
                                  dtype=np.float32)
        self._max_anfer = 200.0
        self._max_amir  =  50.0
        self._faulty    = None   # FaultyEnvV2 instance, rebuilt each episode

    # ──────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        # Corner injection (same 20/80 split as v2)
        r = np.random.random()
        if r < CORNER_PROB:
            idx            = min(int(r / (CORNER_PROB / 3)), 2)
            self.current_w = _CORNERS[idx].copy()
        else:
            self.current_w = np.random.dirichlet(
                np.ones(N_OBJECTIVES)
            ).astype(np.float32)

        # Per-episode action caps (squared-law, same as v2)
        w_neff  = float(self.current_w[1])
        w_water = float(self.current_w[2])
        self._max_anfer = float(np.clip(200.0 * (1.0 - w_neff) ** 2,  2.0, 200.0))
        self._max_amir  = float(np.clip( 50.0 * (1.0 - w_water) ** 2, 3.0,  50.0))

        # Build a fresh FaultyEnvV2 with a new random fault profile
        fault_kwargs = _sample_fault_kwargs()
        self._faulty  = _InnerFaultWrapper(
            self._sos_env, fault_kwargs, self.current_w,
            self._max_anfer, self._max_amir,
        )

        sos_obs = self._sos_env.reset()
        raw_obs = self._encode(sos_obs)
        faulted_obs = self._faulty.corrupt(raw_obs)
        return faulted_obs, {'preference_w': self.current_w.copy(),
                             'fault_profile': fault_kwargs}

    # ──────────────────────────────────────────────────────────────
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()
        anfer  = float(np.clip(action[0], 0.0, self._max_anfer))
        amir   = float(np.clip(action[1], 0.0, self._max_amir))

        # Actuator fault: scale before DSSAT receives
        anfer_eff = anfer * self._faulty.N_efficiency
        amir_eff  = amir  * self._faulty.W_efficiency

        sos_obs, R_daily, done, sos_info = self._sos_env.step(
            {'anfer': anfer_eff, 'amir': amir_eff}
        )

        raw_obs = self._encode(sos_obs)
        faulted_obs = self._faulty.step_corrupt(raw_obs)

        info = dict(sos_info)
        info['preference_w']    = self.current_w.copy()
        info['action_applied']  = np.array([anfer, amir], dtype=np.float32)
        info['action_effective']= np.array([anfer_eff, amir_eff],
                                            dtype=np.float32)
        return faulted_obs, float(R_daily), bool(done), False, info

    # ──────────────────────────────────────────────────────────────
    def close(self):
        self._sos_env.close()

    # ──────────────────────────────────────────────────────────────
    def _encode(self, sos_obs):
        """Encode raw SoS obs dict → 14-dim clean obs (same as v2)."""
        def _g(k):
            return float(sos_obs.get(f'crop_{k}', 0.0) or 0.0)

        sw_layers   = sos_obs.get('crop_sw', None)
        sw_mean     = float(np.mean(np.asarray(sw_layers, dtype=np.float32))) \
                      if sw_layers is not None else 0.0
        sensor_vals = [
            float(sos_obs.get(f'sensor_{i}', 1.0) or 0.0)
            for i in range(self._sos_env.n_sensors)
        ]
        sensors_frac = float(np.mean(sensor_vals)) if sensor_vals else 1.0

        crop_feats = np.array([
            _g('topwt')      / NORM['topwt'],
            _g('grnwt')      / NORM['grnwt'],
            _g('dap')        / NORM['dap'],
            _g('vstage')     / NORM['vstage'],
            _g('xlai')       / NORM['xlai'],
            _g('cumsumfert') / NORM['cumsumfert'],
            _g('totir')      / NORM['totir'],
            sw_mean,
            float(sos_obs.get('energy_budget', 1.0) or 0.0),
            float(sos_obs.get('comm_quality',  1.0) or 0.0),
            sensors_frac,
        ], dtype=np.float32)

        return np.concatenate([crop_feats, self.current_w])   # 14-dim


# ══════════════════════════════════════════════════════════════════════
# Internal helper — stateful fault injector (no gym.Env overhead)
# ══════════════════════════════════════════════════════════════════════
from faulty_env_v2 import (
    N_SENSORS, SENSOR_FEATURES, _ALIVE, _DROPOUT, _STUCK
)


class _InnerFaultWrapper:
    """
    Lightweight stateful fault injector that operates directly on
    numpy arrays (no gym.Env wrapping — that is CAPQLRobustEnv's job).
    Mirrors FaultyEnvV2._tick() and FaultyEnvV2._apply() logic.
    """

    def __init__(self, sos_env, fault_kwargs, current_w,
                 max_anfer, max_amir):
        self.N_efficiency     = float(fault_kwargs.get('N_efficiency', 1.0))
        self.W_efficiency     = float(fault_kwargs.get('W_efficiency', 1.0))
        self.dropout_rate     = float(fault_kwargs.get('dropout_rate', 0.0))
        self.packet_loss_prob = float(fault_kwargs.get('packet_loss_prob', 0.0))
        self.stuck_sensors    = set(fault_kwargs.get('stuck_sensors', []))
        self.stuck_day        = int(fault_kwargs.get('stuck_day', 50))
        self.sensor_bias      = dict(fault_kwargs.get('sensor_bias', {}))
        self.noise_std        = float(fault_kwargs.get('noise_std', 0.0))

        self._states     = [_ALIVE] * N_SENSORS
        self._frozen     = np.zeros(11, dtype=np.float32)
        self._last_valid = np.zeros(11, dtype=np.float32)
        self._day        = 0

    # ── first call (at reset, day 0) ──────────────────────────────
    def corrupt(self, obs14):
        self._last_valid = obs14[:11].copy()
        self._frozen     = obs14[:11].copy()
        return self._apply(obs14)

    # ── subsequent calls (each step) ──────────────────────────────
    def step_corrupt(self, obs14):
        self._day += 1
        self._tick(obs14)
        return self._apply(obs14)

    def _tick(self, clean_obs14):
        for i in range(N_SENSORS):
            if self._states[i] != _ALIVE:
                continue
            for j in SENSOR_FEATURES[i]:
                self._last_valid[j] = clean_obs14[j]
            if i in self.stuck_sensors and self._day >= self.stuck_day:
                self._states[i] = _STUCK
                for j in SENSOR_FEATURES[i]:
                    self._frozen[j] = clean_obs14[j]
            elif np.random.random() < self.dropout_rate:
                self._states[i] = _DROPOUT

    def _apply(self, obs14):
        obs  = obs14.copy()
        mask = np.ones(N_SENSORS, dtype=np.float32)

        for i in range(N_SENSORS):
            feats = SENSOR_FEATURES[i]
            state = self._states[i]

            if state == _DROPOUT:
                for j in feats: obs[j] = 0.0
                mask[i] = 0.0
                continue

            if state == _STUCK:
                for j in feats: obs[j] = self._frozen[j]
                continue

            # alive
            if self.packet_loss_prob > 0.0 and \
               np.random.random() < self.packet_loss_prob:
                for j in feats: obs[j] = 0.0
                mask[i] = 0.0
                continue

            if i in self.sensor_bias:
                b = float(self.sensor_bias[i])
                for j in feats:
                    obs[j] = float(np.clip(obs[j] + b, -2.0, 2.0))

            if self.noise_std > 0.0:
                for j in feats:
                    obs[j] = float(np.clip(
                        obs[j] + np.random.normal(0.0, self.noise_std),
                        -2.0, 2.0
                    ))

        obs[10] = float(np.mean(mask))

        # Return 19-dim: faulted_crop(11) | mask(5) | w(3)
        return np.concatenate([obs[:11], mask, obs[11:14]])
