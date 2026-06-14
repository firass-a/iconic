"""
capql_env_robust_v2.py  —  CAPQL-Robust v2 training environment.

Extends capql_env_robust.py with:
  1. Full fault catalogue (10 fault types instead of 6)
  2. Three-phase curriculum: fault severity ramps with total_steps
  3. fault_type returned in info for per-fault logging

Fault catalogue (drawn per episode reset):
──────────────────────────────────────────
  Phase 1  (0 → 500K steps)   80% clean, mild faults only
  Phase 2  (500K → 1.5M)      50% clean, all types, medium severity
  Phase 3  (1.5M → 3M+)       30% clean, all types, full severity + combos

Observation: 19-dim = faulted_crop(11) | mask(5) | w(3)
Action:       2-dim  = [anfer kg/ha, amir mm]
"""

import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from importlib import import_module

SmartFarmSoSEnv = import_module('02_smart_farm_env_pcppo').SmartFarmSoSEnv
FaultyEnvV3     = import_module('faulty_env_v3').FaultyEnvV3

N_OBJECTIVES = 3
CROP_DIM     = 11
MASK_DIM     = 5
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

_CORNERS    = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
CORNER_PROB = 0.20


# ══════════════════════════════════════════════════════════════════════
# Curriculum phase → fault distribution
# ══════════════════════════════════════════════════════════════════════
_CATALOGUE = {
    # Each phase: list of (fault_type, weight)
    1: [
        ('clean',           8.0),
        ('noise',           1.0),
        ('packet_loss',     1.0),
    ],
    2: [
        ('clean',           5.0),
        ('noise',           0.8),
        ('packet_loss',     0.8),
        ('dropout',         0.8),
        ('stuck',           0.8),
        ('offset',          0.5),
        ('gain',            0.5),
        ('actuator_eff_N',  0.5),
        ('actuator_eff_W',  0.5),
    ],
    3: [
        ('clean',           3.0),
        ('noise',           0.7),
        ('packet_loss',     0.7),
        ('dropout',         0.7),
        ('stuck',           0.7),
        ('offset',          0.7),
        ('gain',            0.7),
        ('spike',           0.5),
        ('out_of_bounds',   0.5),
        ('actuator_eff_N',  0.6),
        ('actuator_eff_W',  0.6),
        ('actuator_stuck_N',0.4),
        ('actuator_stuck_W',0.4),
        ('combo',           0.5),
    ],
}

# Pre-normalise weights to probabilities
_PROBS = {}
for ph, items in _CATALOGUE.items():
    types, weights = zip(*items)
    w = np.array(weights, dtype=np.float64)
    _PROBS[ph] = (list(types), w / w.sum())


def _get_phase(total_steps):
    if total_steps < 500_000:
        return 1
    if total_steps < 1_500_000:
        return 2
    return 3


def _sample_fault(phase):
    """Return (fault_type_str, kwargs_for_FaultyEnvV3)."""
    types, probs = _PROBS[phase]
    ft = np.random.choice(types, p=probs)

    rng = np.random.default_rng()

    # severity scales with phase
    sev = {1: 0.4, 2: 0.7, 3: 1.0}[phase]

    if ft == 'clean':
        return 'clean', {}

    if ft == 'noise':
        std = float(np.random.uniform(0.02, 0.12 * sev))
        return ft, {'noise_std': std}

    if ft == 'packet_loss':
        p = float(np.random.uniform(0.05, 0.30 * sev))
        return ft, {'packet_loss_prob': p}

    if ft == 'dropout':
        rate = float(np.random.uniform(0.03, 0.25 * sev))
        return ft, {'dropout_rate': rate}

    if ft == 'stuck':
        s = int(np.random.randint(0, 5))
        day = int(np.random.uniform(10, max(11, int(100 * sev))))
        return ft, {'stuck_sensors': [s], 'stuck_day': day}

    if ft == 'offset':
        s = int(np.random.choice([0, 1, 2]))   # soil / canopy / grain sensors
        b = float(np.random.uniform(-0.20 * sev, 0.30 * sev))
        return ft, {'sensor_bias': {s: b}}

    if ft == 'gain':
        s = int(np.random.choice([0, 1]))
        g_range = max(0.1, 1.0 - 0.5 * sev), 1.0 + 0.5 * sev
        g = float(np.random.uniform(*g_range))
        return ft, {'sensor_gain': {s: g}}

    if ft == 'spike':
        sensors = list(np.random.choice(5, size=np.random.randint(1, 3), replace=False))
        p = float(np.random.uniform(0.03, 0.10 * sev))
        return ft, {'spike_prob': p, 'spike_sensors': sensors}

    if ft == 'out_of_bounds':
        s = int(np.random.randint(0, 5))
        side = 'high' if np.random.random() > 0.5 else 'low'
        return ft, {'oob_sensors': {s: side}}

    if ft == 'actuator_eff_N':
        eta = float(np.random.uniform(max(0.1, 1.0 - 0.8 * sev), 0.90))
        return ft, {'N_efficiency': eta}

    if ft == 'actuator_eff_W':
        eta = float(np.random.uniform(max(0.1, 1.0 - 0.8 * sev), 0.90))
        return ft, {'W_efficiency': eta}

    if ft == 'actuator_stuck_N':
        # stuck at 0 (jammed closed) or small value
        val = float(np.random.choice([0.0, float(np.random.uniform(0, 30 * sev))]))
        return ft, {'actuator_stuck_anfer': val}

    if ft == 'actuator_stuck_W':
        val = float(np.random.choice([0.0, float(np.random.uniform(0, 15 * sev))]))
        return ft, {'actuator_stuck_amir': val}

    if ft == 'combo':
        # noise + one other mild fault
        std = float(np.random.uniform(0.02, 0.08))
        kwargs = {'noise_std': std}
        # pick a second fault that doesn't conflict
        second = np.random.choice(['packet_loss', 'offset', 'gain'])
        if second == 'packet_loss':
            kwargs['packet_loss_prob'] = float(np.random.uniform(0.05, 0.15))
        elif second == 'offset':
            s = int(np.random.choice([0, 1]))
            kwargs['sensor_bias'] = {s: float(np.random.uniform(-0.10, 0.15))}
        elif second == 'gain':
            s = int(np.random.choice([0, 1]))
            kwargs['sensor_gain'] = {s: float(np.random.uniform(0.75, 1.25))}
        return ft, kwargs

    return 'clean', {}


# ══════════════════════════════════════════════════════════════════════
class CAPQLRobustEnvV2(gym.Env):
    """
    CAPQL-Robust v2 training environment.

    Extra attributes
    ----------------
    curriculum_phase : int (1/2/3)
        Set by the training loop to control fault severity.  Default 1.
    total_steps : int
        Set by training loop; used to auto-compute phase if curriculum_phase
        is not set manually.
    """

    metadata = {'render_modes': []}

    def __init__(self, mode='all', dssat_seed=123,
                 run_dssat_location='run_dssat'):
        super().__init__()

        self._sos_env = SmartFarmSoSEnv(
            mode=mode,
            seed=dssat_seed,
            run_dssat_location=run_dssat_location,
            enable_faults=False,
        )

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(2,), dtype=np.float32
        )

        self.current_w        = np.full(N_OBJECTIVES, 1/N_OBJECTIVES, dtype=np.float32)
        self._max_anfer       = 200.0
        self._max_amir        =  50.0
        self._fault_state     = None   # _FaultStateV3 for current episode
        self.curriculum_phase = 1
        self.total_steps      = 0
        self._last_fault_type = 'clean'

    # ──────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        phase = _get_phase(self.total_steps) \
                if self.curriculum_phase == 0 else self.curriculum_phase

        # Preference weight
        r = np.random.random()
        if r < CORNER_PROB:
            idx = min(int(r / (CORNER_PROB / 3)), 2)
            self.current_w = _CORNERS[idx].copy()
        else:
            self.current_w = np.random.dirichlet(
                np.ones(N_OBJECTIVES)
            ).astype(np.float32)

        # Per-episode action caps
        w_neff  = float(self.current_w[1])
        w_water = float(self.current_w[2])
        self._max_anfer = float(np.clip(200.0 * (1.0 - w_neff) ** 2,  2.0, 200.0))
        self._max_amir  = float(np.clip( 50.0 * (1.0 - w_water) ** 2, 3.0,  50.0))

        fault_type, fault_kwargs = _sample_fault(phase)
        self._fault_state     = _FaultStateV3(fault_kwargs)
        self._last_fault_type = fault_type

        sos_obs = self._sos_env.reset()
        raw_obs = self._encode(sos_obs)
        obs19   = self._fault_state.corrupt(raw_obs)

        return obs19, {
            'preference_w':   self.current_w.copy(),
            'fault_type':     fault_type,
            'fault_kwargs':   fault_kwargs,
            'phase':          phase,
        }

    # ──────────────────────────────────────────────────────────────
    def step(self, action):
        action   = np.asarray(action, dtype=np.float32).flatten()
        anfer    = float(np.clip(action[0], 0.0, self._max_anfer))
        amir     = float(np.clip(action[1], 0.0, self._max_amir))

        anfer_eff = self._fault_state.apply_anfer(anfer)
        amir_eff  = self._fault_state.apply_amir(amir)

        sos_obs, R_daily, done, sos_info = self._sos_env.step(
            {'anfer': anfer_eff, 'amir': amir_eff}
        )
        raw_obs = self._encode(sos_obs)
        obs19   = self._fault_state.step_corrupt(raw_obs)

        info = dict(sos_info)
        info['preference_w']      = self.current_w.copy()
        info['fault_type']        = self._last_fault_type
        info['action_applied']    = np.array([anfer, amir], dtype=np.float32)
        info['action_effective']  = np.array([anfer_eff, amir_eff], dtype=np.float32)
        return obs19, float(R_daily), bool(done), False, info

    # ──────────────────────────────────────────────────────────────
    def close(self):
        self._sos_env.close()

    # ──────────────────────────────────────────────────────────────
    def _encode(self, sos_obs):
        def _g(k): return float(sos_obs.get(f'crop_{k}', 0.0) or 0.0)
        sw_layers    = sos_obs.get('crop_sw', None)
        sw_mean      = float(np.mean(np.asarray(sw_layers, dtype=np.float32))) \
                       if sw_layers is not None else 0.0
        sensor_vals  = [float(sos_obs.get(f'sensor_{i}', 1.0) or 0.0)
                        for i in range(self._sos_env.n_sensors)]
        sensors_frac = float(np.mean(sensor_vals)) if sensor_vals else 1.0

        crop = np.array([
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
        return np.concatenate([crop, self.current_w])   # 14-dim


# ══════════════════════════════════════════════════════════════════════
# Lightweight per-episode fault state (no gym overhead)
# ══════════════════════════════════════════════════════════════════════
from faulty_env_v2 import SENSOR_FEATURES, _ALIVE, _DROPOUT, _STUCK
from faulty_env_v3 import _OOB

N_SEN = 5


class _FaultStateV3:
    def __init__(self, kwargs):
        self.dropout_rate     = float(kwargs.get('dropout_rate',     0.0))
        self.packet_loss_prob = float(kwargs.get('packet_loss_prob', 0.0))
        self.stuck_sensors    = set(kwargs.get('stuck_sensors',      []))
        self.stuck_day        = int(kwargs.get('stuck_day',          50))
        self.sensor_bias      = dict(kwargs.get('sensor_bias',       {}))
        self.noise_std        = float(kwargs.get('noise_std',        0.0))
        self.N_efficiency     = float(kwargs.get('N_efficiency',     1.0))
        self.W_efficiency     = float(kwargs.get('W_efficiency',     1.0))
        self.sensor_gain      = dict(kwargs.get('sensor_gain',       {}))
        self.spike_prob       = float(kwargs.get('spike_prob',       0.0))
        self.spike_sensors    = set(kwargs.get('spike_sensors',      []))
        self.oob_sensors      = dict(kwargs.get('oob_sensors',       {}))
        self.stuck_anfer      = kwargs.get('actuator_stuck_anfer',   None)
        self.stuck_amir       = kwargs.get('actuator_stuck_amir',    None)

        self._states     = [_ALIVE] * N_SEN
        self._frozen     = np.zeros(11, dtype=np.float32)
        self._last_valid = np.zeros(11, dtype=np.float32)
        self._day        = 0

    def apply_anfer(self, v):
        return float(self.stuck_anfer) if self.stuck_anfer is not None \
               else float(v) * self.N_efficiency

    def apply_amir(self, v):
        return float(self.stuck_amir) if self.stuck_amir is not None \
               else float(v) * self.W_efficiency

    def corrupt(self, obs14):
        self._last_valid = obs14[:11].copy()
        self._frozen     = obs14[:11].copy()
        return self._apply(obs14)

    def step_corrupt(self, obs14):
        self._day += 1
        self._tick(obs14)
        return self._apply(obs14)

    def _tick(self, obs14):
        for i in range(N_SEN):
            if self._states[i] != _ALIVE:
                continue
            for j in SENSOR_FEATURES[i]:
                self._last_valid[j] = obs14[j]
            if i in self.stuck_sensors and self._day >= self.stuck_day:
                self._states[i] = _STUCK
                for j in SENSOR_FEATURES[i]:
                    self._frozen[j] = obs14[j]
            elif np.random.random() < self.dropout_rate:
                self._states[i] = _DROPOUT

    def _apply(self, obs14):
        obs  = obs14.copy()
        mask = np.ones(N_SEN, dtype=np.float32)

        for i in range(N_SEN):
            feats = SENSOR_FEATURES[i]
            state = self._states[i]

            if state == _DROPOUT:
                for j in feats: obs[j] = 0.0
                mask[i] = 0.0
                continue

            if state == _STUCK:
                for j in feats: obs[j] = self._frozen[j]
                continue

            if i in self.oob_sensors:
                lo, hi = _OOB.get(i, (-0.2, 1.3))
                val = hi if self.oob_sensors[i] == 'high' else lo
                for j in feats: obs[j] = float(val)
                continue

            if self.packet_loss_prob > 0.0 and \
               np.random.random() < self.packet_loss_prob:
                for j in feats: obs[j] = 0.0
                mask[i] = 0.0
                continue

            if i in self.sensor_gain:
                g = float(self.sensor_gain[i])
                for j in feats:
                    obs[j] = float(np.clip(obs[j] * g, -2.0, 2.0))

            if i in self.sensor_bias:
                b = float(self.sensor_bias[i])
                for j in feats:
                    obs[j] = float(np.clip(obs[j] + b, -2.0, 2.0))

            if self.noise_std > 0.0:
                for j in feats:
                    obs[j] = float(np.clip(
                        obs[j] + np.random.normal(0.0, self.noise_std), -2.0, 2.0))

            if i in self.spike_sensors and self.spike_prob > 0.0 and \
               np.random.random() < self.spike_prob:
                lo, hi = _OOB.get(i, (-0.2, 1.3))
                sv = hi if np.random.random() > 0.5 else lo
                for j in feats: obs[j] = float(np.clip(sv, -2.0, 2.0))

        obs[10] = float(np.mean(mask))
        return np.concatenate([obs[:11], mask, obs[11:14]])   # 19-dim
