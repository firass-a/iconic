"""
faulty_env_v2.py  —  Realistic fault injection wrapper for CAPQL.

Replaces the naive FaultyEnv, which did not actually map sensors to
features (sensor death had no effect on crop readings).

══════════════════════════════════════════════════════════════════════
Sensor-to-feature mapping  (5 IoT nodes, each owns specific obs dims)
══════════════════════════════════════════════════════════════════════
    sensor_0  →  obs[7]        sw_mean       (soil moisture probe)
    sensor_1  →  obs[0], [4]   topwt, xlai   (canopy / biomass camera)
    sensor_2  →  obs[1], [3]   grnwt, vstage (grain / growth sensor)
    sensor_3  →  obs[5]        cumsumfert    (N-flow meter)
    sensor_4  →  obs[6]        totir         (irrigation meter)

Always available — not owned by any sensor node:
    obs[2]   dap            internal day counter (never corrupted)
    obs[8]   energy_budget  IoT energy level (internal)
    obs[9]   comm_quality   radio signal (always readable)
    obs[10]  sensors_frac   recomputed from mask each step
    obs[11:] w              preference vector (never corrupted)

══════════════════════════════════════════════════════════════════════
Fault taxonomy
══════════════════════════════════════════════════════════════════════
TYPE 1 — Observable failures  →  agent KNOWS data is missing
    dropout      sensor node loses power → no signal
                 obs[j] = 0,  mask[i] = 0   (masking applies)

    packet_loss  sensor packet not received each step with prob p
                 obs[j] = 0,  mask[i] = 0   (masking applies)

TYPE 2 — Hidden failures  →  agent receives a value, cannot tell it is wrong
    stuck        hardware frozen → feature repeats last valid reading
                 mask[i] stays 1  (looks valid, GRU must detect from stale pattern)

    bias         systematic miscalibration → obs[j] += constant
                 mask[i] stays 1

    noise        random measurement error → obs[j] += N(0, σ)
                 mask[i] stays 1

ACTUATOR
    N_efficiency  N spreader delivers η_N × commanded   (η_N ∈ [0, 1])
    W_efficiency  pump delivers η_W × commanded         (η_W ∈ [0, 1])
    The action seen by DSSAT is scaled; cumsumfert/totir in the next
    observation reflect the ACTUAL delivery, giving the GRU a gap
    signal to detect degradation over time.

══════════════════════════════════════════════════════════════════════
Output dimension
══════════════════════════════════════════════════════════════════════
    return_mask=False  →  14-dim  (identical layout to CAPQLEnv v2)
                          used for evaluating the unmodified v2 model

    return_mask=True   →  19-dim  = faulted_obs(11) + mask(5) + w(3)
                          used for CAPQL-Robust training and evaluation
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
N_SENSORS = 5

# Each sensor "owns" the obs indices it reads from DSSAT
SENSOR_FEATURES = {
    0: [7],         # soil moisture probe  →  sw_mean
    1: [0, 4],      # canopy / biomass cam →  topwt, xlai
    2: [1, 3],      # grain / stage sensor →  grnwt, vstage
    3: [5],         # N-flow meter         →  cumsumfert
    4: [6],         # irrigation meter     →  totir
}

# Sensor fault states
_ALIVE   = 0   # healthy, reading is current
_DROPOUT = 1   # Type 1 : node lost power — observable, mask = 0
_STUCK   = 2   # Type 2 : hardware frozen — hidden, mask stays 1


# ──────────────────────────────────────────────────────────────────────
class FaultyEnvV2:
    """
    Wraps any CAPQLEnv and injects realistic faults at every step.

    Parameters
    ----------
    base_env : CAPQLEnv instance
    dropout_rate : float
        Probability per day that an alive sensor permanently loses power.
        Once dead the sensor stays dead (no recovery).
    packet_loss_prob : float
        Per-step probability that a sensor packet is not received.
        Independent per sensor, each step.  Resets next step (transient).
    stuck_sensors : iterable of int
        Sensor indices that will freeze at `stuck_day`.
    stuck_day : int
        Season day (0-indexed) at which stuck sensors freeze.
    sensor_bias : dict {int: float}
        Per-sensor constant offset added to all owned features while alive.
        E.g. {0: 0.15} biases sw_mean high by 0.15 normalised units.
    noise_std : float
        Gaussian noise σ applied each step to all alive sensor features.
    N_efficiency : float  ∈ [0, 1]
        N spreader delivery ratio.  actual_anfer = commanded × η_N.
    W_efficiency : float  ∈ [0, 1]
        Irrigation pump delivery ratio.  actual_amir = commanded × η_W.
    return_mask : bool
        False → 14-dim obs (for CAPQL v2 evaluation).
        True  → 19-dim obs with mask flags (for CAPQL-Robust).
    """

    def __init__(
        self,
        base_env,
        # ── Type 1 ──────────────────────────────────────────────────
        dropout_rate     = 0.0,
        packet_loss_prob = 0.0,
        # ── Type 2 ──────────────────────────────────────────────────
        stuck_sensors    = (),
        stuck_day        = 50,
        sensor_bias      = None,
        noise_std        = 0.0,
        # ── Actuator ────────────────────────────────────────────────
        N_efficiency     = 1.0,
        W_efficiency     = 1.0,
        # ── Output ──────────────────────────────────────────────────
        return_mask      = False,
    ):
        self.env              = base_env
        self.dropout_rate     = float(dropout_rate)
        self.packet_loss_prob = float(packet_loss_prob)
        self.stuck_sensors    = set(stuck_sensors)
        self.stuck_day        = int(stuck_day)
        self.sensor_bias      = dict(sensor_bias or {})
        self.noise_std        = float(noise_std)
        self.N_efficiency     = float(N_efficiency)
        self.W_efficiency     = float(W_efficiency)
        self.return_mask      = return_mask

        # Per-sensor state machine
        self._states     = [_ALIVE] * N_SENSORS
        # Last clean reading per obs feature (updated while sensor is alive)
        self._last_valid = np.zeros(11, dtype=np.float32)
        # Frozen reading per obs feature (captured at the moment sensor gets stuck)
        self._frozen     = np.zeros(11, dtype=np.float32)
        self._day        = 0

        obs_dim = 19 if return_mask else 14
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = base_env.action_space

    # ──────────────────────────────────────────────────────────────
    def reset(self, **kwargs):
        obs, info        = self.env.reset(**kwargs)
        self._day        = 0
        self._states     = [_ALIVE] * N_SENSORS
        self._last_valid = obs[:11].copy()
        self._frozen     = obs[:11].copy()
        return self._apply(obs), info

    # ──────────────────────────────────────────────────────────────
    def step(self, action):
        # ── Actuator faults: scale before DSSAT receives the action ──
        action = np.asarray(action, dtype=np.float32).copy()
        action[0] = action[0] * self.N_efficiency   # N spreader
        action[1] = action[1] * self.W_efficiency   # irrigation pump

        obs, r, done, trunc, info = self.env.step(action)
        self._day += 1
        self._tick(obs)   # advance sensor state machines
        return self._apply(obs), r, done, trunc, info

    # ──────────────────────────────────────────────────────────────
    def _tick(self, clean_obs):
        """Advance fault state machine; update last-valid cache."""
        for i in range(N_SENSORS):
            if self._states[i] != _ALIVE:
                continue

            # Cache the last clean reading for this sensor's features
            for j in SENSOR_FEATURES[i]:
                self._last_valid[j] = clean_obs[j]

            # Transition: alive → stuck (Type 2, hidden)
            if i in self.stuck_sensors and self._day >= self.stuck_day:
                self._states[i] = _STUCK
                for j in SENSOR_FEATURES[i]:
                    self._frozen[j] = clean_obs[j]   # freeze NOW

            # Transition: alive → dropout (Type 1, permanent)
            elif np.random.random() < self.dropout_rate:
                self._states[i] = _DROPOUT

    # ──────────────────────────────────────────────────────────────
    def _apply(self, obs):
        """
        Corrupt obs according to current sensor states.
        Returns 14-dim or 19-dim depending on self.return_mask.
        """
        obs  = obs.copy()
        mask = np.ones(N_SENSORS, dtype=np.float32)   # 1=trusted, 0=unknown

        for i in range(N_SENSORS):
            feats = SENSOR_FEATURES[i]
            state = self._states[i]

            # ── Type 1: dropout — observable, mask = 0 ────────────
            if state == _DROPOUT:
                for j in feats:
                    obs[j] = 0.0
                mask[i] = 0.0
                continue

            # ── Type 2: stuck — hidden, value frozen ──────────────
            if state == _STUCK:
                for j in feats:
                    obs[j] = self._frozen[j]
                # mask[i] stays 1.0 — value looks valid to agent
                continue

            # ── Alive sensor ──────────────────────────────────────

            # Packet loss (Type 1, transient per step)
            if self.packet_loss_prob > 0.0 and \
               np.random.random() < self.packet_loss_prob:
                for j in feats:
                    obs[j] = 0.0
                mask[i] = 0.0
                continue   # skip bias/noise for this sensor this step

            # Systematic bias (Type 2)
            if i in self.sensor_bias:
                b = float(self.sensor_bias[i])
                for j in feats:
                    obs[j] = float(np.clip(obs[j] + b, -2.0, 2.0))

            # Random noise (Type 2)
            if self.noise_std > 0.0:
                for j in feats:
                    obs[j] = float(np.clip(
                        obs[j] + np.random.normal(0.0, self.noise_std),
                        -2.0, 2.0,
                    ))

        # Recompute obs[10] (sensors_frac) to reflect actual sensor health
        obs[10] = float(np.mean(mask))

        if self.return_mask:
            # 19-dim layout: faulted_crop(11) | mask(5) | w(3)
            return np.concatenate([obs[:11], mask, obs[11:14]])
        else:
            return obs   # 14-dim, same layout as CAPQLEnv v2

    # ──────────────────────────────────────────────────────────────
    def close(self):
        self.env.close()

    # ──────────────────────────────────────────────────────────────
    @property
    def sensor_states(self):
        """Read-only view of current sensor fault states (for logging)."""
        labels = {_ALIVE: 'alive', _DROPOUT: 'dropout', _STUCK: 'stuck'}
        return {i: labels[s] for i, s in enumerate(self._states)}
