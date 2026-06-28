"""
faulty_env_v3.py  —  Extended fault injection (extends faulty_env_v2.py).

New faults added
────────────────
  Sensor (Type 2, mask stays 1 — hidden from agent):
    gain          multiplicative scaling:  obs[j] *= g   (episode-level)
    spike         transient extreme value, prob p_spike  (per-step)
    out_of_bounds sensor forced outside physical range   (episode-level)

  Actuator:
    stuck_at      actuator jammed at fixed value, ignores commands
                  (overrides N_efficiency / W_efficiency)

All faults from faulty_env_v2 are inherited unchanged.
"""

import numpy as np
from faulty_env_v2 import (
    N_SENSORS, SENSOR_FEATURES, _ALIVE, _DROPOUT, _STUCK, FaultyEnvV2
)

# Physical out-of-bounds values per sensor (normalised units)
# Each sensor: (low_oob, high_oob) — values outside the plausible physical range
_OOB = {
    0: (-0.25, 1.35),   # moisture_ratio  physical [0, 1]
    1: (-0.15, 1.25),   # topwt / xlai  physical [0, ~1]
    2: (-0.15, 1.20),   # grnwt/vstage  physical [0, ~1]
    3: (-0.10, 1.30),   # cumsumfert    physical [0, ~1]
    4: (-0.10, 1.30),   # totir         physical [0, ~1]
}


class FaultyEnvV3(FaultyEnvV2):
    """
    Extends FaultyEnvV2 with gain, spike, out_of_bounds, and actuator stuck_at.

    Extra parameters
    ----------------
    sensor_gain : dict {sensor_idx: float}
        Multiplicative gain per sensor.  g > 1 → reads too high; g < 1 → too low.
        E.g. {0: 1.40} → sw_mean reads 40 % above reality all season.
    spike_prob : float
        Per-step probability of a spike on any sensor in spike_sensors.
    spike_sensors : iterable of int
        Sensors that can spike.
    oob_sensors : dict {sensor_idx: str}
        Sensors forced permanently out-of-bounds.  Value: 'high' or 'low'.
    actuator_stuck_anfer : float | None
        N spreader jammed at this constant (kg N/ha).  Ignores all commands.
    actuator_stuck_amir : float | None
        Pump jammed at this constant (mm).  Ignores all commands.
    """

    def __init__(
        self,
        base_env,
        # ── inherited from v2 ───────────────────────────────────────
        dropout_rate=0.0,
        packet_loss_prob=0.0,
        stuck_sensors=(),
        stuck_day=50,
        sensor_bias=None,
        noise_std=0.0,
        N_efficiency=1.0,
        W_efficiency=1.0,
        return_mask=False,
        # ── new in v3 ───────────────────────────────────────────────
        sensor_gain=None,
        spike_prob=0.0,
        spike_sensors=(),
        oob_sensors=None,
        actuator_stuck_anfer=None,
        actuator_stuck_amir=None,
    ):
        super().__init__(
            base_env,
            dropout_rate=dropout_rate,
            packet_loss_prob=packet_loss_prob,
            stuck_sensors=stuck_sensors,
            stuck_day=stuck_day,
            sensor_bias=sensor_bias,
            noise_std=noise_std,
            N_efficiency=N_efficiency,
            W_efficiency=W_efficiency,
            return_mask=return_mask,
        )
        self.sensor_gain          = dict(sensor_gain or {})
        self.spike_prob           = float(spike_prob)
        self.spike_sensors        = set(spike_sensors)
        self.oob_sensors          = dict(oob_sensors or {})   # {idx: 'high'|'low'}
        self.actuator_stuck_anfer = actuator_stuck_anfer
        self.actuator_stuck_amir  = actuator_stuck_amir

    # ──────────────────────────────────────────────────────────────────
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).copy()

        # Actuator stuck_at overrides efficiency scaling
        if self.actuator_stuck_anfer is not None:
            action[0] = float(self.actuator_stuck_anfer)
        else:
            action[0] = action[0] * self.N_efficiency

        if self.actuator_stuck_amir is not None:
            action[1] = float(self.actuator_stuck_amir)
        else:
            action[1] = action[1] * self.W_efficiency

        obs, r, done, trunc, info = self.env.step(action)
        self._day += 1
        self._tick(obs)
        return self._apply(obs), r, done, trunc, info

    # ──────────────────────────────────────────────────────────────────
    def _apply(self, obs):
        obs  = obs.copy()
        mask = np.ones(N_SENSORS, dtype=np.float32)

        for i in range(N_SENSORS):
            feats = SENSOR_FEATURES[i]
            state = self._states[i]

            # ── Type 1: dropout — observable ──────────────────────────
            if state == _DROPOUT:
                for j in feats:
                    obs[j] = 0.0
                mask[i] = 0.0
                continue

            # ── Type 2: stuck frozen ──────────────────────────────────
            if state == _STUCK:
                for j in feats:
                    obs[j] = self._frozen[j]
                continue   # mask stays 1

            # ── Out-of-bounds (permanent, episode-level) ──────────────
            if i in self.oob_sensors:
                lo, hi = _OOB.get(i, (-0.2, 1.3))
                val = hi if self.oob_sensors[i] == 'high' else lo
                for j in feats:
                    obs[j] = float(val)
                continue   # mask stays 1

            # ── Packet loss (transient) ───────────────────────────────
            if self.packet_loss_prob > 0.0 and \
               np.random.random() < self.packet_loss_prob:
                for j in feats:
                    obs[j] = 0.0
                mask[i] = 0.0
                continue

            # ── Gain (multiplicative) ─────────────────────────────────
            if i in self.sensor_gain:
                g = float(self.sensor_gain[i])
                for j in feats:
                    obs[j] = float(np.clip(obs[j] * g, -2.0, 2.0))

            # ── Bias (additive) ───────────────────────────────────────
            if i in self.sensor_bias:
                b = float(self.sensor_bias[i])
                for j in feats:
                    obs[j] = float(np.clip(obs[j] + b, -2.0, 2.0))

            # ── Noise ─────────────────────────────────────────────────
            if self.noise_std > 0.0:
                for j in feats:
                    obs[j] = float(np.clip(
                        obs[j] + np.random.normal(0.0, self.noise_std),
                        -2.0, 2.0,
                    ))

            # ── Spike (transient, after other Type-2 distortions) ─────
            if i in self.spike_sensors and self.spike_prob > 0.0 and \
               np.random.random() < self.spike_prob:
                lo, hi = _OOB.get(i, (-0.2, 1.3))
                spike_val = hi if np.random.random() > 0.5 else lo
                for j in feats:
                    obs[j] = float(np.clip(spike_val, -2.0, 2.0))

        if self.return_mask:
            return np.concatenate([obs[:11], mask, obs[11:14]])
        else:
            return obs

    # ──────────────────────────────────────────────────────────────────
    @property
    def active_fault_summary(self):
        """Human-readable summary of active faults for logging."""
        parts = []
        if self.sensor_gain:
            parts.append(f"gain={self.sensor_gain}")
        if self.spike_prob > 0:
            parts.append(f"spike_p={self.spike_prob:.2f} on {self.spike_sensors}")
        if self.oob_sensors:
            parts.append(f"oob={self.oob_sensors}")
        if self.actuator_stuck_anfer is not None:
            parts.append(f"stuck_anfer={self.actuator_stuck_anfer:.1f}")
        if self.actuator_stuck_amir is not None:
            parts.append(f"stuck_amir={self.actuator_stuck_amir:.1f}")
        # inherited
        if self.dropout_rate > 0:
            parts.append(f"dropout_rate={self.dropout_rate:.2f}")
        if self.packet_loss_prob > 0:
            parts.append(f"packet_p={self.packet_loss_prob:.2f}")
        if self.stuck_sensors:
            parts.append(f"stuck_sensors={self.stuck_sensors}@day{self.stuck_day}")
        if self.sensor_bias:
            parts.append(f"bias={self.sensor_bias}")
        if self.noise_std > 0:
            parts.append(f"noise_std={self.noise_std:.3f}")
        if self.N_efficiency < 1.0:
            parts.append(f"eta_N={self.N_efficiency:.2f}")
        if self.W_efficiency < 1.0:
            parts.append(f"eta_W={self.W_efficiency:.2f}")
        return "; ".join(parts) if parts else "clean"
