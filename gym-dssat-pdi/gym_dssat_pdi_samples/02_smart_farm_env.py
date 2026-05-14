"""
SmartFarmSoSEnv — 3-objective MORL wrapper for gym-DSSAT.

Reward vector:  R = [R_yield, R_wue, R_energy]

R_yield  — two-phase daily shaping + fertilizer penalty + terminal bonus
R_wue    — daily water-efficiency signal + terminal WUE bonus (irrigation only)
R_energy — daily controllable-energy penalty + terminal season-energy bonus

Design decisions (fully documented):

    R_yield phase switch:
        Phase A (istage ∈ {8,9,1,2,3}): Δtopwt shaping — biomass proxy for
            yield potential during vegetative and floral growth.
        Phase B (istage ∈ {4,5,6}): Δgrnwt shaping — direct grain signal
            during grain filling, with a 2× stronger coefficient.
        Both phases penalise anfer to discourage excess nitrogen application.
        Note: istage follows DSSAT order 8→9→1→2→3→4→5→6, not 0→1→2...

    R_wue daily signal:
        benefit = clip(Δtopwt / (amir + ε) / BENEFIT_NORM, 0, 1)
            — rewards biomass growth per unit of irrigation applied.
            — rain-fed growth days score high because denominator = ε.
        waste   = clip(amir × sw_mean / WASTE_NORM, 0, 1)
            — penalises irrigation into already-wet soil.
        Both are clipped to [0,1] before scaling to keep R_wue bounded.
        Terminal uses irrigation-only denominator (no rainfall) so the
        signal is fully controllable by the agent.

    R_wue terminal recalibration (rainfall excluded):
        Old baseline: grnwt/(irrig+rain) ≈ 7620/648 ≈ 11.76 → BASELINE=10
        New baseline: grnwt/irrig        ≈ 7620/398 ≈ 19.15 → BASELINE=19
        New max:      efficient policy   ≈ 7000/150 ≈ 46.7  → MAX=40

    R_energy daily signal:
        Only controllable costs: pump (0.3×amir) + fertilizer (0.05×anfer).
        Background sensor load (0.5 kWh/day) excluded from daily penalty
        because the agent cannot reduce it — penalising it adds noise.
        Terminal formula unchanged: includes background load correctly.

    Normalization:
        All per-step terms are clipped to [−1,1] or [0,1] before scaling.
        Per-step ranges: R_yield [−0.25,+0.20], R_wue [−0.10,+0.15], R_energy [−0.05,0]
        Terminal ranges: all clip(·,−1,1).
        No episode-level clipping — PPO value function learns the scale.

    Shared prev_biomass:
        self.prev_biomass is read by BOTH R_yield (Δtopwt) and R_wue (Δtopwt).
        It is updated ONCE at the end of _compute_rewards, after both reads.

Run inside Docker:
    python3 /workspace/02_smart_farm_env.py
"""

import gym
import numpy as np
from collections import OrderedDict


class SmartFarmSoSEnv:
    """
    Wraps gym-DSSAT with System-of-Systems modeling.

    Reward is a VECTOR of shape (3,): [R_yield, R_wue, R_energy].
    The observation merges DSSAT crop state with SoS state (sensor health,
    energy budget, comm quality).
    """

    # ================================================================
    # R_yield calibration constants
    # ================================================================
    BASELINE_YIELD          = 7620.0     # kg/ha — rule-based baseline
    MAX_EXPECTED_YIELD      = 11430.0    # kg/ha — 1.5 × baseline
    # Phase A (vegetative): coefficient on clip(Δtopwt/150, −1, 1)
    YIELD_COEF_PHASE_A      = 0.10
    # Phase B (grain fill): coefficient on clip(Δgrnwt/100, −1, 1)
    YIELD_COEF_PHASE_B      = 0.20
    # Fertilizer penalty: coefficient on clip(anfer/100, 0, 1)
    FERT_PENALTY_COEF       = 0.05
    # Normalisers for the per-step shaping clips
    TOPWT_DAILY_NORM        = 150.0     # typical max daily biomass gain kg/ha
    GRNWT_DAILY_NORM        = 100.0     # typical max daily grain gain kg/ha
    ANFER_NORM              = 100.0     # generous single application kg N/ha
    # DSSAT istage values that correspond to grain filling or later
    PHASE_B_STAGES          = {4, 5, 6}

    # ================================================================
    # R_wue calibration constants
    # ================================================================
    # Daily benefit: clip(Δtopwt / (amir + ε) / BENEFIT_NORM, 0, 1)
    WUE_EPSILON             = 0.1       # mm — prevents division by zero
    WUE_BENEFIT_NORM        = 2000.0    # max Δtopwt(200) / min denom(0.1)
    WUE_ALPHA               = 0.15      # benefit scaling coefficient
    # Daily waste: clip(amir × sw_mean / WASTE_NORM, 0, 1)
    WUE_WASTE_NORM          = 50.0      # max amir (action upper bound)
    WUE_BETA                = 0.10      # waste penalty coefficient
    # Terminal WUE — irrigation only, no rainfall
    # Recalibrated: baseline 7620/398=19.15, max ~40 kg/mm
    BASELINE_WUE_KG_PER_MM  = 19.0
    MAX_WUE_KG_PER_MM       = 40.0

    # ================================================================
    # R_energy calibration constants
    # ================================================================
    SYSTEM_BASELINE_KWH_DAY  = 0.5      # background sensor load kWh/day
    SEASON_DAYS              = 161
    # baseline_energy = 398*0.3 + 150*0.05 + 161*0.5 = 207 kWh
    BASELINE_SEASON_ENERGY   = 207.0
    # Daily: max controllable cost = 0.3*50 + 0.05*200 = 25 kWh
    DAILY_ENERGY_NORM        = 25.0
    ENERGY_DAILY_COEF        = 0.05     # daily penalty coefficient

    # ================================================================
    # SoS background constants
    # ================================================================
    PER_STEP_YIELD_COEF      = 0.1      # kept for backward compatibility reference

    # ================================================================
    # Construction
    # ================================================================
    def __init__(self, mode='fertilization', seed=None,
                 enable_faults=False, fault_rate=0.02,
                 n_sensors=5, initial_energy=100.0):
        env_args = {'mode': mode}
        if seed is not None:
            env_args['seed'] = seed
        self.env = gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args)

        self.n_sensors      = n_sensors
        self.initial_energy = initial_energy
        self.enable_faults  = enable_faults
        self.fault_rate     = fault_rate

        # Episode state — filled by reset()
        self.sensor_health  = None
        self.energy_budget  = None
        self.comm_quality   = None
        self.prev_biomass   = None   # topwt at previous step (shared R1+R2)
        self.prev_grnwt     = None   # grnwt at previous step (R1 Phase B)
        self.total_water    = None   # cumulative irrigation mm (R2 terminal)
        self.total_rain     = None   # cumulative rainfall mm (obs only)
        self.total_nitrogen = None   # cumulative N kg/ha (R3 terminal)
        self.day            = None
        self.done           = False
        self._last_crop_obs = {}

    # ================================================================
    # Reset
    # ================================================================
    def reset(self):
        """Reset env and all SoS state."""
        obs = self.env.reset()
        if obs is None:
            obs = {}
        self._last_crop_obs = obs

        self.sensor_health  = np.ones(self.n_sensors)
        self.energy_budget  = self.initial_energy
        self.comm_quality   = 1.0
        self.prev_biomass   = float(obs.get('topwt', 0.0) or 0.0)
        self.prev_grnwt     = float(obs.get('grnwt', 0.0) or 0.0)  # NEW
        self.total_water    = 0.0
        self.total_rain     = 0.0
        self.total_nitrogen = 0.0
        self.day            = 0
        self.done           = False

        return self._build_observation(obs)

    # ================================================================
    # Step
    # ================================================================
    def step(self, action_dict):
        """One day step.

        Args:
            action_dict: {'anfer': float kg N/ha, 'amir': float mm/day}

        Returns:
            merged_obs   : OrderedDict
            reward_vector: np.ndarray shape (3,) — [R_yield, R_wue, R_energy]
            done         : bool
            info         : dict
        """
        # 1. Underlying simulator step
        crop_obs, default_reward, done, info = self.env.step(action_dict)
        if info is None:
            info = {}

        # gym-DSSAT returns None obs on the terminal step; fall back to the
        # last valid obs so the terminal grain weight can be read.
        if crop_obs is None or len(crop_obs) == 0:
            crop_obs            = {}
            crop_obs_for_reward = self._last_crop_obs
        else:
            self._last_crop_obs = crop_obs
            crop_obs_for_reward = crop_obs

        self.done  = done
        self.day  += 1

        # 2. Track resource usage
        nitrogen_applied = float(action_dict.get('anfer', 0.0))
        water_applied    = float(action_dict.get('amir',  0.0))
        rain_today       = float(crop_obs.get('rain', 0.0) or 0.0)

        self.total_nitrogen += nitrogen_applied
        self.total_water    += water_applied
        self.total_rain     += max(rain_today, 0.0)

        # 3. Energy bookkeeping (SoS observation only — not driving R_energy)
        sensor_drain   = float(np.sum(self.sensor_health)) * 0.1
        pump_cost      = water_applied    * 0.3
        fert_cost      = nitrogen_applied * 0.05
        daily_cost     = sensor_drain + pump_cost + fert_cost
        solar_recharge = 2.0
        self.energy_budget = max(0.0, min(100.0,
            self.energy_budget - daily_cost + solar_recharge))

        # 4. Fault injection (perturbs SoS observation, not reward)
        if self.enable_faults:
            self._inject_faults(crop_obs)

        # 5. Compute mean soil water for R_wue waste term
        sw_layers = crop_obs_for_reward.get('sw', None)
        if sw_layers is not None:
            sw_mean = float(np.mean(np.asarray(sw_layers, dtype=np.float32)))
        else:
            sw_mean = 0.0

        # 6. Reward vector
        reward_vector = self._compute_rewards(
            crop_obs   = crop_obs_for_reward,
            done       = done,
            anfer      = nitrogen_applied,
            amir       = water_applied,
            sw_mean    = sw_mean,
        )

        # 7. Merged observation
        merged_obs = self._build_observation(crop_obs)

        # 8. Info dict
        info['reward_components'] = {
            'R_yield':  float(reward_vector[0]),
            'R_wue':    float(reward_vector[1]),
            'R_energy': float(reward_vector[2]),
        }
        info['sos_state'] = {
            'sensor_health':  self.sensor_health.copy(),
            'energy_budget':  self.energy_budget,
            'comm_quality':   self.comm_quality,
            'total_nitrogen': self.total_nitrogen,
            'total_water':    self.total_water,
            'total_rain':     self.total_rain,
            'grnwt': float(crop_obs_for_reward.get('grnwt', 0.0) or 0.0),
        }
        info['default_scalar_reward'] = default_reward

        return merged_obs, reward_vector, done, info

    # ================================================================
    # Reward computation
    # ================================================================
    def _compute_rewards(self, crop_obs, done, anfer, amir, sw_mean):
        """
        Three-component reward vector.

        Args:
            crop_obs : raw DSSAT observation dict (keys WITHOUT 'crop_' prefix)
            done     : bool — True on harvest/terminal step
            anfer    : float — kg N/ha applied this step
            amir     : float — mm irrigation applied this step
            sw_mean  : float — mean volumetric soil water content [0,1]

        Returns:
            np.ndarray shape (3,) — [R_yield, R_wue, R_energy]

        IMPORTANT — prev_biomass / prev_grnwt update order:
            Both R_yield (Phase A) and R_wue read self.prev_biomass.
            We read it FIRST, compute deltas, then update ONCE at the bottom.
            Never update mid-function.
        """
        # ── read current crop state ──────────────────────────────────────
        current_topwt = float(crop_obs.get('topwt', 0.0) or 0.0)
        current_grnwt = float(crop_obs.get('grnwt', 0.0) or 0.0)
        istage        = int(crop_obs.get('istage', 0) or 0)

        # ── compute deltas (read prev BEFORE any update) ─────────────────
        delta_topwt = current_topwt - self.prev_biomass   # used by R1-A and R2
        delta_grnwt = current_grnwt - self.prev_grnwt     # used by R1-B

        # ensure deltas are non-negative clipped for shaping
        # (negative biomass change can happen due to senescence — we allow it)

        # ================================================================
        # R_yield
        # ================================================================
        if istage in self.PHASE_B_STAGES:
            # Phase B — grain filling signal (stronger coefficient)
            shaping = self.YIELD_COEF_PHASE_B * float(
                np.clip(delta_grnwt / self.GRNWT_DAILY_NORM, -1.0, 1.0)
            )
        else:
            # Phase A — biomass proxy (vegetative + floral + pre-emergence)
            shaping = self.YIELD_COEF_PHASE_A * float(
                np.clip(delta_topwt / self.TOPWT_DAILY_NORM, -1.0, 1.0)
            )

        fert_penalty = self.FERT_PENALTY_COEF * float(
            np.clip(anfer / self.ANFER_NORM, 0.0, 1.0)
        )

        R_yield = shaping - fert_penalty

        if done:
            yield_norm = (current_grnwt - self.BASELINE_YIELD) / (
                self.MAX_EXPECTED_YIELD - self.BASELINE_YIELD
            )
            R_yield += float(np.clip(yield_norm, -1.0, 1.0))

        # ================================================================
        # R_wue
        # ================================================================
        # Daily benefit: how much biomass was gained per mm of water applied
        benefit_raw = delta_topwt / (amir + self.WUE_EPSILON)
        benefit     = float(np.clip(
            benefit_raw / self.WUE_BENEFIT_NORM, 0.0, 1.0
        ))

        # Daily waste: irrigating into already-wet soil
        waste = float(np.clip(
            amir * sw_mean / self.WUE_WASTE_NORM, 0.0, 1.0
        ))

        R_wue = self.WUE_ALPHA * benefit - self.WUE_BETA * waste

        if done:
            # Terminal WUE — irrigation only, no rainfall
            wue_kg_per_mm = current_grnwt / (self.total_water + self.WUE_EPSILON)
            wue_norm      = (
                (wue_kg_per_mm - self.BASELINE_WUE_KG_PER_MM)
                / (self.MAX_WUE_KG_PER_MM - self.BASELINE_WUE_KG_PER_MM)
            )
            R_wue += float(np.clip(wue_norm, -1.0, 1.0))

        # ================================================================
        # R_energy
        # ================================================================
        # Daily: controllable costs only (pump + fertilizer production)
        # Background sensor load excluded — agent cannot control it
        daily_controllable_energy = 0.3 * amir + 0.05 * anfer
        R_energy = -self.ENERGY_DAILY_COEF * float(
            np.clip(daily_controllable_energy / self.DAILY_ENERGY_NORM, 0.0, 1.0)
        )

        if done:
            # Terminal: full season energy including background (unchanged)
            season_energy = (
                self.total_water    * 0.3
                + self.total_nitrogen * 0.05
                + self.SEASON_DAYS  * self.SYSTEM_BASELINE_KWH_DAY
            )
            energy_ratio = season_energy / self.BASELINE_SEASON_ENERGY
            R_energy += float(np.clip(1.0 - energy_ratio, -1.0, 1.0))

        # ================================================================
        # Update shared prev values — ONCE, at the end
        # ================================================================
        self.prev_biomass = current_topwt
        self.prev_grnwt   = current_grnwt

        return np.array([R_yield, R_wue, R_energy], dtype=np.float32)

    # ================================================================
    # Fault injection
    # ================================================================
    def _inject_faults(self, crop_obs):
        """Stochastic sensor dropouts and comm-quality fluctuations.

        Perturbs the SoS observation without modifying reward computation.
        R_resilience was removed — these faults are observable but not
        rewarded, enabling future fault-aware analysis.
        """
        for i in range(self.n_sensors):
            if (self.sensor_health[i] == 1.0
                    and np.random.random() < self.fault_rate):
                self.sensor_health[i] = 0.0
        for i in range(self.n_sensors):
            if (self.sensor_health[i] == 0.0
                    and np.random.random() < 0.01):
                self.sensor_health[i] = 1.0

        comm_noise = np.random.normal(0, 0.05)
        self.comm_quality = float(
            np.clip(self.comm_quality + comm_noise, 0.3, 1.0)
        )
        if np.random.random() < 0.005:
            self.comm_quality = 0.3

    # ================================================================
    # Observation
    # ================================================================
    def _build_observation(self, crop_obs):
        """Merge raw DSSAT crop state and SoS state into one OrderedDict."""
        merged = OrderedDict()
        for key, value in crop_obs.items():
            merged[f'crop_{key}'] = value
        for i in range(self.n_sensors):
            merged[f'sensor_{i}'] = self.sensor_health[i]
        merged['energy_budget'] = self.energy_budget / 100.0
        merged['comm_quality']  = self.comm_quality
        return merged

    # ================================================================
    # Scalarization helpers
    # ================================================================
    def scalarize_reward(self, reward_vector, weights=None):
        """Linear scalarization with default yield-dominant weights."""
        if weights is None:
            weights = np.array([0.55, 0.30, 0.15])
        return float(np.dot(weights, reward_vector))

    def chebyshev_scalarize(self, reward_vector, weights=None, ideal=None):
        """Chebyshev scalarization — can recover non-convex Pareto solutions."""
        if weights is None:
            weights = np.array([0.55, 0.30, 0.15])
        if ideal is None:
            ideal = np.array([1.0, 1.0, 1.0])
        weighted_dist = weights * np.abs(reward_vector - ideal)
        return -float(np.max(weighted_dist))

    def close(self):
        self.env.close()


# ================================================================
# Demo / smoke test
# ================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("SmartFarmSoSEnv — revised 3-objective rewards — DEMO")
    print("=" * 70)

    env = SmartFarmSoSEnv(mode='all', seed=42, enable_faults=False)
    obs = env.reset()
    print(f"\nInitial obs keys : {list(obs.keys())[:6]}... ({len(obs)} total)")
    print(f"prev_biomass     : {env.prev_biomass:.1f} kg/ha")
    print(f"prev_grnwt       : {env.prev_grnwt:.1f} kg/ha")

    cumulative  = np.zeros(3)
    step_log    = []

    for day in range(1, 162):
        # Rule-based policy: 40 kg N on days 35/65/95, 10 mm every 3 days
        anfer = 40.0 if day in (35, 65, 95) else 0.0
        amir  = 10.0 if day % 3 == 0 else 0.0

        obs, R, done, info = env.step({'anfer': anfer, 'amir': amir})
        cumulative += R

        # Log a few representative days
        if day in (1, 35, 65, 95, 110) or done:
            rc = info['reward_components']
            step_log.append({
                'day':      day,
                'anfer':    anfer,
                'amir':     amir,
                'R_yield':  rc['R_yield'],
                'R_wue':    rc['R_wue'],
                'R_energy': rc['R_energy'],
            })

        if done:
            sos = info['sos_state']
            print(f"\nEpisode end — day {day}")
            print(f"  yield            = {sos['grnwt']:.0f} kg/ha")
            print(f"  total irrigation = {sos['total_water']:.0f} mm")
            print(f"  total nitrogen   = {sos['total_nitrogen']:.0f} kg/ha")
            print(f"  total rainfall   = {sos['total_rain']:.0f} mm")
            print()
            print(f"  cumulative R_yield  = {cumulative[0]:+.3f}")
            print(f"  cumulative R_wue    = {cumulative[1]:+.3f}")
            print(f"  cumulative R_energy = {cumulative[2]:+.3f}")
            print()
            print(f"  scalar (linear)  = {env.scalarize_reward(R):+.4f}  (terminal step only)")
            print(f"  scalar (cheby)   = {env.chebyshev_scalarize(R):+.4f}")
            break

    print("\nStep-level reward log:")
    print(f"  {'day':>4}  {'anfer':>6}  {'amir':>5}  "
          f"{'R_yield':>9}  {'R_wue':>8}  {'R_energy':>9}")
    print("  " + "-" * 56)
    for row in step_log:
        print(f"  {row['day']:>4}  {row['anfer']:>6.1f}  {row['amir']:>5.1f}  "
              f"  {row['R_yield']:>+7.4f}  {row['R_wue']:>+7.4f}  {row['R_energy']:>+8.4f}")

    env.close()
    print("\nNext: update pc_env.py preference dims if needed, then re-run PC-PPO.")
