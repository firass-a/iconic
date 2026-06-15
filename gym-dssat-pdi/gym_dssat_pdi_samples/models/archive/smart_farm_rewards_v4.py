"""
Reward functions for the Smart Farm MORL problem.

Design (see thesis spec):
  - Daily reward  = shaping (water, N, resource use, environmental losses)
  - Seasonal reward (harvest only) = yield, harvest index, ANE, milestones
  - Terminal step: R_T = R_daily + R_season

The 3-objective vector for PC-PPO:
  R[0] yield      — biomass growth + stress-responsive inputs + harvest grnwt
  R[1] water      — moisture band − irrigation − season water total
  R[2] fertilizer — N uptake − fertilizer − leaching / denitrification − season N total
"""
import numpy as np

# Moisture band (fraction of field capacity SW/DUL)
MOISTURE_OPT_LOW = 0.45
MOISTURE_OPT_HIGH = 0.75
MOISTURE_GOOD = 0.60         # above this, no dryness penalty / no input bonus
MOISTURE_DRY_SPAN = 0.30     # how far below MOISTURE_GOOD before max penalty

# Daily penalty scales — water/N cost less than the corresponding action bonuses
# v4: K_IRRIG bumped to 0.012 so per-mm cost exceeds avoided dryness penalty
#     (water corner was over-irrigating: -0.30 dryness >> -0.004 per mm).
K_IRRIG = 0.012       # per mm/day — was 0.004 (too cheap; agent over-irrigated)
K_FERT = 0.008        # per kg N/ha/day
K_RUNOFF = 0.03       # per mm/day
K_LEACH = 0.025       # per kg N/ha/day leached
K_DENIT = 0.018       # per kg N/ha/day denitrified

# Season totals — present but not dominant over yield harvest term
K_SEASON_WATER = 0.005   # v4: bump from 0.003 to reinforce water-saving signal
K_SEASON_FERT = 0.006

# Yield normalization (kg/ha) — align with strong baseline yields (~10k+)
YIELD_NORM = 10000.0

# Daily grain-accrual bonus (yield channel) — dense yield signal so the agent
# sees grain growth *as it happens*, not just at harvest.
GRAIN_DELTA_SCALE = 100.0    # 1 reward point per +100 kg/ha grain gain
GRAIN_DELTA_WEIGHT = 0.50    # multiplier in daily_yield_component

# One-time daily bonuses when grnwt crosses upward through these bands (kg/ha)
GRNWT_MILESTONES = (500.0, 1500.0, 3000.0, 5000.0, 7000.0, 9000.0)
MILESTONE_BONUS = 0.50   # was 0.35

# Harvest-tier bonuses at season end (kg/ha) — scaled up so end-of-season
# payoff dominates per-day shaping even after gamma^170 discounting.
HARVEST_TIERS = (
    (1500.0, 0.5),
    (3000.0, 1.2),
    (5000.0, 2.5),
    (7000.0, 4.0),
    (9000.0, 6.0),
    (11000.0, 8.0),
)


def moisture_ratio(sw, dul, ll=None):
    """
    MoistureRatio_t = SW_t / DUL_t  (mean over soil layers).

    If *ll* (lower limit) is given, use plant-available fraction instead:
        (SW - LL) / (DUL - LL)
    """
    sw = np.asarray(sw, dtype=np.float64)
    dul = np.asarray(dul, dtype=np.float64)
    if sw.size == 0 or dul.size == 0:
        return 0.0
    n = min(sw.size, dul.size)
    sw, dul = sw[:n], dul[:n]
    if ll is not None:
        ll = np.asarray(ll, dtype=np.float64)[:n]
        avail = np.maximum(dul - ll, 1e-6)
        return float(np.mean(np.clip((sw - ll) / avail, 0.0, 1.0)))
    return float(np.mean(sw / np.maximum(dul, 1e-6)))


def _band_reward(value, low, high, margin=0.15):
    """Reward for staying inside [low, high]; linear penalty outside."""
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(-1.0, 1.0 - (low - value) / margin)
    return max(-1.0, 1.0 - (value - high) / margin)


def grnwt_milestone_delta(prev_grnwt, grnwt):
    """One-time yield bonus for each grnwt band crossed since the previous step."""
    bonus = 0.0
    for threshold in GRNWT_MILESTONES:
        if prev_grnwt < threshold <= grnwt:
            bonus += MILESTONE_BONUS
    return bonus


def harvest_tier_bonus(grnwt):
    """Terminal yield bonus for final harvest level."""
    bonus = 0.0
    for threshold, weight in HARVEST_TIERS:
        if grnwt >= threshold:
            bonus = weight
    return bonus


def daily_yield_component(nstres, trnu, grnwt, topwt, swfac, anfer, amir,
                          moisture=1.0, grnwt_delta=0.0):
    """Yield objective: crop health + daily grain accrual + stress-responsive input bonus.

    Key dense signal: `grnwt_delta` rewards each kg/ha of grain gained today, so the
    agent sees yield growth immediately rather than only via the discounted harvest tier.
    Stress coupling fires whenever soil moisture drops below MOISTURE_GOOD OR swfac
    indicates plant water stress — catches dryness earlier than swfac alone.
    """
    n_bonus = float(np.clip(nstres, 0.0, 1.0))
    uptake = float(np.clip(trnu / 5.0, 0.0, 1.0))
    grain = float(np.clip(grnwt / YIELD_NORM, 0.0, 1.5))
    biomass = float(np.clip(topwt / (YIELD_NORM * 1.5), 0.0, 1.0))

    # Dense daily grain reward — clipped to avoid spikes from numerical noise.
    grain_delta = float(np.clip(max(grnwt_delta, 0.0) / GRAIN_DELTA_SCALE, 0.0, 2.0))

    # Stress = either soil is drying OR plant water-stress is already showing.
    moisture_stress = float(np.clip((MOISTURE_GOOD - moisture) / MOISTURE_DRY_SPAN, 0.0, 1.0))
    plant_stress = float(np.clip(1.0 - swfac, 0.0, 1.0))
    stress = max(moisture_stress, plant_stress)
    input_use = float(np.clip(anfer / 50.0, 0.0, 1.0) + np.clip(amir / 25.0, 0.0, 1.0))
    stress_input_bonus = 0.30 * stress * input_use

    return (
        0.10 * n_bonus + 0.10 * uptake + 0.20 * grain + 0.10 * biomass
        + GRAIN_DELTA_WEIGHT * grain_delta
        + stress_input_bonus
    )


def daily_water_component(moisture, amir, runoff):
    """
    Water objective: penalize dryness + minimize irrigation + penalize runoff.

    Asymmetric moisture shaping: doing nothing while rainfall keeps the field wet
    earns *zero* reward (not +0.45 as before). Negative reward only when soil dries
    below MOISTURE_GOOD — forcing the agent to act in dry conditions instead of
    coasting on luck.
    """
    if moisture >= MOISTURE_GOOD:
        r_moisture = 0.0
    else:
        r_moisture = -float(np.clip((MOISTURE_GOOD - moisture) / MOISTURE_DRY_SPAN, 0.0, 1.0))
    r_resource = -K_IRRIG * float(amir)
    r_loss = -K_RUNOFF * float(max(runoff, 0.0))
    return 0.30 * r_moisture + r_resource + r_loss


def daily_fert_component(nstres, trnu, anfer, tleachd, cnox_daily):
    """Fertilizer objective: N health + minimize applications and losses."""
    r_n = float(np.clip(nstres, 0.0, 1.0))
    r_uptake = float(np.clip(trnu / 5.0, 0.0, 1.0))
    r_resource = -K_FERT * float(anfer)
    r_loss = -K_LEACH * float(max(tleachd, 0.0)) - K_DENIT * float(max(cnox_daily, 0.0))
    return 0.30 * r_n + 0.30 * r_uptake + r_resource + r_loss


def harvest_index(grnwt, topwt):
    if topwt <= 1e-6:
        return 0.0
    return float(np.clip(grnwt / topwt, 0.0, 1.0))


def agronomic_n_efficiency(grnwt, total_n):
    """ANE ≈ kg grain per kg N applied (higher is better)."""
    if total_n <= 1e-6:
        return 0.0
    return float(grnwt / total_n)


def seasonal_yield_component(grnwt, topwt, total_n):
    """Dominant harvest signal — grnwt drives yield-priority behavior."""
    yield_term = np.clip(grnwt / YIELD_NORM, 0.0, 1.5)
    hi = harvest_index(grnwt, topwt)
    ane = np.clip(agronomic_n_efficiency(grnwt, total_n) / 50.0, 0.0, 1.0)
    tiers = harvest_tier_bonus(grnwt)
    return float(2.5 * yield_term + 0.30 * hi + 0.20 * ane + tiers)


def seasonal_water_component(total_water):
    return float(-K_SEASON_WATER * total_water)


def seasonal_fert_component(total_n, grnwt):
    penalty = -K_SEASON_FERT * total_n
    ane_bonus = np.clip(agronomic_n_efficiency(grnwt, total_n) / 80.0, 0.0, 0.4)
    return float(penalty + ane_bonus)


def compute_reward_vector(state, context, action, totals, prev_cnox, done, prev_grnwt=0.0):
    """
    Build the 3-dim reward vector for one day.

    Args:
        state: full DSSAT state dict (post-processed)
        context: static soil context (dul, ll, sat, dlayr)
        action: {'anfer', 'amir'}
        totals: {'water', 'nitrogen'}
        prev_cnox: cumulative cnox at previous step (for daily delta)
        done: harvest flag
        prev_grnwt: grain weight at previous step (for milestone crossings)
    """
    sw = state.get('sw')
    dul = context.get('dul') if context else None
    ll = context.get('ll') if context else None
    moisture = moisture_ratio(sw, dul, ll) if sw is not None and dul is not None else 0.5

    swfac = float(state.get('swfac', 0.0) or 0.0)
    nstres = float(state.get('nstres', 0.0) or 0.0)
    trnu = float(state.get('trnu', 0.0) or 0.0)
    runoff = float(state.get('runoff', 0.0) or 0.0)
    tleachd = float(state.get('tleachd', 0.0) or 0.0)
    cnox = float(state.get('cnox', 0.0) or 0.0)
    cnox_daily = max(cnox - prev_cnox, 0.0)

    amir = float(action.get('amir', 0.0) or 0.0)
    anfer = float(action.get('anfer', 0.0) or 0.0)

    grnwt = float(state.get('grnwt', 0.0) or 0.0)
    topwt = float(state.get('topwt', 0.0) or 0.0)
    grnwt_delta = max(grnwt - prev_grnwt, 0.0)

    r_yield = daily_yield_component(
        nstres, trnu, grnwt, topwt, swfac, anfer, amir,
        moisture=moisture, grnwt_delta=grnwt_delta,
    )
    r_yield += grnwt_milestone_delta(prev_grnwt, grnwt)
    r_water = daily_water_component(moisture, amir, runoff)
    r_fert = daily_fert_component(nstres, trnu, anfer, tleachd, cnox_daily)

    if done:
        r_yield += seasonal_yield_component(grnwt, topwt, totals['nitrogen'])
        r_water += seasonal_water_component(totals['water'])
        r_fert += seasonal_fert_component(totals['nitrogen'], grnwt)

    return np.array([r_yield, r_water, r_fert], dtype=np.float32), moisture
