"""
Reward functions for the Smart Farm MORL problem.

Design (see thesis spec):
  - Daily reward  = shaping (water, N, resource use, environmental losses)
  - Seasonal reward (harvest only) = yield, harvest index, N efficiency, totals
  - Terminal step: R_T = R_daily + R_season

The 3-objective vector for PC-PPO:
  R[0] yield      — agronomic health + terminal yield / HI / ANE
  R[1] water      — moisture management − irrigation − runoff losses
  R[2] fertilizer — N management − fertilizer − leaching / denitrification
"""
import numpy as np

# Moisture band (fraction of field capacity SW/DUL)
MOISTURE_OPT_LOW = 0.45
MOISTURE_OPT_HIGH = 0.75

# Penalty scales — moderate so the agent still acts when needed
K_IRRIG = 0.008       # per mm/day (reduced — was pushing "do nothing")
K_FERT = 0.012        # per kg N/ha/day
K_RUNOFF = 0.05       # per mm/day
K_LEACH = 0.03        # per kg N/ha/day leached
K_DENIT = 0.02        # per kg N/ha/day denitrified

# Seasonal scales (weaker than yield term)
K_SEASON_WATER = 0.001   # per mm season total
K_SEASON_FERT = 0.002    # per kg N season total


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


def daily_yield_component(nstres, trnu):
    """N-health shaping (supports final yield, not a yield proxy itself)."""
    n_bonus = float(np.clip(nstres, 0.0, 1.0))
    uptake = float(np.clip(trnu / 5.0, 0.0, 1.0))
    return 0.5 * n_bonus + 0.5 * uptake


def daily_water_component(moisture, swfac, amir, runoff):
    r_moisture = _band_reward(moisture, MOISTURE_OPT_LOW, MOISTURE_OPT_HIGH)
    r_stress = float(np.clip(swfac, 0.0, 1.0))
    r_resource = -K_IRRIG * float(amir)
    r_loss = -K_RUNOFF * float(max(runoff, 0.0))
    return 0.35 * r_moisture + 0.35 * r_stress + r_resource + r_loss


def daily_fert_component(nstres, trnu, anfer, tleachd, cnox_daily):
    r_n = float(np.clip(nstres, 0.0, 1.0))
    r_uptake = float(np.clip(trnu / 5.0, 0.0, 1.0))
    r_resource = -K_FERT * float(anfer)
    r_loss = -K_LEACH * float(max(tleachd, 0.0)) - K_DENIT * float(max(cnox_daily, 0.0))
    return 0.35 * r_n + 0.35 * r_uptake + r_resource + r_loss


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
    """Dominant harvest signal — normalized to roughly [-1, 1] scale."""
    yield_term = np.clip(grnwt / 10000.0, 0.0, 1.5)
    hi = harvest_index(grnwt, topwt)
    ane = np.clip(agronomic_n_efficiency(grnwt, total_n) / 50.0, 0.0, 1.0)
    return float(yield_term + 0.2 * hi + 0.15 * ane)


def seasonal_water_component(total_water):
    return float(-K_SEASON_WATER * total_water)


def seasonal_fert_component(total_n, grnwt):
    penalty = -K_SEASON_FERT * total_n
    ane_bonus = np.clip(agronomic_n_efficiency(grnwt, total_n) / 80.0, 0.0, 0.5)
    return float(penalty + ane_bonus)


def compute_reward_vector(state, context, action, totals, prev_cnox, done):
    """
    Build the 3-dim reward vector for one day.

    Args:
        state: full DSSAT state dict (post-processed)
        context: static soil context (dul, ll, sat, dlayr)
        action: {'anfer', 'amir'}
        totals: {'water', 'nitrogen'}
        prev_cnox: cumulative cnox at previous step (for daily delta)
        done: harvest flag
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

    # Daily components
    r_yield = daily_yield_component(nstres, trnu)
    r_water = daily_water_component(moisture, swfac, amir, runoff)
    r_fert = daily_fert_component(nstres, trnu, anfer, tleachd, cnox_daily)

    if done:
        r_yield += seasonal_yield_component(grnwt, topwt, totals['nitrogen'])
        r_water += seasonal_water_component(totals['water'])
        r_fert += seasonal_fert_component(totals['nitrogen'], grnwt)

    return np.array([r_yield, r_water, r_fert], dtype=np.float32), moisture
