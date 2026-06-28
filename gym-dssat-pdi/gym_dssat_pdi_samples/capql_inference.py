"""
CAPQL v2 inference — run one season with a fixed preference vector.

Used by the smart-farm web API (interface/api). Requires DSSAT (Linux/Docker).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import import_module
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

CAPQLEnv = import_module('capql_env_v2').CAPQLEnv

OBS_DIM = 14
ACTION_DIM = 2
HIDDEN = 256
ACT_LOW = np.array([0.0, 0.0], dtype=np.float32)
ACT_HIGH = np.array([200.0, 50.0], dtype=np.float32)
ACT_RANGE = ACT_HIGH - ACT_LOW
TOTAL_STEPS = 1_000_000

_DEVICE = torch.device('cpu')
_ACT_MID = torch.FloatTensor((ACT_HIGH + ACT_LOW) / 2.0)
_ACT_HALF = torch.FloatTensor(ACT_RANGE / 2.0)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, ACTION_DIM), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs) * _ACT_HALF + _ACT_MID


from weather_config import resolve_weather

DSSAT_STEP_TIMEOUT_SEC = int(os.environ.get('DSSAT_STEP_TIMEOUT_SEC', '90'))
# ZMQ recv uses DSSAT_ZMQ_TIMEOUT_MS in gym_dssat_pdi.envs.dssat_pdi (default 120s).
AGRONOMIC_GUARDS_ENABLED = os.environ.get('CAPQL_AGRONOMIC_GUARDS', '1') != '0'
TRAINING_WEATHER_ID = 'wgen-ufga'


def _force_close_dssat(env) -> None:
    """Kill hung DSSAT subprocess. Do not call env.close() — it can deadlock on ZMQ."""
    try:
        inner = getattr(getattr(env, 'sos_env', None), 'env', None)
        if inner is not None:
            unwrapped = getattr(inner, 'unwrapped', inner)
            cleanup = getattr(unwrapped, '_cleanup_process', None)
            if cleanup:
                cleanup()
    except Exception:
        pass


def _step_env_with_timeout(env, action) -> Tuple:
    step = len(getattr(env, 'history', {}).get('action', [])) + 1
    wx_id = getattr(getattr(env, 'sos_env', None), '_weather_id', None)
    try:
        result = env.step(action)
    except TimeoutError as exc:
        _force_close_dssat(env)
        raise TimeoutError(
            f'DSSAT stopped responding at simulation step {step} '
            f'(timeout {DSSAT_STEP_TIMEOUT_SEC}s'
            f'{f", weather={wx_id}" if wx_id else ""}). '
            'Try Gainesville climate or a different weather seed.'
        ) from exc
    if result is None or (isinstance(result, tuple) and len(result) >= 3 and result[0] is None):
        _force_close_dssat(env)
        raise TimeoutError(
            f'DSSAT stopped responding at simulation step {step} '
            f'(weather={wx_id or "unknown"}). '
            'Try Gainesville climate or a different weather seed.'
        )
    return result


def default_model_dir() -> str:
    if os.environ.get('CAPQL_MODEL_DIR'):
        return os.environ['CAPQL_MODEL_DIR']
    if os.path.isdir('/workspace/models/capql_v2'):
        return '/workspace/models/capql_v2'
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, '..', '..', 'models', 'capql_v2'))


def load_actor(model_dir: Optional[str] = None) -> Actor:
    model_dir = model_dir or default_model_dir()
    path = os.path.join(model_dir, 'actor.pt')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'CAPQL actor not found: {path}')
    actor = Actor()
    actor.load_state_dict(torch.load(path, map_location=_DEVICE))
    actor.eval()
    return actor


def _normalize_preference(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32).flatten()
    if w.shape != (3,):
        raise ValueError('preference must have 3 values [yield, n_eff, water]')
    s = float(w.sum())
    if s <= 0:
        raise ValueError('preference weights must sum to a positive value')
    return (w / s).astype(np.float32)


def _apply_preference_caps(env: CAPQLEnv, w: np.ndarray) -> None:
    w_neff = float(w[1])
    w_water = float(w[2])
    env.current_w = w.copy()
    env._max_anfer = float(np.clip(200.0 * (1.0 - w_neff) ** 2, 2.0, 200.0))
    env._max_amir = float(np.clip(50.0 * (1.0 - w_water) ** 2, 3.0, 50.0))


def _apply_agronomic_guards(
    action: np.ndarray,
    crop_before: dict,
    moist_before: tuple[float, bool],
    stress_before: tuple[float, float],
) -> tuple[np.ndarray, list[str]]:
    """Post-process actor output with simple agronomic sanity checks."""
    if not AGRONOMIC_GUARDS_ENABLED:
        return action.astype(np.float32), []

    anfer = float(action[0])
    amir = float(action[1])
    notes: list[str] = []
    dap = int(crop_before.get('dap', 0) or 0)
    xlai = float(crop_before.get('xlai', 0.0) or 0.0)
    moist, is_fc = moist_before
    swfac_b, nstres_b = stress_before

    if amir > 0.5 and is_fc and moist >= 0.85 and swfac_b >= 0.95:
        notes.append('hold_irrig_wet_soil')
        amir = 0.0

    if anfer > 0.5 and nstres_b >= 0.88 and dap < 20 and xlai < 0.6:
        notes.append('hold_n_low_demand')
        anfer = 0.0

    return np.array([anfer, amir], dtype=np.float32), notes


def _recommendation_reason(crop: dict, anfer: float, amir: float) -> str:
    swfac = float(crop.get('swfac', 1.0) or 1.0)
    rain = float(crop.get('rain', 0.0) or 0.0)
    nstres = float(crop.get('nstres', 1.0) or 1.0)

    if amir > 5.0 and swfac < 0.9:
        return (
            'Soil moisture is below the optimal range while plant water stress '
            'is elevated; irrigation is recommended.'
        )
    if anfer > 2.0 and nstres < 0.85:
        return (
            'Crop nitrogen stress is limiting growth; a nitrogen application '
            'is recommended.'
        )
    if rain > 8.0 and amir < 1.0:
        return 'Recent rainfall reduces irrigation need; holding water application low.'
    if anfer < 0.5 and amir < 1.0:
        return 'Crop status is stable; minimal inputs recommended for today.'
    return 'Policy balances yield, nitrogen efficiency, and water saving for current conditions.'


def _moisture_phrase(moisture: float, is_fc: bool) -> str:
    if is_fc:
        pct = int(round(min(1.0, max(0.0, moisture)) * 100))
        return f'soil at {pct}% of field capacity'
    return f'water stress factor {moisture:.2f}'


def _irrigation_note(
    *,
    amir: float,
    rain: float,
    swfac: float,
    dap: int,
    xlai: float,
    moisture_before: float,
    moisture_fc_before: bool,
) -> str:
    if amir >= 0.1:
        parts = []
        if swfac < 0.85:
            parts.append(f'elevated water stress (swfac {swfac:.2f})')
        if moisture_fc_before and moisture_before < 0.55:
            parts.append(_moisture_phrase(moisture_before, True) + ' before watering')
        elif not moisture_fc_before and swfac < 0.9:
            parts.append(_moisture_phrase(moisture_before, False))
        if rain < 2.0 and dap > 0:
            parts.append('little rain today')
        if moisture_fc_before and moisture_before >= 0.85 and swfac >= 0.95:
            parts.append(
                f'{_moisture_phrase(moisture_before, True)} with no plant stress — '
                'learned mid-season irrigation dose (not a wetness trigger)'
            )
        if parts:
            return f'Applied {amir:.1f} mm: {"; ".join(parts)}.'
        return f'Applied {amir:.1f} mm to meet crop water needs.'

    why = []
    if rain >= 8.0:
        why.append(f'{rain:.1f} mm rain — soil fed naturally')
    elif rain >= 3.0:
        why.append(f'{rain:.1f} mm rain reduces irrigation need')
    if moisture_fc_before and moisture_before >= 0.72:
        why.append(f'{_moisture_phrase(moisture_before, True)} — already wet')
    elif swfac >= 0.9:
        why.append(f'low water stress (swfac {swfac:.2f})')
    if dap <= 0:
        why.append('pre-plant / at planting')
    if 0 < dap < 10 and xlai < 0.2:
        why.append('seedling stage — low water demand')
    if not why:
        why.append('policy held water to balance yield and water saving')
    return f'0 mm irrigation: {"; ".join(why)}.'


def _nitrogen_note(*, anfer: float, nstres: float, dap: int, xlai: float) -> str:
    if anfer >= 0.1:
        parts = []
        if nstres < 0.85:
            parts.append(f'N stress limiting (nstres {nstres:.2f})')
        if dap >= 20:
            parts.append(f'mid-season demand (DAP {dap})')
        if xlai >= 1.0:
            parts.append(f'active canopy (LAI {xlai:.1f})')
        if parts:
            return f'Applied {anfer:.1f} kg/ha N: {"; ".join(parts)}.'
        return f'Applied {anfer:.1f} kg/ha N for crop demand.'

    why = []
    if nstres >= 0.88:
        why.append(f'adequate N supply (nstres {nstres:.2f})')
    if dap < 14:
        why.append(f'early season (DAP {dap}) — small N demand')
    if xlai < 0.4 and dap < 25:
        why.append(f'small canopy (LAI {xlai:.1f})')
    if dap <= 0:
        why.append('before crop establishment')
    if not why:
        why.append('policy held N for efficiency and lower leaching risk')
    return f'0 kg/ha nitrogen: {"; ".join(why)}.'


def _soil_moisture_for_day(sos, crop: dict, sos_state: dict) -> tuple[float, bool]:
    """
    Return (moisture 0–1, is_field_capacity_fraction).

    Prefer plant-available water / field capacity from soil layers.
    Fall back to swfac (stress factor, not FC) only when layers are unavailable.
    """
    mr = sos_state.get('moisture_ratio')
    if mr is not None:
        return float(mr), True

    mr = sos._moisture_ratio(crop)
    if mr is not None:
        return float(mr), True

    merged = sos._build_observation(crop)
    sw_layers = merged.get('crop_sw')
    if sw_layers is not None:
        mr = sos._moisture_ratio({'sw': sw_layers})
        if mr is not None:
            return float(mr), True

    swfac = float(sos_state.get('swfac', crop.get('swfac', 1.0)) or 1.0)
    return swfac, False


@dataclass
class DayRecord:
    day: int
    dap: int
    rain_mm: float
    tmax_c: float
    srad: float
    irrigation_mm: float
    nitrogen_kg_ha: float
    soil_moisture: float
    soil_moisture_fc: bool
    soil_moisture_before: float
    soil_moisture_fc_before: bool
    swfac: float
    swfac_before: float
    nstres: float
    nstres_before: float
    cumulative_irrigation_mm: float
    growth_stage: float
    xlai: float
    grnwt: float
    topwt: float
    n_uptake: float
    n_uptake_cumulative_kg_ha: float
    recommendation: str
    irrigation_note: str = ''
    nitrogen_note: str = ''
    agronomic_guard: str = ''

    def to_dict(self) -> dict:
        return {
            'day': self.day,
            'dap': self.dap,
            'rain_mm': round(self.rain_mm, 2),
            'tmax_c': round(self.tmax_c, 1),
            'srad': round(self.srad, 1),
            'irrigation_mm': round(self.irrigation_mm, 2),
            'nitrogen_kg_ha': round(self.nitrogen_kg_ha, 2),
            'soil_moisture': round(self.soil_moisture, 3),
            'soil_moisture_fc': self.soil_moisture_fc,
            'soil_moisture_before': round(self.soil_moisture_before, 3),
            'soil_moisture_fc_before': self.soil_moisture_fc_before,
            'swfac': round(self.swfac, 3),
            'swfac_before': round(self.swfac_before, 3),
            'nstres': round(self.nstres, 3),
            'nstres_before': round(self.nstres_before, 3),
            'cumulative_irrigation_mm': round(self.cumulative_irrigation_mm, 1),
            'growth_stage': round(self.growth_stage, 2),
            'xlai': round(self.xlai, 3),
            'grnwt': round(self.grnwt, 1),
            'topwt': round(self.topwt, 1),
            'n_uptake': round(self.n_uptake, 3),
            'n_uptake_cumulative_kg_ha': round(self.n_uptake_cumulative_kg_ha, 1),
            'recommendation': self.recommendation,
            'irrigation_note': self.irrigation_note,
            'nitrogen_note': self.nitrogen_note,
            'agronomic_guard': self.agronomic_guard,
        }


@dataclass
class SeasonResult:
    preference: List[float]
    seed: int
    days: List[DayRecord] = field(default_factory=list)
    yield_kg_ha: float = 0.0
    total_n_kg_ha: float = 0.0
    total_water_mm: float = 0.0
    total_rain_mm: float = 0.0
    harvest_index: float = 0.0
    ane: float = 0.0
    water_productivity: float = 0.0
    r_yield: float = 0.0
    r_ane: float = 0.0
    r_water_eff: float = 0.0
    cum_reward: float = 0.0
    ep_length: int = 0
    weather_id: str = ''
    weather_label: str = ''

    def to_dict(self) -> dict:
        last = self.days[-1].to_dict() if self.days else {}
        wx = resolve_weather(self.weather_id or None)
        return {
            'preference': [round(float(x), 4) for x in self.preference],
            'seed': self.seed,
            'weather_id': self.weather_id or wx.id,
            'weather_label': self.weather_label or wx.label,
            'ep_length': self.ep_length,
            'days': [d.to_dict() for d in self.days],
            'summary': {
                'yield_kg_ha': round(self.yield_kg_ha, 1),
                'total_n_kg_ha': round(self.total_n_kg_ha, 1),
                'total_water_mm': round(self.total_water_mm, 1),
                'total_rain_mm': round(self.total_rain_mm, 1),
                'weather_id': self.weather_id or wx.id,
                'weather_label': self.weather_label or wx.label,
                'weather_path': wx.path,
                'harvest_index': round(self.harvest_index, 3),
                'ane': round(self.ane, 2),
                'water_productivity': round(self.water_productivity, 2),
                'r_yield': round(self.r_yield, 4),
                'r_ane': round(self.r_ane, 4),
                'r_water_eff': round(self.r_water_eff, 4),
                'cum_reward': round(self.cum_reward, 4),
            },
            'current': last,
        }


ProgressCallback = Callable[[int, int, Optional[DayRecord], Optional[str]], None]


def _record_day(
    sos,
    day_index: int,
    crop: dict,
    sos_state: dict,
    anfer: float,
    amir: float,
    *,
    moist_before: Optional[tuple[float, bool]] = None,
    stress_before: Optional[tuple[float, float]] = None,
    guard_notes: Optional[list[str]] = None,
) -> DayRecord:
    moisture, moisture_fc = _soil_moisture_for_day(sos, crop, sos_state)
    if moist_before is not None:
        moisture_before, moisture_fc_before = moist_before
    else:
        moisture_before, moisture_fc_before = moisture, moisture_fc
    swfac_after = float(sos_state.get('swfac', crop.get('swfac', 1.0)) or 1.0)
    nstres_after = float(sos_state.get('nstres', crop.get('nstres', 1.0)) or 1.0)
    if stress_before is not None:
        swfac_before, nstres_before = stress_before
    else:
        swfac_before, nstres_before = swfac_after, nstres_after
    totir = float(sos_state.get('totir', crop.get('totir', 0.0)) or 0.0)
    rain = float(crop.get('rain', 0.0) or 0.0)
    dap = int(sos_state.get('dap', crop.get('dap', 0)) or 0)
    xlai = float(crop.get('xlai', 0.0) or 0.0)
    irr_note = _irrigation_note(
        amir=amir,
        rain=rain,
        swfac=swfac_before,
        dap=dap,
        xlai=xlai,
        moisture_before=moisture_before,
        moisture_fc_before=moisture_fc_before,
    )
    n_note = _nitrogen_note(anfer=anfer, nstres=nstres_before, dap=dap, xlai=xlai)
    guard = guard_notes or []
    if 'hold_irrig_wet_soil' in guard and amir < 0.1:
        irr_note = (
            '0 mm irrigation: soil already near field capacity with low plant stress — '
            'agronomic guard held water.'
        )
    if 'hold_n_low_demand' in guard and anfer < 0.1:
        n_note = (
            '0 kg/ha nitrogen: early season with adequate N supply — '
            'agronomic guard held fertilizer.'
        )
    return DayRecord(
        day=day_index,
        dap=dap,
        rain_mm=rain,
        tmax_c=float(crop.get('tmax', 0.0) or 0.0),
        srad=float(crop.get('srad', 0.0) or 0.0),
        irrigation_mm=amir,
        nitrogen_kg_ha=anfer,
        soil_moisture=moisture,
        soil_moisture_fc=moisture_fc,
        soil_moisture_before=moisture_before,
        soil_moisture_fc_before=moisture_fc_before,
        swfac=swfac_after,
        swfac_before=swfac_before,
        nstres=nstres_after,
        nstres_before=nstres_before,
        cumulative_irrigation_mm=totir,
        growth_stage=float(crop.get('vstage', 0.0) or 0.0),
        xlai=float(crop.get('xlai', 0.0) or 0.0),
        grnwt=float(crop.get('grnwt', 0.0) or 0.0),
        topwt=float(crop.get('topwt', 0.0) or 0.0),
        n_uptake=float(crop.get('trnu', 0.0) or sos_state.get('trnu', 0.0) or 0.0),
        n_uptake_cumulative_kg_ha=float(
            crop.get('wtnup', 0.0) or sos_state.get('wtnup', 0.0) or 0.0
        ),
        recommendation=_recommendation_reason(crop, anfer, amir),
        irrigation_note=irr_note,
        nitrogen_note=n_note,
        agronomic_guard=', '.join(guard),
    )


def _finalize_season(
    w: np.ndarray,
    seed: int,
    days: List[DayRecord],
    last_info: dict,
    crop: dict,
    cum_daily: float,
    *,
    weather_id: str = '',
    weather_label: str = '',
) -> SeasonResult:
    state = last_info.get('sos_state', {})
    rv = last_info.get('reward_vec', np.zeros(3, dtype=np.float32))
    grnwt = float(state.get('grnwt', 0.0))
    total_n = float(state.get('total_nitrogen', 0.0))
    total_w = float(state.get('total_water', 0.0))
    total_rain = float(state.get('total_rain', 0.0))
    topwt = float(crop.get('topwt', 0.0) or 0.0) if days else 0.0
    hi = grnwt / topwt if topwt > 1e-6 else 0.0
    ane = grnwt / max(total_n, 1.0)
    wp = grnwt / max(total_w, 1.0)
    return SeasonResult(
        preference=w.tolist(),
        seed=seed,
        days=days,
        yield_kg_ha=grnwt,
        total_n_kg_ha=total_n,
        total_water_mm=total_w,
        total_rain_mm=total_rain,
        harvest_index=hi,
        ane=ane,
        water_productivity=wp,
        r_yield=float(rv[0]),
        r_ane=float(rv[1]),
        r_water_eff=float(rv[2]),
        cum_reward=cum_daily + float(np.dot(w, rv)),
        ep_length=len(days),
        weather_id=weather_id,
        weather_label=weather_label,
    )


class AdvisorySession:
    """Step DSSAT + CAPQL one day at a time (weather in → state → action out)."""

    def __init__(
        self,
        preference: List[float],
        seed: int = 123,
        actor: Optional[Actor] = None,
        weather_id: Optional[str] = None,
    ):
        self.w = _normalize_preference(np.array(preference, dtype=np.float32))
        self.seed = int(seed)
        self.weather_id = weather_id
        wx = resolve_weather(weather_id)
        self.weather_label = wx.label
        self.actor = actor or load_actor()
        self.env: Optional[CAPQLEnv] = None
        self.obs: Optional[np.ndarray] = None
        self.days: List[DayRecord] = []
        self.done = False
        self.cum_daily = 0.0
        self.last_info: dict = {}
        self.result: Optional[SeasonResult] = None

    def start(self) -> None:
        if self.env is not None:
            self.close()
        wx = resolve_weather(self.weather_id)
        print(f'[advisory] seed={self.seed} weather={wx.id} starting DSSAT session…', flush=True)
        self.env = CAPQLEnv(
            mode='all',
            dssat_seed=self.seed,
            run_dssat_location='run_dssat',
            weather_id=self.weather_id,
        )
        self.obs, _ = self.env.reset()
        _apply_preference_caps(self.env, self.w)
        self.obs[-3:] = self.w
        self.days = []
        self.done = False
        self.cum_daily = 0.0
        self.last_info = {}
        self.result = None

    def step(self) -> DayRecord:
        if self.env is None:
            raise RuntimeError('Session not started')
        if self.done:
            raise RuntimeError('Season already finished')

        sos = self.env.sos_env
        crop_before = dict(sos._last_crop_obs)
        moist_before = _soil_moisture_for_day(sos, crop_before, {})
        stress_before = (
            float(crop_before.get('swfac', 1.0) or 1.0),
            float(crop_before.get('nstres', 1.0) or 1.0),
        )

        with torch.no_grad():
            obs_t = torch.FloatTensor(self.obs).unsqueeze(0)
            action = self.actor(obs_t).squeeze(0).numpy().astype(np.float32)

        action, guard_notes = _apply_agronomic_guards(
            action, crop_before, moist_before, stress_before,
        )

        self.obs, r_daily, self.done, _, self.last_info = _step_env_with_timeout(self.env, action)
        self.obs[-3:] = self.w
        self.cum_daily += float(r_daily)

        crop = dict(sos._last_crop_obs)
        sos_state = self.last_info.get('sos_state', {})
        applied = self.last_info.get('action_applied', action)
        anfer = float(applied[0])
        amir = float(applied[1])

        rec = _record_day(
            sos, len(self.days) + 1, crop, sos_state, anfer, amir,
            moist_before=moist_before, stress_before=stress_before,
            guard_notes=guard_notes,
        )
        self.days.append(rec)

        if self.done:
            self.result = _finalize_season(
                self.w, self.seed, self.days, self.last_info, crop, self.cum_daily,
                weather_id=resolve_weather(self.weather_id).id,
                weather_label=self.weather_label,
            )
            print(f'[advisory] harvest DAP {rec.dap} yield {self.result.yield_kg_ha:.0f}', flush=True)

        return rec

    def snapshot(self) -> dict:
        partial = _finalize_season(
            self.w,
            self.seed,
            self.days,
            self.last_info or {'sos_state': {}},
            dict(self.env.sos_env._last_crop_obs) if self.env else {},
            self.cum_daily,
            weather_id=resolve_weather(self.weather_id).id,
            weather_label=self.weather_label,
        ) if self.days else None

        if self.result is not None:
            payload = self.result.to_dict()
        elif partial is not None:
            payload = partial.to_dict()
        else:
            payload = None

        return {
            'seed': self.seed,
            'weather_id': resolve_weather(self.weather_id).id,
            'weather_label': self.weather_label,
            'preference': [round(float(x), 4) for x in self.w.tolist()],
            'started': self.env is not None,
            'done': self.done,
            'day_count': len(self.days),
            'current': self.days[-1].to_dict() if self.days else None,
            'result': payload,
        }

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


def run_season(
    preference: List[float],
    seed: int = 123,
    actor: Optional[Actor] = None,
    on_progress: Optional[ProgressCallback] = None,
    weather_id: Optional[str] = None,
) -> SeasonResult:
    """Run one CAPQL v2 season with fixed preference w = [yield, n_eff, water]."""
    w = _normalize_preference(np.array(preference, dtype=np.float32))
    wx = resolve_weather(weather_id)
    est_len = 165

    if on_progress:
        on_progress(0, est_len, None, 'loading_model')
    actor = actor or load_actor()

    if on_progress:
        on_progress(0, est_len, None, 'starting_dssat')
    print(f'[run_season] seed={seed} weather={wx.id} creating CAPQL env…', flush=True)
    env = CAPQLEnv(
        mode='all',
        dssat_seed=seed,
        run_dssat_location='run_dssat',
        weather_id=weather_id,
    )
    print('[run_season] env ready, resetting season…', flush=True)
    obs, _ = env.reset()
    _apply_preference_caps(env, w)
    obs[-3:] = w

    if on_progress:
        on_progress(0, est_len, None, 'simulating')

    sos = env.sos_env
    days: List[DayRecord] = []
    cum_daily = 0.0
    done = False
    last_info: dict = {}

    while not done:
        crop_before = dict(sos._last_crop_obs)
        moist_before = _soil_moisture_for_day(sos, crop_before, {})
        stress_before = (
            float(crop_before.get('swfac', 1.0) or 1.0),
            float(crop_before.get('nstres', 1.0) or 1.0),
        )

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action = actor(obs_t).squeeze(0).numpy().astype(np.float32)

        action, guard_notes = _apply_agronomic_guards(
            action, crop_before, moist_before, stress_before,
        )

        try:
            obs, r_daily, done, _, last_info = _step_env_with_timeout(env, action)
        except TimeoutError:
            raise
        obs[-3:] = w
        cum_daily += float(r_daily)

        crop = dict(sos._last_crop_obs)
        sos_state = last_info.get('sos_state', {})
        applied = last_info.get('action_applied', action)
        anfer = float(applied[0])
        amir = float(applied[1])

        rec = _record_day(
            sos, len(days) + 1, crop, sos_state, anfer, amir,
            moist_before=moist_before, stress_before=stress_before,
            guard_notes=guard_notes,
        )
        days.append(rec)

        if on_progress:
            on_progress(len(days), est_len, rec, 'simulating')
            if len(days) == 1 or len(days) % 20 == 0:
                print(f'[run_season] step {len(days)} DAP {rec.dap}', flush=True)

    try:
        env.close()
    except Exception:
        pass

    crop = dict(sos._last_crop_obs)
    return _finalize_season(
        w, seed, days, last_info, crop, cum_daily,
        weather_id=wx.id,
        weather_label=wx.label,
    )
