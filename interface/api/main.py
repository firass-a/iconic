"""
Smart Farm DSS API — real CAPQL v2 + DSSAT simulations.
"""
from __future__ import annotations

import csv
import os
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# gym_dssat_pdi_samples on path (Docker: /workspace/gym-dssat-pdi/gym_dssat_pdi_samples)
_SAMPLES = os.environ.get(
    'GYM_SAMPLES_DIR',
    str(Path(__file__).resolve().parents[2] / 'gym-dssat-pdi' / 'gym_dssat_pdi_samples'),
)
if _SAMPLES not in sys.path:
    sys.path.insert(0, _SAMPLES)

from capql_inference import (  # noqa: E402
    AGRONOMIC_GUARDS_ENABLED,
    TRAINING_WEATHER_ID,
    AdvisorySession,
    load_actor,
    run_season,
)
from weather_config import (  # noqa: E402
    DEFAULT_WEATHER_ID,
    list_available_weather_options,
    resolve_ufga_cli,
    resolve_weather,
)
from city_climate import CITY_PRESETS, build_city_climate, city_cli_path, ensure_city_climate, is_city_cli_built  # noqa: E402

app = FastAPI(title='Smart Farm DSS API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

_executor = ThreadPoolExecutor(max_workers=1)
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_advisory_sessions: Dict[str, AdvisorySession] = {}
_advisory_lock = threading.Lock()
_actor = None
_actor_lock = threading.Lock()


def _workspace_root() -> Path:
    if os.path.isdir('/workspace'):
        return Path('/workspace')
    return Path(__file__).resolve().parents[2]


def _eval_csv_path() -> Path:
    p = _workspace_root() / 'capql_v2_eval_results.csv'
    if p.is_file():
        return p
    return _workspace_root() / 'gym-dssat-pdi' / 'gym_dssat_pdi_samples' / 'capql_v2_eval_results.csv'


def _get_actor():
    global _actor
    with _actor_lock:
        if _actor is None:
            _actor = load_actor()
        return _actor


class PreferenceBody(BaseModel):
    w_yield: float = Field(..., ge=0)
    w_neff: float = Field(..., ge=0)
    w_water: float = Field(..., ge=0)
    seed: int = 123
    weather_id: Optional[str] = None

    @field_validator('w_yield', 'w_neff', 'w_water')
    @classmethod
    def non_negative(cls, v: float) -> float:
        return float(v)

    def as_vector(self) -> List[float]:
        s = self.w_yield + self.w_neff + self.w_water
        if s <= 0:
            raise ValueError('preference weights must sum to > 0')
        return [self.w_yield / s, self.w_neff / s, self.w_water / s]


class SimulateRequest(BaseModel):
    preference: PreferenceBody


def _load_pareto() -> List[dict]:
    path = _eval_csv_path()
    if not path.is_file():
        return []
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append({
                'id': f"{r['corner']}-{r['episode']}",
                'corner': r['corner'],
                'episode': int(r['episode']),
                'preference': [
                    float(r['w_yield']),
                    float(r['w_neff']),
                    float(r['w_water']),
                ],
                'yield_kg_ha': float(r['yield_kg_ha']),
                'total_n_kg_ha': float(r['total_N_kg_ha']),
                'total_water_mm': float(r['total_W_mm']),
                'r_yield': float(r['R_yield']),
                'r_ane': float(r['R_ane']),
                'r_water_eff': float(r['R_water_eff']),
            })
    return rows


def _run_job(job_id: str, preference: List[float], seed: int, weather_id: Optional[str]) -> None:
    def on_progress(current: int, total: int, day, phase: Optional[str] = None) -> None:
        pct = min(99, max(2, int(100 * current / max(total, 1)))) if current else {
            'loading_model': 2,
            'starting_dssat': 4,
            'simulating': 5,
        }.get(phase or '', 5)
        with _jobs_lock:
            _jobs[job_id]['progress'] = {
                'current_day': current,
                'current_dap': int(getattr(day, 'dap', 0) or 0) if day else _jobs[job_id]['progress'].get('current_dap', 0),
                'total_days': total,
                'percent': pct,
                'phase': phase or ('simulating' if current else 'starting_dssat'),
            }

    try:
        with _jobs_lock:
            _jobs[job_id]['status'] = 'running'
            _jobs[job_id]['progress'] = {
                'current_day': 0,
                'current_dap': 0,
                'total_days': 165,
                'percent': 1,
                'phase': 'loading_model',
            }
        print(f'[simulate] job {job_id} seed={seed} loading actor…', flush=True)
        actor = _get_actor()
        print(f'[simulate] job {job_id} starting DSSAT season…', flush=True)
        result = run_season(
            preference=preference,
            seed=seed,
            actor=actor,
            on_progress=on_progress,
            weather_id=weather_id,
        )
        with _jobs_lock:
            _jobs[job_id]['status'] = 'completed'
            _jobs[job_id]['progress'] = {
                'current_day': result.ep_length,
                'current_dap': result.days[-1].dap if result.days else 0,
                'total_days': result.ep_length,
                'percent': 100,
                'phase': 'done',
            }
            _jobs[job_id]['result'] = result.to_dict()
            _jobs[job_id]['completed_at'] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        print(f'[simulate] job {job_id} failed: {exc}', flush=True)
        with _jobs_lock:
            _jobs[job_id]['status'] = 'failed'
            _jobs[job_id]['error'] = str(exc)


@app.on_event('startup')
def _startup():
    cli = resolve_ufga_cli()
    if cli:
        print(f'[startup] UFGA climate file: {cli}')
    else:
        print('[startup] WARNING: UFGA.CLI not found — WGEN rain may be zero')
    for city_id in CITY_PRESETS:
        try:
            meta = ensure_city_climate(city_id)
            if meta:
                print(f'[startup] rebuilt NASA climate for {city_id}: {meta["path"]}', flush=True)
        except Exception as exc:
            print(f'[startup] WARNING: could not rebuild {city_id} climate: {exc}', flush=True)


@app.get('/health')
def health():
    return {
        'ok': True,
        'api_version': '1.1.0',
        'features': {'advisory_daily': True, 'weather_select': True},
        'dssat': os.path.isfile('/opt/dssat_pdi/run_dssat') or bool(os.environ.get('PATH')),
        'actor_loaded': _actor is not None,
        'weather_cli': resolve_ufga_cli(),
        'weather_cli_loaded': resolve_ufga_cli() is not None,
        'weather_options_count': len(list_available_weather_options()),
        'eval_csv': str(_eval_csv_path()),
        'eval_csv_exists': _eval_csv_path().is_file(),
    }


@app.get('/config')
def config():
    weather_options = [o.to_dict() for o in list_available_weather_options()]
    default_wx = next((o for o in weather_options if o['recommended']), weather_options[0] if weather_options else None)
    return {
        'model': 'CAPQL v2',
        'crop': 'Maize (CERES-Maize)',
        'location': 'DSSAT benchmark site (UFGA)',
        'objectives': ['yield', 'nitrogen_efficiency', 'water_saving'],
        'preference_labels': {
            'w_yield': 'Yield',
            'w_neff': 'Nitrogen efficiency',
            'w_water': 'Water saving',
        },
        'presets': {
            'maximum_yield': [1.0, 0.0, 0.0],
            'n_efficiency': [0.0, 1.0, 0.0],
            'water_saver': [0.0, 0.0, 1.0],
            'balanced': [1 / 3, 1 / 3, 1 / 3],
        },
        'defaults': {
            'seed': 123,
            'weather_id': default_wx['id'] if default_wx else DEFAULT_WEATHER_ID,
        },
        'weather_options': weather_options,
        'city_presets': [
            {
                'city_id': cid,
                'name': spec.name,
                'lat': spec.lat,
                'lon': spec.lon,
                'weather_id': f'wgen-nasa-{cid}',
                'built': is_city_cli_built(cid),
                'period': f'{spec.start_year}–{spec.end_year}',
            }
            for cid, spec in CITY_PRESETS.items()
        ],
        'planting_date': '2024-04-15',
        'training_weather_id': TRAINING_WEATHER_ID,
        'training_weather_label': 'Gainesville WGEN (UFGA — training climate)',
        'agronomic_guards': AGRONOMIC_GUARDS_ENABLED,
        'read_only_fields': [
            'crop', 'location', 'soil_profile', 'planting_date', 'season_length',
        ],
        'notes': (
            'Simulation runs a full DSSAT season (~160 days). Choose synthetic WGEN weather '
            '(varies with seed) or a measured Gainesville .WTH year. Crop, soil, and planting '
            'are fixed by the DSSAT experiment configuration.'
        ),
    }


@app.get('/weather')
def weather_options():
    options = [o.to_dict() for o in list_available_weather_options()]
    return {'options': options, 'count': len(options), 'default_id': DEFAULT_WEATHER_ID}


@app.post('/weather/cities/{city_id}/build')
def build_city_weather(city_id: str, force: bool = False):
    if city_id not in CITY_PRESETS:
        raise HTTPException(status_code=404, detail=f'Unknown city: {city_id}')
    try:
        if force:
            meta = build_city_climate(city_id)
        else:
            meta = ensure_city_climate(city_id) or {
                'weather_id': f'wgen-nasa-{city_id}',
                'path': str(city_cli_path(city_id)),
                'message': 'already up to date',
            }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f'NASA climate build failed: {exc}') from exc
    options = [o.to_dict() for o in list_available_weather_options()]
    return {
        'ok': True,
        'city_id': city_id,
        'weather_id': meta['weather_id'],
        'meta': meta,
        'weather_options': options,
    }


@app.get('/pareto')
def pareto():
    points = _load_pareto()
    return {'model': 'capql_v2', 'points': points, 'count': len(points)}


@app.post('/simulate')
def start_simulation(body: SimulateRequest):
    try:
        w = body.preference.as_vector()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job_id = str(uuid.uuid4())
    weather_id = body.preference.weather_id
    with _jobs_lock:
        _jobs[job_id] = {
            'id': job_id,
            'status': 'queued',
            'preference': w,
            'seed': body.preference.seed,
            'weather_id': resolve_weather(weather_id).id,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'progress': {'current_day': 0, 'current_dap': 0, 'total_days': 165, 'percent': 0, 'phase': 'queued'},
            'result': None,
            'error': None,
        }

    _executor.submit(_run_job, job_id, w, body.preference.seed, weather_id)
    return {'job_id': job_id, 'status': 'queued'}


@app.get('/simulate/{job_id}')
def get_simulation(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')
    return job


@app.get('/simulate/latest/result')
def latest_result():
    with _jobs_lock:
        completed = [
            j for j in _jobs.values()
            if j.get('status') == 'completed' and j.get('result')
        ]
    if not completed:
        return {'result': None}
    completed.sort(key=lambda j: j.get('completed_at', ''), reverse=True)
    return {'result': completed[0]['result'], 'job_id': completed[0]['id']}


@app.post('/advisory/session')
def start_advisory_session(body: SimulateRequest):
    try:
        w = body.preference.as_vector()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session_id = str(uuid.uuid4())
    try:
        actor = _get_actor()
        session = AdvisorySession(
            preference=w,
            seed=body.preference.seed,
            actor=actor,
            weather_id=body.preference.weather_id,
        )
        session.start()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with _advisory_lock:
        stale = list(_advisory_sessions.keys())
        for sid in stale:
            try:
                _advisory_sessions[sid].close()
            except Exception:
                pass
            del _advisory_sessions[sid]
        _advisory_sessions[session_id] = session

    snap = session.snapshot()
    return {'session_id': session_id, **snap}


@app.get('/advisory/session/{session_id}')
def get_advisory_session(session_id: str):
    with _advisory_lock:
        session = _advisory_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail='Advisory session not found')
    return {'session_id': session_id, **session.snapshot()}


@app.post('/advisory/session/{session_id}/step')
def step_advisory_session(session_id: str):
    with _advisory_lock:
        session = _advisory_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail='Advisory session not found')
        try:
            session.step()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        snap = session.snapshot()
    return {'session_id': session_id, **snap}


@app.delete('/advisory/session/{session_id}')
def close_advisory_session(session_id: str):
    with _advisory_lock:
        session = _advisory_sessions.pop(session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail='Advisory session not found')
    session.close()
    return {'ok': True}
