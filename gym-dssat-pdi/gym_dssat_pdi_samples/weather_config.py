"""Weather file discovery and resolution for gym-DSSAT (UFGA maize benchmark)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class WeatherChoice:
    id: str
    label: str
    mode: str  # 'wgen' | 'measured'
    path: Optional[str]
    available: bool
    year: Optional[int] = None
    recommended: bool = False
    notes: str = ''

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'label': self.label,
            'mode': self.mode,
            'path': self.path,
            'available': self.available,
            'year': self.year,
            'recommended': self.recommended,
            'notes': self.notes,
        }


@dataclass(frozen=True)
class ResolvedWeather:
    id: str
    label: str
    mode: str
    random_weather: bool
    auxiliary_file_paths: List[str]
    path: Optional[str]

    @property
    def uses_seed(self) -> bool:
        return self.mode == 'wgen'


DEFAULT_WEATHER_ID = 'wgen-ufga'
DEFAULT_SEASON_YEAR = 1982
_STATION = 'UFGA'
_WTH_RE = re.compile(r'^UFGA(\d{2})01\.WTH$', re.IGNORECASE)


def _samples_dir() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    return _samples_dir().parent.parent


def _search_dirs() -> List[Path]:
    here = _samples_dir()
    root = _repo_root()
    dirs: List[Path] = []
    for p in (
        Path('/opt/dssat_pdi/Weather'),
        Path('/opt/dssat_pdi/Weather/Climate'),
        Path('/workspace/weather'),
        root / 'weather',
        here / 'test_files',
    ):
        if p.is_dir():
            dirs.append(p)
    return dirs


def _find_file(filename: str) -> Optional[str]:
    for directory in _search_dirs():
        candidate = directory / filename
        if candidate.is_file():
            return os.path.normpath(str(candidate))
    return None


def _wth_year(filename: str) -> Optional[int]:
    m = _WTH_RE.match(filename)
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 58 else 2000 + yy


def _wth_label(year: int) -> str:
    if year == DEFAULT_SEASON_YEAR:
        return f'Gainesville {year} (measured daily — default season)'
    return f'Gainesville {year} (measured daily weather)'


def resolve_ufga_cli() -> str | None:
    """Return path to UFGA.CLI if present (legacy helper)."""
    wx = resolve_weather(DEFAULT_WEATHER_ID)
    if wx.mode == 'wgen' and wx.path:
        return wx.path
    return None


def list_weather_options() -> List[WeatherChoice]:
    """All UFGA weather choices; ``available`` reflects whether the file exists locally."""
    options: List[WeatherChoice] = []

    try:
        from city_climate import CITY_PRESETS, city_weather_option_id, is_city_weather_built

        for city_id, spec in CITY_PRESETS.items():
            built = is_city_weather_built(city_id)
            options.append(WeatherChoice(
                id=city_weather_option_id(city_id),
                label=f'{spec.name} — NASA daily weather (seed picks year)',
                mode='measured',
                path=None,
                available=built,
                recommended=False,
                notes=(
                    f'Daily rain and temperature from NASA POWER {spec.start_year}–{spec.end_year}. '
                    'Weather seed selects which historical year is replayed. '
                    'Crop and soil stay on the Gainesville benchmark.'
                ),
            ))
    except ImportError:
        pass

    repo_cli = _find_file('UFGA.CLI')
    docker_cli = '/opt/dssat_pdi/Weather/Climate/UFGA.CLI'
    docker_cli = docker_cli if os.path.isfile(docker_cli) else None

    if repo_cli and (not docker_cli or os.path.normpath(repo_cli) != os.path.normpath(docker_cli)):
        options.append(WeatherChoice(
            id='wgen-ufga-calibrated',
            label='Synthetic season (NASA POWER calibrated climate)',
            mode='wgen',
            path=repo_cli,
            available=True,
            recommended=False,
            notes='WGEN draws a new season each run; use the weather seed to vary rain.',
        ))

    bundled_path = docker_cli or repo_cli
    options.append(WeatherChoice(
        id='wgen-ufga',
        label='Synthetic season (Gainesville climate — varied each run)',
        mode='wgen',
        path=bundled_path,
        available=bundled_path is not None,
        recommended=True,
        notes='Default for CAPQL training. Set weather seed for a different synthetic year.',
    ))

    seen_wth: set[str] = set()
    for directory in _search_dirs():
        if directory.name == 'Climate':
            continue
        try:
            names = sorted(p.name for p in directory.glob('UFGA*.WTH'))
        except OSError:
            continue
        for name in names:
            if name in seen_wth:
                continue
            seen_wth.add(name)
            year = _wth_year(name)
            if year is None:
                continue
            path = _find_file(name)
            options.append(WeatherChoice(
                id=f'wth-ufga-{year}',
                label=_wth_label(year),
                mode='measured',
                path=path,
                available=path is not None,
                year=year,
                recommended=year == DEFAULT_SEASON_YEAR,
                notes=(
                    'Replay exact daily rain and temperature from DSSAT archives.'
                    if year == DEFAULT_SEASON_YEAR
                    else f'Historical {year} record; default planting is {DEFAULT_SEASON_YEAR}.'
                ),
            ))

    options.sort(key=lambda o: (
        0 if o.id.startswith('wgen-nasa-') else 1,
        0 if o.mode == 'wgen' else 1,
        0 if o.recommended else 1,
        o.year or 0,
        o.id,
    ))
    return options


def list_available_weather_options() -> List[WeatherChoice]:
    return [o for o in list_weather_options() if o.available]


def resolve_weather(weather_id: Optional[str] = None, seed: Optional[int] = None) -> ResolvedWeather:
    """Resolve a weather choice id to DSSAT env kwargs."""
    if os.environ.get('USE_UFGA_CLI', '1').strip().lower() in ('0', 'false', 'no'):
        return ResolvedWeather(
            id='none',
            label='DSSAT default (no climate file)',
            mode='wgen',
            random_weather=True,
            auxiliary_file_paths=[],
            path=None,
        )

    wid = weather_id or os.environ.get('WEATHER_ID') or DEFAULT_WEATHER_ID
    options = {o.id: o for o in list_weather_options()}

    if wid not in options or not options[wid].available:
        fallback = next((o for o in list_weather_options() if o.available), None)
        if fallback is None:
            return ResolvedWeather(
                id='none',
                label='No weather file found',
                mode='wgen',
                random_weather=True,
                auxiliary_file_paths=[],
                path=None,
            )
        wid = fallback.id

    choice = options[wid]

    if wid.startswith('wgen-nasa-'):
        city_id = wid.replace('wgen-nasa-', '', 1)
        try:
            from city_climate import get_city_spec, pick_nasa_year, resolve_city_wth_path

            spec = get_city_spec(city_id)
            use_seed = int(seed if seed is not None else 123)
            year = pick_nasa_year(city_id, use_seed)
            path = resolve_city_wth_path(city_id, use_seed)
            if path:
                return ResolvedWeather(
                    id=choice.id,
                    label=f'{spec.name} — NASA {year} daily replay',
                    mode='measured',
                    random_weather=False,
                    auxiliary_file_paths=[path],
                    path=path,
                )
        except (ImportError, KeyError):
            pass

    if choice.mode == 'wgen':
        return ResolvedWeather(
            id=choice.id,
            label=choice.label,
            mode='wgen',
            random_weather=True,
            auxiliary_file_paths=[choice.path] if choice.path else [],
            path=choice.path,
        )

    return ResolvedWeather(
        id=choice.id,
        label=choice.label,
        mode='measured',
        random_weather=False,
        auxiliary_file_paths=[choice.path] if choice.path else [],
        path=choice.path,
    )
