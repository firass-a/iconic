"""City climate presets — NASA POWER → DSSAT measured .WTH for custom locations."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from nasa_climate import (
    CityClimateSpec,
    DSSAT_BENCHMARK_WTH,
    build_city_cli,
    build_city_wth_library,
    download_nasa_power,
)

# DSSAT FileX for this project uses WSTA=UFGA; measured weather must be UFGA8201.WTH
DSSAT_STATION = 'UFGA'


CITY_PRESETS: Dict[str, CityClimateSpec] = {
    'algiers': CityClimateSpec(
        city_id='algiers',
        name='Algiers, Algeria',
        lat=36.754,
        lon=3.059,
        elev=50,
        station=DSSAT_STATION,
        start_year=2004,
        end_year=2023,
    ),
}


def _repo_weather_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent / 'weather'


def city_cli_path(city_id: str) -> Path:
    return _repo_weather_root() / 'cities' / city_id / 'UFGA.CLI'


def city_wth_root(city_id: str) -> Path:
    return _repo_weather_root() / 'cities' / city_id / 'wth'


def city_wth_path(city_id: str, source_year: int) -> Path:
    return city_wth_root(city_id) / str(source_year) / DSSAT_BENCHMARK_WTH


def city_weather_option_id(city_id: str) -> str:
    return f'wgen-nasa-{city_id}'


def get_city_spec(city_id: str) -> CityClimateSpec:
    if city_id not in CITY_PRESETS:
        raise KeyError(f'Unknown city: {city_id}')
    return CITY_PRESETS[city_id]


def pick_nasa_year(city_id: str, seed: int) -> int:
    spec = get_city_spec(city_id)
    span = spec.end_year - spec.start_year + 1
    return spec.start_year + (int(seed) % span)


def is_city_weather_built(city_id: str) -> bool:
    spec = get_city_spec(city_id)
    return city_wth_path(city_id, spec.start_year).is_file()


def city_weather_needs_rebuild(city_id: str) -> bool:
    if not is_city_weather_built(city_id):
        return True
    spec = get_city_spec(city_id)
    for year in range(spec.start_year, spec.end_year + 1):
        if not city_wth_path(city_id, year).is_file():
            return True
    return False


def ensure_city_climate(city_id: str, *, force: bool = False) -> Optional[dict]:
    if not force and not city_weather_needs_rebuild(city_id):
        return None
    return build_city_climate(city_id)


def build_city_climate(city_id: str) -> dict:
    spec = get_city_spec(city_id)
    records = download_nasa_power(spec.lat, spec.lon, spec.start_year, spec.end_year)
    wth_meta = build_city_wth_library(spec, str(city_wth_root(city_id)), records=records)
    cli_meta = build_city_cli(spec, str(city_cli_path(city_id)))
    return {
        **cli_meta,
        **wth_meta,
        'weather_id': city_weather_option_id(city_id),
        'label': f'{spec.name} — NASA daily weather (seed picks year)',
        'mode': 'measured',
    }


def resolve_city_wth_path(city_id: str, seed: int = 123) -> Optional[str]:
    year = pick_nasa_year(city_id, seed)
    path = city_wth_path(city_id, year)
    if path.is_file():
        return os.path.normpath(str(path))
    docker = Path(f'/workspace/weather/cities/{city_id}/wth/{year}/{DSSAT_BENCHMARK_WTH}')
    if docker.is_file():
        return os.path.normpath(str(docker))
    return None


# Legacy helpers (CLI still built for reference)
def is_city_cli_built(city_id: str) -> bool:
    return is_city_weather_built(city_id)


def city_cli_needs_rebuild(city_id: str) -> bool:
    return city_weather_needs_rebuild(city_id)


def resolve_city_cli_path(city_id: str) -> Optional[str]:
    path = city_cli_path(city_id)
    if path.is_file():
        return os.path.normpath(str(path))
    docker = Path(f'/workspace/weather/cities/{city_id}/UFGA.CLI')
    if docker.is_file():
        return os.path.normpath(str(docker))
    return None
