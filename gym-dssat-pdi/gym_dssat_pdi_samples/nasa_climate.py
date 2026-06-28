"""Download NASA POWER daily data and write DSSAT climate (.CLI) or weather (.WTH) files."""
from __future__ import annotations

import json
import os
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import calendar
from typing import List, Optional

import numpy as np

# UFGA maize benchmark FileX uses 1982 dates (SDATE 82030) and UFGA8201.WTH.
DSSAT_BENCHMARK_YEAR = 1982
DSSAT_BENCHMARK_WTH = 'UFGA8201.WTH'

WET_THRESH = 0.1

# DSSAT WGEN can hang when gamma alpha is very small or Markov probs are extreme.
_ALPHA_MIN = 0.25
_ALPHA_MAX = 5.0
_PDW_MIN = 0.05
_PDW_MAX = 0.95
_STD_MIN = 0.5


def _sanitize_wgen_params(wgen_params: list) -> list:
    """Clamp WGEN statistics to ranges that DSSAT handles reliably."""
    out = []
    for wp in wgen_params:
        row = dict(wp)
        row['alpha'] = float(np.clip(row['alpha'], _ALPHA_MIN, _ALPHA_MAX))
        row['pdw'] = float(np.clip(row['pdw'], _PDW_MIN, _PDW_MAX))
        for key in ('sdsd', 'swsd', 'xdsd', 'xwsd', 'nasd'):
            row[key] = max(float(row[key]), _STD_MIN)
        if row['swmn'] > row['sdmn']:
            row['swmn'] = row['sdmn'] * 0.85
        # Very light rain spread across many wet days can stall DSSAT WGEN.
        if row['rnum'] > 0 and row['rtot'] > 0:
            mean_wet = row['rtot'] / row['rnum']
            if mean_wet < 5.0:
                row['rnum'] = max(1.0, row['rtot'] / 8.0)
        if row['rtot'] < 15.0 and row['rnum'] > 4.0:
            row['rnum'] = max(1.0, min(row['rnum'], row['rtot'] / 6.0))
        out.append(row)
    return out


@dataclass(frozen=True)
class CityClimateSpec:
    city_id: str
    name: str
    lat: float
    lon: float
    elev: int
    station: str = 'UFGA'
    start_year: int = 2004
    end_year: int = 2023


def download_nasa_power(
    lat: float,
    lon: float,
    start_year: int,
    end_year: int,
) -> List[dict]:
    url = (
        'https://power.larc.nasa.gov/api/temporal/daily/point'
        f'?parameters=ALLSKY_SFC_SW_DWN,T2M_MAX,T2M_MIN,PRECTOTCORR'
        f'&community=AG'
        f'&longitude={lon}'
        f'&latitude={lat}'
        f'&start={start_year}0101'
        f'&end={end_year}1231'
        f'&format=JSON'
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    props = data['properties']['parameter']
    srad = props['ALLSKY_SFC_SW_DWN']
    tmax = props['T2M_MAX']
    tmin = props['T2M_MIN']
    rain = props['PRECTOTCORR']

    records = []
    for yyyymmdd, sr in srad.items():
        y, m, d = int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:])
        tx = tmax.get(yyyymmdd, -99)
        tn = tmin.get(yyyymmdd, -99)
        rn = rain.get(yyyymmdd, 0.0)
        if any(v < -90 for v in [sr, tx, tn]):
            continue
        if rn < -90:
            rn = 0.0
        records.append({
            'y': y, 'm': m, 'd': d,
            'srad': float(sr), 'tmax': float(tx),
            'tmin': float(tn), 'rain': float(rn),
        })
    if not records:
        raise RuntimeError(f'NASA POWER returned no usable days for {lat},{lon}')
    return records


def compute_wgen(records: List[dict], start_year: int, end_year: int):
    by_month = defaultdict(list)
    for r in records:
        by_month[r['m']].append(r)

    n_years = end_year - start_year + 1
    monthly_avgs = []
    wgen_params = []

    for m in range(1, 13):
        days = by_month[m]
        srad_all = np.array([d['srad'] for d in days])
        tmax_all = np.array([d['tmax'] for d in days])
        tmin_all = np.array([d['tmin'] for d in days])
        rain_all = np.array([d['rain'] for d in days])
        wet_mask = rain_all >= WET_THRESH

        samn = float(np.mean(srad_all))
        xamn = float(np.mean(tmax_all))
        namn = float(np.mean(tmin_all))
        rtot = float(np.sum(rain_all)) / n_years
        rnum = float(np.sum(wet_mask)) / n_years

        monthly_avgs.append({
            'mth': m, 'samn': samn, 'xamn': xamn, 'namn': namn,
            'rtot': rtot, 'rnum': rnum,
        })

        srad_dry = srad_all[~wet_mask]
        srad_wet = srad_all[wet_mask]
        tmax_dry = tmax_all[~wet_mask]
        tmax_wet = tmax_all[wet_mask]
        rain_wet = rain_all[wet_mask]

        sdmn = float(np.mean(srad_dry)) if len(srad_dry) > 1 else samn
        sdsd = float(np.std(srad_dry)) if len(srad_dry) > 1 else 1.0
        swmn = float(np.mean(srad_wet)) if len(srad_wet) > 1 else samn * 0.6
        swsd = float(np.std(srad_wet)) if len(srad_wet) > 1 else 1.0
        xdmn = float(np.mean(tmax_dry)) if len(tmax_dry) > 1 else xamn
        xdsd = float(np.std(tmax_dry)) if len(tmax_dry) > 1 else 2.0
        xwmn = float(np.mean(tmax_wet)) if len(tmax_wet) > 1 else xamn - 1.0
        xwsd = float(np.std(tmax_wet)) if len(tmax_wet) > 1 else 2.0
        nasd = float(np.std(tmin_all)) if len(tmin_all) > 1 else 2.0

        if len(rain_wet) > 1:
            mu = float(np.mean(rain_wet))
            var = float(np.var(rain_wet))
            alpha = (mu ** 2 / var) if var > 0 else 0.5
        else:
            alpha = 0.3

        by_year = defaultdict(list)
        for d in days:
            by_year[d['y']].append(d)
        dw_count = dry_total = 0
        for yr_days in by_year.values():
            yr_days_sorted = sorted(yr_days, key=lambda x: x['d'])
            rains = [d['rain'] for d in yr_days_sorted]
            for i in range(len(rains) - 1):
                if rains[i] < WET_THRESH:
                    dry_total += 1
                    if rains[i + 1] >= WET_THRESH:
                        dw_count += 1
        pdw = float(dw_count / dry_total) if dry_total > 0 else 0.2

        wgen_params.append({
            'mth': m,
            'sdmn': sdmn, 'sdsd': sdsd,
            'swmn': swmn, 'swsd': swsd,
            'xdmn': xdmn, 'xdsd': xdsd,
            'xwmn': xwmn, 'xwsd': xwsd,
            'namn': namn, 'nasd': nasd,
            'alpha': alpha, 'rtot': rtot,
            'pdw': pdw, 'rnum': rnum,
        })

    return monthly_avgs, wgen_params


def _reference_cli_path() -> Optional[str]:
    from pathlib import Path
    candidates = [
        '/opt/dssat_pdi/Weather/Climate/UFGA.CLI',
        Path(__file__).resolve().parent.parent.parent / 'weather' / 'UFGA.CLI',
        Path(__file__).resolve().parent.parent.parent.parent / 'weather' / 'UFGA.CLI',
    ]
    for candidate in candidates:
        path = str(candidate)
        if os.path.isfile(path):
            return path
    return None


def _parse_wgen_from_cli(path: str) -> list:
    rows = []
    in_wgen = False
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('*WGEN PARAMETERS'):
                in_wgen = True
                continue
            if in_wgen and line.startswith('*'):
                break
            stripped = line.strip()
            if not in_wgen or not stripped or stripped.startswith('@'):
                continue
            parts = stripped.split()
            if len(parts) < 15:
                continue
            rows.append({
                'mth': int(parts[0]),
                'sdmn': float(parts[1]), 'sdsd': float(parts[2]),
                'swmn': float(parts[3]), 'swsd': float(parts[4]),
                'xdmn': float(parts[5]), 'xdsd': float(parts[6]),
                'xwmn': float(parts[7]), 'xwsd': float(parts[8]),
                'namn': float(parts[9]), 'nasd': float(parts[10]),
                'alpha': float(parts[11]), 'rtot': float(parts[12]),
                'pdw': float(parts[13]), 'rnum': float(parts[14]),
            })
    return rows


def _blend_wgen_with_reference(city_wgen: list, ref_wgen: list) -> list:
    """Keep proven Gainesville WGEN shape; apply city rain/temperature totals."""
    ref_by_m = {row['mth']: row for row in ref_wgen}
    blended = []
    for cw in city_wgen:
        ref = ref_by_m.get(cw['mth'])
        if ref is None:
            blended.append(cw)
            continue
        row = dict(ref)
        row['rtot'] = cw['rtot']
        row['rnum'] = cw['rnum']
        row['namn'] = cw['namn']
        row['alpha'] = float(np.clip(0.35 * ref['alpha'] + 0.65 * cw['alpha'], _ALPHA_MIN, _ALPHA_MAX))
        row['pdw'] = float(np.clip(0.5 * ref['pdw'] + 0.5 * cw['pdw'], _PDW_MIN, _PDW_MAX))
        for key in ('sdmn', 'swmn', 'xdmn', 'xwmn'):
            if ref[key] > 0:
                row[key] = float(cw.get(key, ref[key]))
        blended.append(row)
    return _sanitize_wgen_params(blended)


def finalize_wgen_params(city_wgen: list) -> list:
    ref_path = _reference_cli_path()
    if ref_path:
        try:
            ref_wgen = _parse_wgen_from_cli(ref_path)
            if len(ref_wgen) == 12:
                return _blend_wgen_with_reference(city_wgen, ref_wgen)
        except OSError:
            pass
    return _sanitize_wgen_params(city_wgen)


def compute_tav_amp(records: List[dict]) -> tuple[float, float]:
    by_month = defaultdict(list)
    for r in records:
        by_month[r['m']].append((r['tmax'] + r['tmin']) / 2.0)
    monthly_means = [float(np.mean(by_month[m])) for m in range(1, 13)]
    tav = float(np.mean(monthly_means))
    amp = float((max(monthly_means) - min(monthly_means)) / 2.0)
    return round(tav, 1), round(amp, 1)


def write_cli_file(
    *,
    out_path: str,
    spec: CityClimateSpec,
    monthly_avgs: list,
    wgen_params: list,
    tav: float,
    amp: float,
) -> str:
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    sray = round(float(np.mean([ma['samn'] for ma in monthly_avgs])), 1)
    tmxy = round(float(np.mean([ma['xamn'] for ma in monthly_avgs])), 1)
    tmny = round(float(np.mean([ma['namn'] for ma in monthly_avgs])), 1)
    raiy = round(sum(ma['rtot'] for ma in monthly_avgs), 0)

    n_years = spec.end_year - spec.start_year + 1
    lines = [
        f'*CLIMATE:{spec.station}              {spec.name}',
        '',
        '@ INSI      LAT     LONG  ELEV   TAV   AMP  SRAY  TMXY  TMNY  RAIY',
        (
            f'  {spec.station}   {spec.lat:7.3f}  {spec.lon:7.3f}  {spec.elev:4d}'
            f'  {tav:5.1f}  {amp:5.1f}  {sray:4.1f}'
            f'  {tmxy:4.1f}  {tmny:4.1f}  {raiy:4.0f}'
        ),
        '@START  DURN  ANGA  ANGB REFHT WNDHT SOURCE',
        (
            f'  {spec.start_year}  {n_years:4d}  0.25  0.50   2.0   2.0 '
            f'NASA_POWER_{spec.start_year}_{spec.end_year}'
        ),
        '@ GSST  GSDU',
        '     1   365',
        '',
        '*MONTHLY AVERAGES',
        '@  MTH  SAMN  XAMN  NAMN  RTOT  RNUM  SHMN  AMTH  BMTH',
    ]
    for ma in monthly_avgs:
        lines.append(
            f"  {ma['mth']:4d}  {ma['samn']:5.1f}  {ma['xamn']:5.1f}"
            f"  {ma['namn']:5.1f}  {ma['rtot']:6.1f}  {ma['rnum']:5.1f}"
            f"   -99 0.250 0.500"
        )
    lines.extend(['', '*WGEN PARAMETERS',
                  '@  MTH  SDMN  SDSD  SWMN  SWSD  XDMN  XDSD  XWMN  XWSD  NAMN  NASD ALPHA  RTOT   PDW  RNUM'])
    for wp in wgen_params:
        lines.append(
            f"  {wp['mth']:4d}"
            f"  {wp['sdmn']:5.1f}  {wp['sdsd']:5.1f}"
            f"  {wp['swmn']:5.1f}  {wp['swsd']:5.1f}"
            f"  {wp['xdmn']:5.1f}  {wp['xdsd']:5.1f}"
            f"  {wp['xwmn']:5.1f}  {wp['xwsd']:5.1f}"
            f"  {wp['namn']:5.1f}  {wp['nasd']:5.1f}"
            f"  {wp['alpha']:5.3f}  {wp['rtot']:5.1f}"
            f"  {wp['pdw']:5.3f}  {wp['rnum']:5.1f}"
        )
    lines.extend([
        '',
        '*RANGE CHECK VALUES',
        '@      SRAD  TMAX  TMIN  RAIN  DEWP  WIND  SUNH   PAR  TDRY  TWET  EVAP  RHUM',
        'MIN :   0.5 -10.0 -10.0   0.0 -40.0   0.0   0.0   5.0 -10.0 -10.0   0.0   0.0',
        'MAX :  85.0  45.0  35.0 600.0  40.0 500.0 100.0  85.0  45.0  40.0  15.0 100.0',
        'RATE:  70.0  20.0  20.0 500.0   5.0 300.0  90.0  70.0  20.0  20.0  15.0  75.0',
        '',
        '*FLAGGED DATA COUNT',
        '@BEGYR BEGMN BEGDY ENDYR ENDMN ENDDY',
        f'  {spec.start_year}     1     1  {spec.end_year}    12    31',
        '',
    ])

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return os.path.normpath(out_path)


def _yydoy(calendar_year: int, month: int, day: int) -> int:
    max_day = calendar.monthrange(calendar_year, month)[1]
    safe_day = min(day, max_day)
    doy = date(calendar_year, month, safe_day).timetuple().tm_yday
    return (calendar_year % 100) * 1000 + doy


def write_measured_wth(
    records: List[dict],
    *,
    source_year: int,
    out_path: str,
    spec: CityClimateSpec,
    calendar_year: int = DSSAT_BENCHMARK_YEAR,
) -> str:
    """Write DSSAT measured daily weather for one NASA year on the benchmark calendar."""
    days = sorted(
        (r for r in records if r['y'] == source_year),
        key=lambda r: (r['m'], r['d']),
    )
    if len(days) < 360:
        raise RuntimeError(f'Only {len(days)} days for NASA year {source_year}')

    tav, amp = compute_tav_amp(days)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    lines = [
        f'*WEATHER DATA : {spec.station}',
        '',
        '@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT',
        (
            f'  {spec.station}   {spec.lat:7.3f}  {spec.lon:7.3f}  {spec.elev:4d}'
            f'  {tav:5.1f}  {amp:5.1f}   2.0   2.0'
        ),
        f'@DATE  SRAD  TMAX  TMIN  RAIN',
    ]
    seen_dates: set[int] = set()
    for r in days:
        yydoy = _yydoy(calendar_year, r['m'], r['d'])
        if yydoy in seen_dates:
            continue
        seen_dates.add(yydoy)
        lines.append(
            f"{yydoy:5d} {r['srad']:5.1f} {r['tmax']:5.1f} {r['tmin']:5.1f} {r['rain']:5.1f}"
        )

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return os.path.normpath(out_path)


def build_city_wth_library(
    spec: CityClimateSpec,
    out_root: str,
    records: Optional[List[dict]] = None,
) -> dict:
    """One measured .WTH per NASA year — reliable alternative to WGEN for city demos."""
    records = records or download_nasa_power(spec.lat, spec.lon, spec.start_year, spec.end_year)
    written = []
    for year in range(spec.start_year, spec.end_year + 1):
        year_dir = os.path.join(out_root, str(year))
        out_path = os.path.join(year_dir, DSSAT_BENCHMARK_WTH)
        write_measured_wth(records, source_year=year, out_path=out_path, spec=spec)
        written.append(year)
    return {
        'wth_root': os.path.normpath(out_root),
        'years': written,
        'days_per_year': len(written),
        'wth_name': DSSAT_BENCHMARK_WTH,
    }


def build_city_cli(spec: CityClimateSpec, out_path: str) -> dict:
    records = download_nasa_power(spec.lat, spec.lon, spec.start_year, spec.end_year)
    tav, amp = compute_tav_amp(records)
    monthly_avgs, wgen_params = compute_wgen(records, spec.start_year, spec.end_year)
    wgen_params = finalize_wgen_params(wgen_params)
    path = write_cli_file(
        out_path=out_path,
        spec=spec,
        monthly_avgs=monthly_avgs,
        wgen_params=wgen_params,
        tav=tav,
        amp=amp,
    )
    annual_rain = round(sum(ma['rtot'] for ma in monthly_avgs), 0)
    return {
        'path': path,
        'city_id': spec.city_id,
        'name': spec.name,
        'lat': spec.lat,
        'lon': spec.lon,
        'period': f'{spec.start_year}–{spec.end_year}',
        'tav_c': tav,
        'amp_c': amp,
        'annual_rain_mm': annual_rain,
        'days': len(records),
    }
