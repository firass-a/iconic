"""
Download NASA POWER daily weather data for Gainesville, FL (1984-2023) and
generate a calibrated UFGA.CLI file for gym-DSSAT's WGEN weather generator.

The generated UFGA.CLI replaces the Docker image's default climate file via
auxiliary_file_paths, so WGEN produces statistically realistic Gainesville
weather instead of generic synthetic data.

Run once inside Docker before training:
    cd /workspace/gym-dssat-pdi/gym_dssat_pdi_samples
    /opt/gym_dssat_pdi/bin/python3 07_weather_setup.py

Output:
    /workspace/weather/UFGA.CLI
"""
import os
import json
import urllib.request
import numpy as np
from datetime import date, timedelta
from collections import defaultdict

# ================================================================== #
# Config                                                               #
# ================================================================== #
LAT        = 29.65      # Gainesville, FL
LON        = -82.33
ELEV       = 25         # meters
STATION    = 'UFGA'     # must match WSTA in DSSAT FileX
START_YEAR = 1984
END_YEAR   = 2023
OUT_DIR    = '/workspace/weather'
OUT_FILE   = f'{OUT_DIR}/UFGA.CLI'
WET_THRESH = 0.1        # mm — minimum rain to count as a "wet" day

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


# ================================================================== #
# Download NASA POWER                                                   #
# ================================================================== #
def download_nasa_power():
    print("Downloading NASA POWER data for Gainesville, FL (1984-2023)...")
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=ALLSKY_SFC_SW_DWN,T2M_MAX,T2M_MIN,PRECTOTCORR"
        f"&community=AG"
        f"&longitude={LON}"
        f"&latitude={LAT}"
        f"&start={START_YEAR}0101"
        f"&end={END_YEAR}1231"
        f"&format=JSON"
    )
    print(f"  URL: {url[:80]}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    props = data['properties']['parameter']
    srad  = props['ALLSKY_SFC_SW_DWN']  # MJ/m²/day
    tmax  = props['T2M_MAX']            # °C
    tmin  = props['T2M_MIN']            # °C
    rain  = props['PRECTOTCORR']        # mm/day

    # Parse into (year, month, day) records
    records = []
    for yyyymmdd, sr in srad.items():
        y, m, d = int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:])
        tx = tmax.get(yyyymmdd, -99)
        tn = tmin.get(yyyymmdd, -99)
        rn = rain.get(yyyymmdd, 0.0)
        # Skip fill values
        if any(v < -90 for v in [sr, tx, tn]):
            continue
        if rn < -90:
            rn = 0.0
        records.append({'y': y, 'm': m, 'd': d,
                        'srad': float(sr), 'tmax': float(tx),
                        'tmin': float(tn), 'rain': float(rn)})

    print(f"  Downloaded {len(records)} daily records.")
    return records


# ================================================================== #
# Compute WGEN parameters                                              #
# ================================================================== #
def compute_wgen(records):
    """
    Compute monthly WGEN parameters from daily records.
    Returns two lists of 12 dicts: monthly_avgs, wgen_params.
    """
    # Group by month
    by_month = defaultdict(list)
    for r in records:
        by_month[r['m']].append(r)

    monthly_avgs = []
    wgen_params  = []

    for m in range(1, 13):
        days = by_month[m]
        n_years = END_YEAR - START_YEAR + 1

        srad_all = np.array([d['srad'] for d in days])
        tmax_all = np.array([d['tmax'] for d in days])
        tmin_all = np.array([d['tmin'] for d in days])
        rain_all = np.array([d['rain'] for d in days])
        wet_mask = rain_all >= WET_THRESH
        dry_mask = ~wet_mask

        # ---- Monthly averages ----
        samn = float(np.mean(srad_all))
        xamn = float(np.mean(tmax_all))
        namn = float(np.mean(tmin_all))
        rtot = float(np.sum(rain_all)) / n_years
        rnum = float(np.sum(wet_mask)) / n_years

        monthly_avgs.append({
            'mth': m, 'samn': samn, 'xamn': xamn, 'namn': namn,
            'rtot': rtot, 'rnum': rnum,
        })

        # ---- WGEN parameters ----
        # Solar radiation on dry/wet days
        srad_dry = srad_all[dry_mask]
        srad_wet = srad_all[wet_mask]
        sdmn = float(np.mean(srad_dry)) if len(srad_dry) > 1 else samn
        sdsd = float(np.std(srad_dry))  if len(srad_dry) > 1 else 1.0
        swmn = float(np.mean(srad_wet)) if len(srad_wet) > 1 else samn * 0.6
        swsd = float(np.std(srad_wet))  if len(srad_wet) > 1 else 1.0

        # Max temperature on dry/wet days
        tmax_dry = tmax_all[dry_mask]
        tmax_wet = tmax_all[wet_mask]
        xdmn = float(np.mean(tmax_dry)) if len(tmax_dry) > 1 else xamn
        xdsd = float(np.std(tmax_dry))  if len(tmax_dry) > 1 else 2.0
        xwmn = float(np.mean(tmax_wet)) if len(tmax_wet) > 1 else xamn - 1.0
        xwsd = float(np.std(tmax_wet))  if len(tmax_wet) > 1 else 2.0

        # Min temperature (all days)
        nasd = float(np.std(tmin_all)) if len(tmin_all) > 1 else 2.0

        # Rainfall gamma distribution (wet days only)
        rain_wet = rain_all[wet_mask]
        if len(rain_wet) > 1:
            mu  = float(np.mean(rain_wet))
            var = float(np.var(rain_wet))
            alpha = (mu ** 2 / var) if var > 0 else 0.5
        else:
            alpha = 0.3

        # Markov chain: P(wet | previous dry) = PDW
        # Count D→W transitions within each year/month
        # Group records by year to compute transitions properly
        by_year = defaultdict(list)
        for d in days:
            by_year[d['y']].append(d)

        dw_count = 0
        dry_total = 0
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


# ================================================================== #
# Compute TAV and AMP                                                  #
# ================================================================== #
def compute_tav_amp(records):
    """Annual mean temperature and mean monthly amplitude."""
    by_month = defaultdict(list)
    for r in records:
        by_month[r['m']].append((r['tmax'] + r['tmin']) / 2.0)

    monthly_means = [np.mean(by_month[m]) for m in range(1, 13)]
    tav = float(np.mean(monthly_means))
    amp = float((max(monthly_means) - min(monthly_means)) / 2.0)
    return round(tav, 1), round(amp, 1)


# ================================================================== #
# Write UFGA.CLI                                                        #
# ================================================================== #
def write_cli(monthly_avgs, wgen_params, tav, amp):
    os.makedirs(OUT_DIR, exist_ok=True)

    lines = []
    lines.append(f'*CLIMATE:{STATION}              Gainesville, Florida, USA')
    lines.append('')
    lines.append(f'@ INSI      LAT     LONG  ELEV   TAV   AMP  SRAY  TMXY  TMNY  RAIY')

    sray = round(np.mean([ma['samn'] for ma in monthly_avgs]), 1)
    tmxy = round(np.mean([ma['xamn'] for ma in monthly_avgs]), 1)
    tmny = round(np.mean([ma['namn'] for ma in monthly_avgs]), 1)
    raiy = round(sum([ma['rtot'] for ma in monthly_avgs]), 0)

    lines.append(
        f'  {STATION}   {LAT:7.3f}  {LON:7.3f}  {ELEV:4d}'
        f'  {tav:5.1f}  {amp:5.1f}  {sray:4.1f}'
        f'  {tmxy:4.1f}  {tmny:4.1f}  {raiy:4.0f}'
    )
    lines.append(f'@START  DURN  ANGA  ANGB REFHT WNDHT SOURCE')
    lines.append(f'  {START_YEAR}    99  0.25  0.50   2.0   2.0 NASA_POWER_{START_YEAR}_{END_YEAR}')
    lines.append(f'@ GSST  GSDU')
    lines.append(f'     1   365')
    lines.append('')

    lines.append('*MONTHLY AVERAGES')
    lines.append('@  MTH  SAMN  XAMN  NAMN  RTOT  RNUM  SHMN  AMTH  BMTH')
    for ma in monthly_avgs:
        lines.append(
            f"  {ma['mth']:4d}  {ma['samn']:5.1f}  {ma['xamn']:5.1f}"
            f"  {ma['namn']:5.1f}  {ma['rtot']:6.1f}  {ma['rnum']:5.1f}"
            f"   -99 0.250 0.500"
        )
    lines.append('')

    lines.append('*WGEN PARAMETERS')
    lines.append('@  MTH  SDMN  SDSD  SWMN  SWSD  XDMN  XDSD  XWMN  XWSD  NAMN  NASD ALPHA  RTOT   PDW  RNUM')
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
    lines.append('')

    lines.append('*RANGE CHECK VALUES')
    lines.append('@      SRAD  TMAX  TMIN  RAIN  DEWP  WIND  SUNH   PAR  TDRY  TWET  EVAP  RHUM')
    lines.append('MIN :   0.5 -10.0 -10.0   0.0 -40.0   0.0   0.0   5.0 -10.0 -10.0   0.0   0.0')
    lines.append('MAX :  85.0  45.0  35.0 600.0  40.0 500.0 100.0  85.0  45.0  40.0  15.0 100.0')
    lines.append('RATE:  70.0  20.0  20.0 500.0   5.0 300.0  90.0  70.0  20.0  20.0  15.0  75.0')
    lines.append('')

    lines.append('*FLAGGED DATA COUNT')
    lines.append('@BEGYR BEGMN BEGDY ENDYR ENDMN ENDDY')
    lines.append(f'  {START_YEAR}     1     1  {END_YEAR}    12    31')

    with open(OUT_FILE, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"\n✓ Written → {OUT_FILE}")
    print(f"  Station : {STATION}  ({LAT}N, {LON}E)")
    print(f"  Period  : {START_YEAR}–{END_YEAR}  ({END_YEAR - START_YEAR + 1} years)")
    print(f"  TAV={tav}°C  AMP={amp}°C  SRAY={sray} MJ/m²/d  Annual rain={raiy:.0f} mm")


# ================================================================== #
# Main                                                                 #
# ================================================================== #
if __name__ == '__main__':
    print("=" * 60)
    print("NASA POWER → DSSAT UFGA.CLI generator")
    print("=" * 60)

    records = download_nasa_power()
    tav, amp = compute_tav_amp(records)
    monthly_avgs, wgen_params = compute_wgen(records)
    write_cli(monthly_avgs, wgen_params, tav, amp)

    print("\nNext step: run 04_pc_ppo_quick_train.py — it will pick up UFGA.CLI automatically.")
