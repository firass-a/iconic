import type { DayRecord } from '@/types';

/** Format soil moisture — FC fraction vs water-stress fallback. */
export function formatSoilMoisture(day: Pick<DayRecord, 'soil_moisture' | 'soil_moisture_fc'>) {
  return formatMoistureReading(day.soil_moisture, day.soil_moisture_fc);
}

export function formatMoistureReading(
  moisture: number,
  isFieldCapacity?: boolean,
): { label: string; pct: number; value: string; unit: string } {
  if (isFieldCapacity !== false) {
    const pct = Math.round(Math.min(1, Math.max(0, moisture)) * 100);
    return {
      label: `${pct}% of field capacity`,
      pct,
      value: String(pct),
      unit: '% field capacity',
    };
  }
  const value = moisture.toFixed(2);
  return {
    label: `water stress ${value}`,
    pct: Math.round(moisture * 100),
    value,
    unit: 'water stress (1 = no stress)',
  };
}

export function formatNitrogenStatus(nstres: number | undefined): string {
  const n = nstres ?? 1;
  if (n >= 0.9) return 'Low N stress';
  if (n >= 0.7) return 'Moderate N stress';
  return 'High N stress';
}

export function formatWaterStress(swfac: number): string {
  if (swfac >= 0.9) return 'Low water stress';
  if (swfac >= 0.7) return 'Moderate stress';
  return 'High water stress';
}

/** Short label for KPI tiles. */
export function formatWaterStressShort(swfac: number): string {
  if (swfac >= 0.9) return 'Low';
  if (swfac >= 0.7) return 'Moderate';
  return 'High';
}

/** SWFAC bar colors — higher supply is healthier. */
export function swfacBarColors(swfac: number): { fill: string; track: string } {
  if (swfac >= 0.9) return { fill: 'bg-emerald-500', track: 'bg-emerald-100' };
  if (swfac >= 0.7) return { fill: 'bg-amber-500', track: 'bg-amber-100' };
  return { fill: 'bg-rose-500', track: 'bg-rose-100' };
}

export function formatSwfacReading(swfac: number): {
  label: string;
  pct: number;
  short: string;
  status: string;
} {
  const v = Math.min(1, Math.max(0, swfac));
  const pct = Math.round(v * 100);
  return {
    label: `SWFAC ${v.toFixed(3)} — ${formatWaterStress(v).toLowerCase()}`,
    pct,
    short: v.toFixed(3),
    status: formatWaterStress(v),
  };
}

/** Farmer-facing soil wetness from % field capacity (before decision). */
export function formatSoilWetnessStatus(pct: number): string {
  if (pct >= 85) return 'Well supplied';
  if (pct >= 70) return 'Adequate';
  if (pct >= 55) return 'Getting dry';
  return 'Dry — irrigation likely';
}

/** Why CAPQL may irrigate even when SWFAC ≈ 1. */
export function irrigationDriverNote(
  day: Pick<
    DayRecord,
    | 'irrigation_mm'
    | 'swfac_before'
    | 'swfac'
    | 'soil_moisture'
    | 'soil_moisture_fc'
    | 'soil_moisture_before'
    | 'soil_moisture_fc_before'
  >,
): string | null {
  if (day.irrigation_mm < 0.1) return null;
  const swfac = day.swfac_before ?? day.swfac;
  const moist = formatDecisionMoisture(day);
  if (!moist.isFieldCapacity) return null;
  if (swfac >= 0.95 && moist.pct < 85) {
    return `Irrigation is driven mainly by soil moisture (${moist.pct}% FC before watering). Plant water supply (SWFAC ${swfac.toFixed(2)}) is still high — CAPQL irrigates before the crop shows stress.`;
  }
  if (swfac >= 0.95 && moist.pct >= 85) {
    return `Soil was already wet (${moist.pct}% FC) and the crop shows no water stress (SWFAC ${swfac.toFixed(2)}). CAPQL still applied water — this is a learned mid-season irrigation habit from training, not a simple “if wet, skip” rule. Extra water may run off in DSSAT.`;
  }
  if (swfac < 0.9) {
    return `Irrigation responds to plant water stress (SWFAC ${swfac.toFixed(2)}) and soil at ${moist.pct}% FC.`;
  }
  return null;
}

/** SWFAC / N-stress at decision time (before today's irrigation/N). */
export function formatDecisionSwfac(
  day: Pick<DayRecord, 'swfac' | 'swfac_before' | 'irrigation_mm' | 'soil_moisture_before' | 'soil_moisture_fc_before'>,
): { value: number; timing: 'before' | 'after' | 'estimated'; label: string } {
  if (day.swfac_before !== undefined) {
    const v = day.swfac_before;
    return { value: v, timing: 'before', label: `SWFAC ${v.toFixed(2)} (before decision)` };
  }
  // Legacy replays stored post-irrigation SWFAC (~always 1.0) — rough proxy from moisture
  if (day.irrigation_mm > 0.1 && day.swfac >= 0.99 && day.soil_moisture_before !== undefined) {
    const isFc = day.soil_moisture_fc_before !== false;
    if (isFc) {
      const moist = Math.min(1, Math.max(0, day.soil_moisture_before));
      const est = Math.min(0.95, Math.max(0.45, 0.35 + moist * 0.65));
      return {
        value: est,
        timing: 'estimated',
        label: `~${est.toFixed(2)} (estimated from ${Math.round(moist * 100)}% FC)`,
      };
    }
  }
  return { value: day.swfac, timing: 'after', label: `SWFAC ${day.swfac.toFixed(2)} (after step)` };
}

export function formatDecisionNstres(
  day: Pick<DayRecord, 'nstres' | 'nstres_before'>,
): { value: number; timing: 'before' | 'after' } {
  if (day.nstres_before !== undefined) {
    return { value: day.nstres_before, timing: 'before' };
  }
  return { value: day.nstres ?? 1, timing: 'after' };
}

export function formatDecisionMoisture(
  day: Pick<
    DayRecord,
    'soil_moisture' | 'soil_moisture_fc' | 'soil_moisture_before' | 'soil_moisture_fc_before'
  >,
): { label: string; pct: number; isFieldCapacity: boolean; timing: 'before' | 'after' } {
  const hasBefore = day.soil_moisture_before !== undefined;
  const moisture = hasBefore ? day.soil_moisture_before! : day.soil_moisture;
  const isFc = hasBefore
    ? day.soil_moisture_fc_before ?? day.soil_moisture_fc
    : day.soil_moisture_fc;
  const reading = formatMoistureReading(moisture, isFc);
  return {
    ...reading,
    isFieldCapacity: isFc !== false,
    timing: hasBefore ? 'before' : 'after',
  };
}

export function formatNstresReading(nstres: number | undefined): {
  label: string;
  pct: number;
  short: string;
} {
  const v = Math.min(1, Math.max(0, nstres ?? 1));
  const pct = Math.round(v * 100);
  return {
    label: formatNitrogenStatus(v),
    pct,
    short: v.toFixed(2),
  };
}

export function nstresBarColors(nstres: number | undefined): { fill: string; track: string } {
  const v = nstres ?? 1;
  if (v >= 0.9) return { fill: 'bg-emerald-500', track: 'bg-emerald-100' };
  if (v >= 0.7) return { fill: 'bg-amber-500', track: 'bg-amber-100' };
  return { fill: 'bg-rose-500', track: 'bg-rose-100' };
}

/** Maize V-stage (DSSAT vstage): number of leaf collars on the main stem. */
export function formatGrowthStage(vstage: number): {
  label: string;
  explanation: string;
  tooltip: string;
} {
  const v = Math.max(0, vstage);
  const collars = Math.floor(v);
  const label = `V${v.toFixed(1)}`;

  let explanation: string;
  if (v < 1) {
    explanation = 'Emergence — seedling leaves unfolding';
  } else if (v < 4) {
    explanation = `${collars} leaf collar${collars === 1 ? '' : 's'} — early vegetative`;
  } else if (v < 7) {
    explanation = `${collars} leaf collars — mid vegetative, rapid growth`;
  } else if (v < 11) {
    explanation = `${collars} leaf collars — late vegetative, canopy developing`;
  } else if (v < 15) {
    explanation = `${collars} leaf collars — pre-tassel, nearing flowering`;
  } else {
    explanation = `${collars} leaf collars — late vegetative, near tassel (VT)`;
  }

  const tooltip =
    'V-stage is the maize vegetative scale: each unit is one fully emerged leaf collar on the main stem (V1 = first collar, V6 = sixth, etc.). Reproductive stages (R) follow tasseling.';

  return { label, explanation, tooltip };
}

export interface HealthBarReading {
  pct: number;
  status: string;
  hint: string;
  fill: string;
  track: string;
}

function healthBarColors(pct: number): { fill: string; track: string } {
  if (pct >= 85) return { fill: 'bg-emerald-500', track: 'bg-emerald-100' };
  if (pct >= 65) return { fill: 'bg-amber-500', track: 'bg-amber-100' };
  return { fill: 'bg-rose-500', track: 'bg-rose-100' };
}

function cropHealthStatus(pct: number): string {
  if (pct >= 90) return 'Healthy';
  if (pct >= 75) return 'Good';
  if (pct >= 60) return 'Fair';
  return 'Stressed';
}

/** Plant water + N supply before today's inputs (SWFAC & nstres). */
export function computeCropHealth(
  day: Pick<DayRecord, 'swfac' | 'swfac_before' | 'nstres' | 'nstres_before' | 'irrigation_mm' | 'soil_moisture_before' | 'soil_moisture_fc_before'>,
): HealthBarReading {
  const swfac = formatDecisionSwfac(day).value;
  const nstres = formatDecisionNstres(day).value;
  const score = (swfac + nstres) / 2;
  const pct = Math.round(Math.min(1, Math.max(0, score)) * 100);
  const colors = healthBarColors(pct);
  return {
    pct,
    status: cropHealthStatus(pct),
    hint: `Water supply ${swfac.toFixed(2)} · N supply ${nstres.toFixed(2)} (before decision)`,
    ...colors,
  };
}

/** Root-zone wetness before today's inputs (% field capacity when available). */
export function computeFieldConditions(
  day: Pick<
    DayRecord,
    | 'soil_moisture'
    | 'soil_moisture_fc'
    | 'soil_moisture_before'
    | 'soil_moisture_fc_before'
    | 'swfac'
    | 'swfac_before'
    | 'rain_mm'
    | 'irrigation_mm'
  >,
): HealthBarReading {
  const moist = formatDecisionMoisture(day);
  const colors = healthBarColors(moist.pct);

  if (moist.isFieldCapacity) {
    return {
      pct: moist.pct,
      status: formatSoilWetnessStatus(moist.pct),
      hint:
        moist.timing === 'before'
          ? `Soil wetness before irrigation · ${day.rain_mm.toFixed(1)} mm rain today`
          : `Soil at ${moist.pct}% field capacity`,
      ...colors,
    };
  }

  const swfac = formatDecisionSwfac(day).value;
  const pct = Math.round(swfac * 100);
  return {
    pct,
    status: formatWaterStress(swfac),
    hint: 'Soil layer data unavailable — using plant water supply as proxy',
    ...healthBarColors(pct),
  };
}
