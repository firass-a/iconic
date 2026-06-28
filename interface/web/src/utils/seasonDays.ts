import type { DayRecord, SeasonResult } from '@/types';

/** Fixed planting date for DSSAT UFGA maize benchmark (display only). */
export const PLANTING_DATE = '2024-04-15';

/** Typical full maize season length in this DSSAT setup (~165 simulation steps). */
export const EXPECTED_SEASON_LENGTH = 165;

/**
 * Season day = simulation step counter from season start (1, 2, 3…).
 * DAP = days after planting (crop age). Early season many steps are still DAP 0 (pre-plant).
 */
export function isSeasonComplete(
  result: SeasonResult,
  options?: { advisoryDone?: boolean },
): boolean {
  if (options?.advisoryDone === true) return true;
  if (options?.advisoryDone === false) return false;
  if (result.summary.yield_kg_ha > 500 && result.ep_length >= 140) return true;
  const last = result.days[result.days.length - 1];
  return Boolean(last && result.ep_length >= 150 && last.grnwt > 500);
}

export function seasonProgressPct(result: SeasonResult): number {
  return Math.min(100, Math.round((result.ep_length / EXPECTED_SEASON_LENGTH) * 100));
}

/** Simulation step index where planting occurs (DAP 0 on this day). */
export function findPlantingSimDay(days: DayRecord[]): number {
  if (!days.length) return 1;
  const firstDapOne = days.find((d) => d.dap === 1);
  if (firstDapOne) return firstDapOne.day - 1;
  const firstPositive = days.find((d) => d.dap > 0);
  if (firstPositive) return firstPositive.day;
  return days[0].day;
}

/** Days when the crop is in the ground (hide bare pre-plant soil prep in browse UI). */
export function isCropActiveDay(day: DayRecord, plantingSimDay: number): boolean {
  return day.dap >= 1 || day.day >= plantingSimDay;
}

export function filterCropActiveDays(days: DayRecord[], plantingSimDay: number): DayRecord[] {
  return days.filter((d) => isCropActiveDay(d, plantingSimDay));
}

export function cropActiveBounds(
  days: DayRecord[],
  plantingSimDay: number,
): { min: number; max: number } {
  const browsable = filterCropActiveDays(days, plantingSimDay);
  if (!browsable.length) return simDayBounds(days);
  return { min: browsable[0].day, max: browsable[browsable.length - 1].day };
}

export function stepSimDayCropActive(
  days: DayRecord[],
  simDay: number,
  delta: number,
  plantingSimDay: number,
): number {
  const browsable = filterCropActiveDays(days, plantingSimDay);
  if (!browsable.length) return stepSimDay(days, simDay, delta);
  const idx = browsable.findIndex((d) => d.day === simDay);
  const baseIdx = idx >= 0 ? idx : browsable.length - 1;
  return browsable[Math.min(browsable.length - 1, Math.max(0, baseIdx + delta))].day;
}

export function formatPlantingDate(plantingIso = PLANTING_DATE): string {
  return new Date(`${plantingIso}T12:00:00`).toLocaleDateString(undefined, {
    weekday: 'short',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}

/**
 * Map simulation step to a calendar date, anchored so plantingSimDay = planting date.
 */
export function simDayToCalendarDate(
  simDay: number,
  plantingIso = PLANTING_DATE,
  plantingSimDay = 1,
): Date {
  const d = new Date(`${plantingIso}T12:00:00`);
  d.setDate(d.getDate() + simDay - plantingSimDay);
  return d;
}

export function formatSimDayDate(
  simDay: number,
  plantingIso = PLANTING_DATE,
  plantingSimDay = 1,
): string {
  return simDayToCalendarDate(simDay, plantingIso, plantingSimDay).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  });
}

/** @deprecated Use formatSimDayDate for timeline selection; DAP alone duplicates on calendar. */
export function formatCalendarDate(dap: number, plantingIso = PLANTING_DATE): string {
  const d = new Date(`${plantingIso}T12:00:00`);
  d.setDate(d.getDate() + dap);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export function formatSeasonDayLabel(day: Pick<DayRecord, 'day' | 'dap'>): string {
  return `Season day ${day.day} · DAP ${day.dap}`;
}

export function findDayBySimDay(days: DayRecord[], simDay: number): DayRecord | undefined {
  return days.find((d) => d.day === simDay);
}

export function findSimDayIndex(days: DayRecord[], simDay: number): number {
  return days.findIndex((d) => d.day === simDay);
}

export function stepSimDay(days: DayRecord[], simDay: number, delta: number): number {
  const idx = findSimDayIndex(days, simDay);
  if (idx < 0) return simDay;
  return days[Math.min(days.length - 1, Math.max(0, idx + delta))].day;
}

export function getDaysInSimDayRange(
  days: DayRecord[],
  startDay: number,
  endDay: number,
): DayRecord[] {
  const lo = Math.min(startDay, endDay);
  const hi = Math.max(startDay, endDay);
  return days.filter((d) => d.day >= lo && d.day <= hi);
}

export function simDayBounds(days: DayRecord[]): { min: number; max: number } {
  if (!days.length) return { min: 1, max: 1 };
  return {
    min: days[0].day,
    max: days[days.length - 1].day,
  };
}

/** Default browse interval — values are simulation `day` indices (not agronomic DAP). */
export function defaultDapInterval(days: DayRecord[]): { start: number; end: number; focus: number } {
  if (!days.length) return { start: 1, end: 1, focus: 1 };

  const plantingSimDay = findPlantingSimDay(days);
  const pool = filterCropActiveDays(days, plantingSimDay);
  const scoped = pool.length ? pool : days;
  const { min, max } = simDayBounds(scoped);

  const byIrrig = [...scoped].sort((a, b) => b.irrigation_mm - a.irrigation_mm)[0];
  if (byIrrig.irrigation_mm > 0.1) {
    return {
      start: Math.max(min, byIrrig.day - 4),
      end: Math.min(max, byIrrig.day + 4),
      focus: byIrrig.day,
    };
  }

  const active = scoped.filter((d) => d.irrigation_mm > 0.5 || d.nitrogen_kg_ha > 0.5);
  if (!active.length) {
    const mid = scoped[Math.floor(scoped.length / 2)];
    return { start: mid.day, end: mid.day, focus: mid.day };
  }

  const peak = active.reduce((best, d) =>
    d.irrigation_mm + d.nitrogen_kg_ha > best.irrigation_mm + best.nitrogen_kg_ha ? d : best,
  );

  return {
    start: Math.max(min, peak.day - 5),
    end: Math.min(max, peak.day + 5),
    focus: peak.day,
  };
}

export function activityLevel(day: DayRecord): number {
  return Math.min(1, (day.irrigation_mm / 25 + day.nitrogen_kg_ha / 40) / 2);
}

export function rangeTotals(days: DayRecord[]) {
  return days.reduce(
    (acc, d) => ({
      irrigation_mm: acc.irrigation_mm + d.irrigation_mm,
      nitrogen_kg_ha: acc.nitrogen_kg_ha + d.nitrogen_kg_ha,
      rain_mm: acc.rain_mm + d.rain_mm,
    }),
    { irrigation_mm: 0, nitrogen_kg_ha: 0, rain_mm: 0 },
  );
}
