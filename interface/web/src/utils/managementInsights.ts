import type { DayRecord } from '@/types';
import { filterCropActiveDays, isCropActiveDay } from '@/utils/seasonDays';

export interface WeeklyRollup {
  startDay: number;
  endDay: number;
  dayCount: number;
  rain_mm: number;
  irrigation_mm: number;
  nitrogen_kg_ha: number;
  avgSwfac: number;
  avgNstres: number;
  managementDays: number;
}

export interface ManagementHint {
  kind: 'irrigation' | 'nitrogen' | 'info';
  source: 'known' | 'heuristic';
  seasonDay?: number;
  dap?: number;
  amount?: number;
  message: string;
}

const MGMT_THRESH = 0.5;

export function weeklySummary(
  days: DayRecord[],
  focusDay: number,
  plantingSimDay: number,
): WeeklyRollup | null {
  const cropDays = days.filter(
    (d) => d.day <= focusDay && isCropActiveDay(d, plantingSimDay),
  );
  if (!cropDays.length) return null;

  const window = cropDays.slice(Math.max(0, cropDays.length - 7));
  const swfacSum = window.reduce((s, d) => s + (d.swfac_before ?? d.swfac), 0);
  const nstresSum = window.reduce((s, d) => s + (d.nstres_before ?? d.nstres ?? 1), 0);
  const mgmtDays = window.filter(
    (d) => d.irrigation_mm > MGMT_THRESH || d.nitrogen_kg_ha > MGMT_THRESH,
  ).length;

  return {
    startDay: window[0].day,
    endDay: window[window.length - 1].day,
    dayCount: window.length,
    rain_mm: window.reduce((s, d) => s + d.rain_mm, 0),
    irrigation_mm: window.reduce((s, d) => s + d.irrigation_mm, 0),
    nitrogen_kg_ha: window.reduce((s, d) => s + d.nitrogen_kg_ha, 0),
    avgSwfac: swfacSum / window.length,
    avgNstres: nstresSum / window.length,
    managementDays: mgmtDays,
  };
}

export function managementOutlook(
  days: DayRecord[],
  focusDay: DayRecord,
  plantingSimDay: number,
  seasonComplete: boolean,
): ManagementHint[] {
  const hints: ManagementHint[] = [];
  const after = days.filter((d) => d.day > focusDay.day);

  const nextIrrig = after.find((d) => d.irrigation_mm > MGMT_THRESH);
  if (nextIrrig) {
    hints.push({
      kind: 'irrigation',
      source: 'known',
      seasonDay: nextIrrig.day,
      dap: nextIrrig.dap,
      amount: nextIrrig.irrigation_mm,
      message: `Next irrigation likely ~${nextIrrig.irrigation_mm.toFixed(0)} mm on season day ${nextIrrig.day} (DAP ${nextIrrig.dap}).`,
    });
  }

  const nextN = after.find((d) => d.nitrogen_kg_ha > MGMT_THRESH);
  if (nextN) {
    hints.push({
      kind: 'nitrogen',
      source: 'known',
      seasonDay: nextN.day,
      dap: nextN.dap,
      amount: nextN.nitrogen_kg_ha,
      message: `Next nitrogen likely ~${nextN.nitrogen_kg_ha.toFixed(0)} kg/ha on season day ${nextN.day} (DAP ${nextN.dap}).`,
    });
  }

  if (hints.length || seasonComplete) return hints;

  const swfac = focusDay.swfac_before ?? focusDay.swfac;
  const nstres = focusDay.nstres_before ?? focusDay.nstres ?? 1;

  if (swfac < 0.85) {
    hints.push({
      kind: 'info',
      source: 'heuristic',
      message:
        'Soil is dry or plants are water-stressed — irrigation may appear in the next few days if conditions stay dry.',
    });
  } else if (nstres < 0.85 && focusDay.dap >= 18) {
    hints.push({
      kind: 'info',
      source: 'heuristic',
      message:
        'Nitrogen stress is rising — a fertilizer pulse may occur in the next few days.',
    });
  } else {
    const cropPast = filterCropActiveDays(
      days.filter((d) => d.day <= focusDay.day),
      plantingSimDay,
    );
    const mgmt = cropPast.filter(
      (d) => d.irrigation_mm > MGMT_THRESH || d.nitrogen_kg_ha > MGMT_THRESH,
    );
    if (mgmt.length >= 2) {
      const gaps: number[] = [];
      for (let i = 1; i < mgmt.length; i += 1) {
        gaps.push(mgmt[i].day - mgmt[i - 1].day);
      }
      const avgGap = Math.round(gaps.reduce((a, b) => a + b, 0) / gaps.length);
      const estDay = mgmt[mgmt.length - 1].day + avgGap;
      const estDap = focusDay.dap + (estDay - focusDay.day);
      hints.push({
        kind: 'info',
        source: 'heuristic',
        seasonDay: estDay,
        dap: estDap,
        message: `Based on past input pulses (~every ${avgGap} days), next application often around season day ${estDay} (DAP ~${estDap}).`,
      });
    } else {
      hints.push({
        kind: 'info',
        source: 'heuristic',
        message:
          'CAPQL applies water and N in pulses — most days show 0 mm / 0 kg/ha. Use weekly summary for the bigger picture.',
      });
    }
  }

  return hints;
}
