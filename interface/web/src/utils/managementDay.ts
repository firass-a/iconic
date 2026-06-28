import type { DayRecord } from '@/types';
import { formatMoistureReading, formatNitrogenStatus, formatWaterStress } from '@/utils/formatAgronomic';

export type ApplicationLevel = 'none' | 'light' | 'moderate' | 'heavy';

export interface DayIndicator {
  id: string;
  label: string;
  value: string;
}

export interface ManagementDayAnalysis {
  irrigationLevel: ApplicationLevel;
  nitrogenLevel: ApplicationLevel;
  irrigationApplied: boolean;
  nitrogenApplied: boolean;
  irrigationNote: string;
  nitrogenNote: string;
  indicators: DayIndicator[];
  calendarTitle: string;
}

const IRRIG_THRESH = 0.1;
const N_THRESH = 0.1;

function irrigationLevel(mm: number): ApplicationLevel {
  if (mm < IRRIG_THRESH) return 'none';
  if (mm < 5) return 'light';
  if (mm < 15) return 'moderate';
  return 'heavy';
}

function nitrogenLevel(kg: number): ApplicationLevel {
  if (kg < N_THRESH) return 'none';
  if (kg < 5) return 'light';
  if (kg < 20) return 'moderate';
  return 'heavy';
}

function moistureLabel(
  moisture: number,
  isFc: boolean | undefined,
): string {
  return formatMoistureReading(moisture, isFc).label;
}

export function analyzeManagementDay(day: DayRecord): ManagementDayAnalysis {
  const irrig = day.irrigation_mm;
  const nkg = day.nitrogen_kg_ha;
  const rain = day.rain_mm;
  const swfac = day.swfac_before ?? day.swfac;
  const nstres = day.nstres_before ?? day.nstres ?? 1;
  const dap = day.dap;
  const moistBefore = day.soil_moisture_before ?? day.soil_moisture;
  const moistFcBefore = day.soil_moisture_fc_before ?? day.soil_moisture_fc;
  const xlai = day.xlai;

  const irrigationApplied = irrig >= IRRIG_THRESH;
  const nitrogenApplied = nkg >= N_THRESH;

  let irrigationNote = day.irrigation_note ?? '';
  let nitrogenNote = day.nitrogen_note ?? '';

  if (!irrigationNote) {
    if (irrigationApplied) {
      const because: string[] = [];
      if (swfac < 0.85) because.push(formatWaterStress(swfac).toLowerCase());
      if (moistFcBefore !== false && moistBefore < 0.55) {
        because.push(`soil at ${moistureLabel(moistBefore, moistFcBefore)} before watering`);
      } else if (moistFcBefore === false && swfac < 0.9) {
        because.push(`soil water stress factor ${swfac.toFixed(2)}`);
      }
      if (rain < 2 && dap > 0) because.push('little or no rain today');
      irrigationNote =
        because.length > 0
          ? `Applied ${irrig.toFixed(1)} mm: ${because.join('; ')}.`
          : `Applied ${irrig.toFixed(1)} mm to support crop water needs.`;
    } else {
      const why: string[] = [];
      if (rain >= 8) why.push(`${rain.toFixed(1)} mm rain — soil fed naturally`);
      else if (rain >= 3) why.push(`${rain.toFixed(1)} mm rain reduces irrigation need`);
      if (moistFcBefore !== false && moistBefore >= 0.72) {
        why.push(`soil already wet (${moistureLabel(moistBefore, moistFcBefore)})`);
      } else if (swfac >= 0.9) {
        why.push(`${formatWaterStress(swfac).toLowerCase()} — crop not water-limited`);
      }
      if (dap <= 0) why.push('pre‑plant / at planting — water usually held');
      if (dap > 0 && dap < 10 && xlai < 0.2) why.push('seedling stage — low transpiration');
      if (!why.length) why.push('policy held water to balance yield and water saving');
      irrigationNote = `0 mm irrigation: ${why.join('; ')}.`;
    }
  }

  if (!nitrogenNote) {
    if (nitrogenApplied) {
      const because: string[] = [];
      if (nstres < 0.85) because.push(formatNitrogenStatus(nstres).toLowerCase());
      if (dap >= 20) because.push(`mid‑season demand (DAP ${dap})`);
      if (xlai >= 1) because.push(`canopy LAI ${xlai.toFixed(1)} — active uptake`);
      nitrogenNote =
        because.length > 0
          ? `Applied ${nkg.toFixed(1)} kg/ha N: ${because.join('; ')}.`
          : `Applied ${nkg.toFixed(1)} kg/ha N for crop nitrogen demand.`;
    } else {
      const why: string[] = [];
      if (nstres >= 0.88) why.push(`${formatNitrogenStatus(nstres).toLowerCase()} — N supply adequate`);
      if (dap < 14) why.push(`early season (DAP ${dap}) — small N demand`);
      if (xlai < 0.4 && dap < 25) why.push(`small canopy (LAI ${xlai.toFixed(1)})`);
      if (dap <= 0) why.push('before crop establishment');
      if (!why.length) why.push('policy held N to protect efficiency and limit leaching risk');
      nitrogenNote = `0 kg/ha nitrogen: ${why.join('; ')}.`;
    }
  }

  const indicators: DayIndicator[] = [
    { id: 'dap', label: 'DAP', value: String(dap) },
    { id: 'rain', label: 'Rain', value: `${rain.toFixed(1)} mm` },
    {
      id: 'moisture',
      label: 'Soil moisture (before)',
      value: moistureLabel(moistBefore, moistFcBefore),
    },
    { id: 'swfac', label: 'Water stress', value: formatWaterStress(swfac) },
    { id: 'nstres', label: 'N status', value: formatNitrogenStatus(nstres) },
    { id: 'stage', label: 'Growth', value: `V${day.growth_stage.toFixed(1)}` },
  ];

  const titleParts: string[] = [];
  if (irrigationApplied) titleParts.push(`Water ${irrig.toFixed(0)} mm`);
  else titleParts.push('No irrigation');
  if (nitrogenApplied) titleParts.push(`N ${nkg.toFixed(0)} kg/ha`);
  else titleParts.push('No N');

  return {
    irrigationLevel: irrigationLevel(irrig),
    nitrogenLevel: nitrogenLevel(nkg),
    irrigationApplied,
    nitrogenApplied,
    irrigationNote,
    nitrogenNote,
    indicators,
    calendarTitle: titleParts.join(' · '),
  };
}

export function levelBarWidth(level: ApplicationLevel): string {
  switch (level) {
    case 'heavy':
      return 'w-full';
    case 'moderate':
      return 'w-2/3';
    case 'light':
      return 'w-1/3';
    default:
      return 'w-0';
  }
}
