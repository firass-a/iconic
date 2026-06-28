import type { DayRecord } from '@/types';
import { FieldHealthBars } from '@/components/FieldHealthBars';
import { MaizeStageIcon } from '@/components/MaizeStageIcon';
import { ProgressBar } from '@/components/ProgressBar';
import {
  formatDecisionMoisture,
  formatDecisionNstres,
  formatDecisionSwfac,
  formatGrowthStage,
  formatNstresReading,
  formatSoilWetnessStatus,
  formatSwfacReading,
  irrigationDriverNote,
  swfacBarColors,
} from '@/utils/formatAgronomic';

const MAX_WATER_MM = 50;
const MAX_N_KG = 200;

interface FieldSimViewProps {
  focusDay: DayRecord;
  dateLabel: string;
  compact?: boolean;
  hideHealthBars?: boolean;
}

function IndicatorTile({
  label,
  value,
  sub,
  tone = 'neutral',
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: 'good' | 'warn' | 'bad' | 'neutral';
}) {
  const toneClass = {
    good: 'border-emerald-100 bg-emerald-50/60',
    warn: 'border-amber-100 bg-amber-50/60',
    bad: 'border-rose-100 bg-rose-50/60',
    neutral: 'border-slate-100 bg-slate-50',
  }[tone];
  return (
    <div className={`rounded-xl border px-2.5 py-2 text-center ${toneClass}`}>
      <p className="text-[10px] font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-0.5 text-sm font-semibold text-slate-900">{value}</p>
      {sub && <p className="mt-0.5 text-[10px] leading-tight text-slate-500">{sub}</p>}
    </div>
  );
}

function swfacTone(swfac: number): 'good' | 'warn' | 'bad' {
  if (swfac >= 0.9) return 'good';
  if (swfac >= 0.7) return 'warn';
  return 'bad';
}

function nstresTone(nstres: number | undefined): 'good' | 'warn' | 'bad' {
  const v = nstres ?? 1;
  if (v >= 0.9) return 'good';
  if (v >= 0.7) return 'warn';
  return 'bad';
}

export function FieldSimView({
  focusDay,
  dateLabel,
  compact = false,
  hideHealthBars = false,
}: FieldSimViewProps) {
  const growth = formatGrowthStage(focusDay.growth_stage);
  const swfacDec = formatDecisionSwfac(focusDay);
  const swfac = formatSwfacReading(swfacDec.value);
  const swfacColors = swfacBarColors(swfacDec.value);
  const decisionMoist = formatDecisionMoisture(focusDay);
  const nstresDec = formatDecisionNstres(focusDay);
  const nstres = formatNstresReading(nstresDec.value);

  const irrigated = focusDay.irrigation_mm > 0.1;
  const postMoist =
    irrigated && decisionMoist.timing === 'before'
      ? formatDecisionMoisture({
          soil_moisture: focusDay.soil_moisture,
          soil_moisture_fc: focusDay.soil_moisture_fc,
        })
      : null;

  const driverNote = irrigationDriverNote(focusDay);

  return (
    <div className="dash-card space-y-5 p-5">
      <div className="flex items-start gap-4 border-b border-slate-100 pb-4">
        <MaizeStageIcon vstage={focusDay.growth_stage} className="h-14 w-14 shrink-0" />
        <div className="min-w-0 flex-1">
          <p className="text-sm text-slate-500">{dateLabel}</p>
          <p className="text-lg font-semibold text-slate-900">
            {growth.label}
            <span className="ml-2 text-base font-normal text-slate-500">· DAP {focusDay.dap}</span>
          </p>
          {!compact && (
            <p className="mt-0.5 text-sm text-slate-600">{growth.explanation}</p>
          )}
        </div>
      </div>

      {!hideHealthBars && <FieldHealthBars day={focusDay} compact />}

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <IndicatorTile
          label={decisionMoist.timing === 'before' ? 'Soil (before)' : 'Soil moisture'}
          value={
            decisionMoist.isFieldCapacity
              ? `${decisionMoist.pct}% FC`
              : decisionMoist.label
          }
          sub={decisionMoist.isFieldCapacity ? formatSoilWetnessStatus(decisionMoist.pct) : 'stress proxy'}
          tone={
            decisionMoist.isFieldCapacity && decisionMoist.pct < 40
              ? 'bad'
              : decisionMoist.isFieldCapacity && decisionMoist.pct < 55
                ? 'warn'
                : 'neutral'
          }
        />
        <IndicatorTile
          label="SWFAC (plant)"
          value={swfac.short}
          sub={
            swfacDec.timing === 'before'
              ? swfac.status
              : swfacDec.timing === 'estimated'
                ? 'estimated'
                : `${swfac.status} · after`
          }
          tone={swfacTone(swfacDec.value)}
        />
        <IndicatorTile
          label="N supply"
          value={nstres.short}
          sub={nstres.label}
          tone={nstresTone(nstresDec.value)}
        />
        <IndicatorTile
          label="Rain today"
          value={`${focusDay.rain_mm.toFixed(1)} mm`}
          sub={`${focusDay.tmax_c.toFixed(0)}°C max`}
          tone="neutral"
        />
      </div>

      <div className="space-y-4">
        <ProgressBar
          label="Water today"
          value={focusDay.irrigation_mm}
          max={MAX_WATER_MM}
          unit="mm"
          colorClass="bg-sky-500"
          trackClass="bg-sky-100"
        />
        <ProgressBar
          label="Fertilizer today"
          value={focusDay.nitrogen_kg_ha}
          max={MAX_N_KG}
          unit="kg/ha"
          colorClass="bg-violet-500"
          trackClass="bg-violet-100"
        />
        <ProgressBar
          label="Crop water supply (SWFAC)"
          value={swfac.pct}
          max={100}
          unit="· 1.0 = no stress"
          colorClass={swfacColors.fill}
          trackClass={swfacColors.track}
        />
        {decisionMoist.isFieldCapacity && (
          <ProgressBar
            label={
              decisionMoist.timing === 'before'
                ? 'Soil wetness before decision'
                : 'Soil moisture'
            }
            value={decisionMoist.pct}
            max={100}
            unit="% FC"
            colorClass="bg-teal-600"
            trackClass="bg-teal-100"
          />
        )}
      </div>

      {driverNote && (
        <p className="rounded-lg border border-sky-100 bg-sky-50 px-3 py-2 text-xs leading-relaxed text-sky-900">
          {driverNote}
        </p>
      )}

      {irrigated && postMoist && postMoist.pct > decisionMoist.pct + 5 && !driverNote && (
        <p className="rounded-lg bg-sky-50 px-3 py-2 text-xs leading-relaxed text-sky-900">
          Irrigation applied when SWFAC was {swfac.short} and soil was {decisionMoist.pct}% FC
          {focusDay.swfac !== undefined && focusDay.swfac > swfacDec.value + 0.05
            ? ` — SWFAC rose to ${focusDay.swfac.toFixed(2)} after watering`
            : ''}
          {postMoist.pct > decisionMoist.pct + 5 ? `, moisture to ${postMoist.pct}% FC` : ''}.
        </p>
      )}

      <div className="grid grid-cols-2 gap-2 text-center text-xs sm:grid-cols-3">
        <div className="rounded-lg bg-slate-50 px-2 py-2">
          <p className="text-slate-500">Total irrig</p>
          <p className="mt-0.5 font-semibold text-slate-900">
            {(focusDay.cumulative_irrigation_mm ?? 0).toFixed(0)} mm
          </p>
        </div>
        <div className="rounded-lg bg-slate-50 px-2 py-2">
          <p className="text-slate-500">LAI</p>
          <p className="mt-0.5 font-semibold text-slate-900">{focusDay.xlai.toFixed(1)}</p>
        </div>
        <div className="rounded-lg bg-slate-50 px-2 py-2 sm:col-span-1 col-span-2">
          <p className="text-slate-500">Solar rad</p>
          <p className="mt-0.5 font-semibold text-slate-900">{focusDay.srad.toFixed(0)} MJ/m²</p>
        </div>
      </div>
    </div>
  );
}
