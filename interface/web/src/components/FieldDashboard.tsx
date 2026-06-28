import { Droplets, FlaskConical, Leaf, Sprout } from 'lucide-react';
import type { DayRecord, SeasonResult } from '@/types';
import { DayAgeHelp } from '@/components/DayAgeHelp';
import { FieldHealthBars } from '@/components/FieldHealthBars';
import { FieldSimView } from '@/components/FieldSimView';
import { KpiCard } from '@/components/KpiCard';
import { ManagementTasks } from '@/components/ManagementTasks';
import { MoistureChart } from '@/components/MoistureChart';
import { ProgressBar } from '@/components/ProgressBar';
import { SeasonStatusBanner } from '@/components/SeasonStatusBanner';
import { WeatherWidget } from '@/components/WeatherWidget';
import {
  formatDecisionMoisture,
  formatDecisionSwfac,
  formatGrowthStage,
  formatSoilWetnessStatus,
  formatSwfacReading,
} from '@/utils/formatAgronomic';
import { findSimDayIndex, formatSimDayDate, isSeasonComplete } from '@/utils/seasonDays';

interface FieldDashboardProps {
  result: SeasonResult;
  focusDay: DayRecord;
  plantingDate?: string;
  plantingSimDay?: number;
  showSeasonTotals?: boolean;
  advisoryDone?: boolean;
  advisoryError?: string | null;
  atLastDay?: boolean;
}

export function FieldDashboard({
  result,
  focusDay,
  plantingDate,
  plantingSimDay = 1,
  showSeasonTotals = false,
  advisoryDone,
  advisoryError,
  atLastDay,
}: FieldDashboardProps) {
  const dateLabel = formatSimDayDate(focusDay.day, plantingDate, plantingSimDay);
  const focusIdx = findSimDayIndex(result.days, focusDay.day);
  const prev = focusIdx > 0 ? result.days[focusIdx - 1] : null;
  const swfacDec = formatDecisionSwfac(focusDay);
  const swfac = formatSwfacReading(swfacDec.value);
  const decisionMoist = formatDecisionMoisture(focusDay);
  const growth = formatGrowthStage(focusDay.growth_stage);
  const seasonComplete = isSeasonComplete(result, { advisoryDone });

  const chartDays = result.days.slice(Math.max(0, focusIdx - 6), focusIdx + 1);
  const upcomingWeather = result.days.slice(focusIdx + 1, focusIdx + 5);

  const waterTrend =
    prev && prev.irrigation_mm !== focusDay.irrigation_mm
      ? {
          value: `${Math.abs(focusDay.irrigation_mm - prev.irrigation_mm).toFixed(0)} mm vs yesterday`,
          up: focusDay.irrigation_mm > prev.irrigation_mm,
        }
      : undefined;

  const moistHint = decisionMoist.isFieldCapacity
    ? `${formatSoilWetnessStatus(decisionMoist.pct)} · SWFAC ${swfac.short} (plant)`
    : `SWFAC ${swfac.short} · plant water supply`;

  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KpiCard
          label="Water today"
          value={focusDay.irrigation_mm.toFixed(0)}
          unit="mm"
          trend={waterTrend}
          hint={!waterTrend && focusDay.irrigation_mm < 0.1 ? 'No irrigation' : undefined}
          icon={<Droplets className="h-5 w-5" />}
          iconTone="sky"
        />
        <KpiCard
          label="Fertilizer today"
          value={focusDay.nitrogen_kg_ha < 0.1 ? '—' : focusDay.nitrogen_kg_ha.toFixed(1)}
          unit={focusDay.nitrogen_kg_ha >= 0.1 ? 'kg/ha' : undefined}
          hint={
            focusDay.nitrogen_kg_ha < 0.1
              ? `No application today · season total ${result.summary.total_n_kg_ha.toFixed(0)} kg/ha`
              : undefined
          }
          icon={<FlaskConical className="h-5 w-5" />}
          iconTone="violet"
        />
        <KpiCard
          label="Soil wetness (before)"
          value={decisionMoist.isFieldCapacity ? decisionMoist.pct : '—'}
          unit={decisionMoist.isFieldCapacity ? '% FC' : undefined}
          hint={moistHint}
          icon={<Leaf className="h-5 w-5" />}
          iconTone={
            decisionMoist.pct >= 70 ? 'emerald' : decisionMoist.pct >= 55 ? 'amber' : 'violet'
          }
        />
        <KpiCard
          label="Crop stage"
          value={growth.label}
          hint={`Season day ${focusDay.day} · DAP ${focusDay.dap} (crop age)`}
          icon={<Sprout className="h-5 w-5" />}
          iconTone="amber"
        />
      </div>

      <FieldHealthBars day={focusDay} />

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <MoistureChart days={chartDays} />
        </div>
        <WeatherWidget
          current={focusDay}
          upcoming={upcomingWeather}
          dateLabel={dateLabel}
          plantingDate={plantingDate}
          plantingSimDay={plantingSimDay}
        />
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <FieldSimView focusDay={focusDay} dateLabel={dateLabel} compact hideHealthBars />
        </div>
        <ManagementTasks
          days={result.days}
          focusDay={focusDay.day}
          plantingDate={plantingDate}
          plantingSimDay={plantingSimDay}
        />
      </div>

      {showSeasonTotals && (
        <>
          <SeasonStatusBanner
            result={result}
            advisoryDone={advisoryDone}
            advisoryError={advisoryError}
            atLastDay={atLastDay}
          />
          <DayAgeHelp compact />
          <div className="dash-card p-5">
            <p className="mb-4 text-sm font-semibold text-slate-800">Season totals</p>
            <div className="space-y-4">
              <ProgressBar
                label="Total irrigation"
                value={result.summary.total_water_mm}
                max={Math.max(result.summary.total_water_mm, 500)}
                unit="mm"
                colorClass="bg-sky-500"
                trackClass="bg-sky-100"
              />
              <ProgressBar
                label="Total nitrogen"
                value={result.summary.total_n_kg_ha}
                max={Math.max(result.summary.total_n_kg_ha, 300)}
                unit="kg/ha"
                colorClass="bg-violet-500"
                trackClass="bg-violet-100"
              />
              {seasonComplete ? (
                <ProgressBar
                  label="Grain yield (harvest)"
                  value={result.summary.yield_kg_ha}
                  max={Math.max(result.summary.yield_kg_ha, 12000)}
                  unit="kg/ha"
                  colorClass="bg-amber-600"
                  trackClass="bg-amber-100"
                />
              ) : (
                <div className="rounded-lg bg-slate-50 px-3 py-2.5">
                  <p className="text-xs font-medium text-slate-600">Grain yield (harvest)</p>
                  <p className="mt-1 text-sm text-slate-800">
                    Not harvested yet
                    {focusDay.grnwt > 10 && (
                      <span className="text-slate-500">
                        {' '}
                        · grain filling now: {focusDay.grnwt.toFixed(0)} kg/ha
                      </span>
                    )}
                  </p>
                  <p className="mt-0.5 text-xs text-slate-500">
                    Yield is 0 until the crop reaches harvest — you are at DAP {focusDay.dap}.
                  </p>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
