import { CalendarRange, Droplets, FlaskConical } from 'lucide-react';
import type { WeeklyRollup } from '@/utils/managementInsights';

interface WeeklySummaryCardProps {
  summary: WeeklyRollup;
}

export function WeeklySummaryCard({ summary }: WeeklySummaryCardProps) {
  const swfacPct = Math.round(summary.avgSwfac * 100);
  const nstresPct = Math.round(summary.avgNstres * 100);

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-2 text-sm font-medium text-slate-900">
        <CalendarRange className="h-4 w-4 text-emerald-600" />
        Weekly summary
        <span className="font-normal text-slate-500">
          (season days {summary.startDay}–{summary.endDay})
        </span>
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <div className="flex items-start gap-2 text-sm">
          <Droplets className="mt-0.5 h-4 w-4 text-sky-500" />
          <div>
            <p className="font-medium text-slate-800">{summary.irrigation_mm.toFixed(0)} mm irrigated</p>
            <p className="text-slate-500">{summary.rain_mm.toFixed(0)} mm rain</p>
          </div>
        </div>
        <div className="flex items-start gap-2 text-sm">
          <FlaskConical className="mt-0.5 h-4 w-4 text-violet-500" />
          <div>
            <p className="font-medium text-slate-800">
              {summary.nitrogen_kg_ha.toFixed(0)} kg/ha N applied
            </p>
            <p className="text-slate-500">
              {summary.managementDays} management day
              {summary.managementDays === 1 ? '' : 's'} in window
            </p>
          </div>
        </div>
      </div>
      <p className="mt-3 text-xs text-slate-500">
        Avg plant water supply {swfacPct}% · avg N supply {nstresPct}% over {summary.dayCount} crop
        days.
      </p>
    </div>
  );
}
