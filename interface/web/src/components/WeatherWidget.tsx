import { CloudSun, Droplets, Wind } from 'lucide-react';
import type { DayRecord } from '@/types';
import { formatSimDayDate } from '@/utils/seasonDays';

interface WeatherWidgetProps {
  current: DayRecord;
  upcoming: DayRecord[];
  dateLabel: string;
  plantingDate?: string;
  plantingSimDay?: number;
}

export function WeatherWidget({
  current,
  upcoming,
  dateLabel,
  plantingDate,
  plantingSimDay = 1,
}: WeatherWidgetProps) {
  const avgT = ((current.tmax_c + (current.tmax_c - 8)) / 2).toFixed(0);

  return (
    <div className="dash-card p-5">
      <p className="text-sm font-semibold text-slate-800">Weather</p>
      <p className="text-xs text-slate-400">{dateLabel}</p>

      <div className="mt-4 flex items-center gap-4">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-sky-50 text-sky-500">
          <CloudSun className="h-8 w-8" />
        </div>
        <div>
          <p className="text-3xl font-bold text-slate-900">{current.tmax_c.toFixed(0)}°C</p>
          <p className="text-sm text-slate-500">
            {current.rain_mm > 2 ? 'Rainy period' : 'Clear · simulated'}
          </p>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2 text-center">
        <div className="rounded-xl bg-slate-50 px-2 py-2.5">
          <Droplets className="mx-auto h-4 w-4 text-sky-500" />
          <p className="mt-1 text-xs text-slate-500">Rain</p>
          <p className="text-sm font-semibold text-slate-800">{current.rain_mm.toFixed(0)} mm</p>
        </div>
        <div className="rounded-xl bg-slate-50 px-2 py-2.5">
          <Wind className="mx-auto h-4 w-4 text-slate-400" />
          <p className="mt-1 text-xs text-slate-500">Avg temp</p>
          <p className="text-sm font-semibold text-slate-800">{avgT}°C</p>
        </div>
        <div className="rounded-xl bg-slate-50 px-2 py-2.5">
          <CloudSun className="mx-auto h-4 w-4 text-amber-500" />
          <p className="mt-1 text-xs text-slate-500">Radiation</p>
          <p className="text-sm font-semibold text-slate-800">{current.srad.toFixed(0)}</p>
        </div>
      </div>

      {upcoming.length > 0 && (
        <div className="mt-4 border-t border-slate-100 pt-4">
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">Next days</p>
          <div className="flex justify-between gap-1">
            {upcoming.slice(0, 4).map((d) => (
              <div key={d.day} className="flex-1 rounded-lg bg-slate-50 px-1 py-2 text-center">
                <p className="text-[10px] text-slate-400">
                  {formatSimDayDate(d.day, plantingDate, plantingSimDay)}
                </p>
                <p className="text-sm font-semibold text-slate-800">{d.tmax_c.toFixed(0)}°</p>
                <p className="text-[10px] text-sky-600">{d.rain_mm > 0 ? `${d.rain_mm.toFixed(0)}mm` : '—'}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
