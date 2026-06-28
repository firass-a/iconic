import { Droplets, FlaskConical, Minus } from 'lucide-react';
import type { DayRecord } from '@/types';
import { formatSimDayDate } from '@/utils/seasonDays';

function taskTone(d: DayRecord): { label: string; className: string } {
  const hasWater = d.irrigation_mm > 0.1;
  const hasN = d.nitrogen_kg_ha > 0.1;
  if (hasWater && hasN) return { label: 'Water + N', className: 'bg-emerald-100 text-emerald-700' };
  if (hasWater) return { label: 'Irrigate', className: 'bg-sky-100 text-sky-700' };
  if (hasN) return { label: 'Fertilize', className: 'bg-violet-100 text-violet-700' };
  return { label: 'Hold', className: 'bg-slate-100 text-slate-500' };
}

export function ManagementTasks({
  days,
  focusDay,
  plantingDate,
  plantingSimDay = 1,
  title = 'Upcoming management',
}: {
  days: DayRecord[];
  focusDay: number;
  plantingDate?: string;
  plantingSimDay?: number;
  title?: string;
}) {
  const idx = days.findIndex((d) => d.day === focusDay);
  const upcoming = idx >= 0 ? days.slice(idx, idx + 5) : days.slice(0, 5);

  return (
    <div className="dash-card p-5">
      <p className="text-sm font-semibold text-slate-800">{title}</p>
      <ul className="mt-3 space-y-2">
        {upcoming.map((d, i) => {
          const tone = taskTone(d);
          const date = formatSimDayDate(d.day, plantingDate, plantingSimDay);
          return (
            <li
              key={d.day}
              className="flex items-center gap-3 rounded-xl border border-slate-100 bg-slate-50/80 px-3 py-2.5"
            >
              <span className={`shrink-0 rounded-lg px-2 py-0.5 text-[10px] font-semibold ${tone.className}`}>
                {i === 0 ? 'Today' : tone.label}
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-slate-800">
                  DAP {d.dap} · {date}
                </p>
                <p className="text-xs text-slate-500">
                  {d.irrigation_mm > 0.1 || d.nitrogen_kg_ha > 0.1
                    ? `${d.irrigation_mm.toFixed(0)} mm water · ${d.nitrogen_kg_ha.toFixed(0)} kg/ha N`
                    : 'No application scheduled'}
                </p>
              </div>
              <div className="flex shrink-0 gap-1 text-slate-400">
                {d.irrigation_mm > 0.1 ? (
                  <Droplets className="h-4 w-4 text-sky-500" />
                ) : (
                  <Minus className="h-4 w-4" />
                )}
                {d.nitrogen_kg_ha > 0.1 ? (
                  <FlaskConical className="h-4 w-4 text-violet-500" />
                ) : (
                  <Minus className="h-4 w-4" />
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
