import type { ReactNode } from 'react';
import { TrendingDown, TrendingUp } from 'lucide-react';

type IconTone = 'emerald' | 'sky' | 'violet' | 'amber';

const toneStyles: Record<IconTone, string> = {
  emerald: 'bg-emerald-50 text-emerald-600',
  sky: 'bg-sky-50 text-sky-600',
  violet: 'bg-violet-50 text-violet-600',
  amber: 'bg-amber-50 text-amber-600',
};

interface KpiCardProps {
  label: string;
  value: string | number;
  unit?: string;
  hint?: string;
  trend?: { value: string; up?: boolean };
  icon: ReactNode;
  iconTone?: IconTone;
}

export function KpiCard({
  label,
  value,
  unit,
  hint,
  trend,
  icon,
  iconTone = 'emerald',
}: KpiCardProps) {
  return (
    <div className="dash-card flex items-start justify-between gap-3 p-5">
      <div className="min-w-0">
        <p className="text-sm font-medium text-slate-500">{label}</p>
        <p className="mt-1 text-2xl font-bold tracking-tight text-slate-900">
          {value}
          {unit && <span className="ml-1 text-base font-semibold text-slate-400">{unit}</span>}
        </p>
        {trend && (
          <p className="mt-2 flex items-center gap-1 text-xs text-slate-500">
            {trend.up ? (
              <TrendingUp className="h-3.5 w-3.5 text-emerald-500" />
            ) : (
              <TrendingDown className="h-3.5 w-3.5 text-rose-500" />
            )}
            <span className={trend.up ? 'text-emerald-600' : 'text-rose-600'}>{trend.value}</span>
          </p>
        )}
        {hint && !trend && <p className="mt-2 text-xs text-slate-500">{hint}</p>}
      </div>
      <div className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${toneStyles[iconTone]}`}>
        {icon}
      </div>
    </div>
  );
}
