interface ProgressBarProps {
  label: string;
  value: number;
  max: number;
  unit: string;
  colorClass?: string;
  trackClass?: string;
}

export function ProgressBar({
  label,
  value,
  max,
  unit,
  colorClass = 'bg-brand-600',
  trackClass = 'bg-brand-100',
}: ProgressBarProps) {
  const pct = max > 0 ? Math.min(100, Math.round((value / max) * 100)) : 0;
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <span className="text-sm font-medium text-slate-700">{label}</span>
        <span className="text-sm tabular-nums text-slate-900">
          {value.toFixed(1)}
          <span className="ml-0.5 text-xs font-normal text-slate-500">{unit}</span>
        </span>
      </div>
      <div className={`h-2.5 overflow-hidden rounded-full ${trackClass}`}>
        <div
          className={`h-full rounded-full transition-all ${colorClass}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
