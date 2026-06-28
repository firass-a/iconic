import type { DayRecord } from '@/types';
import { ProgressBar } from '@/components/ProgressBar';
import { computeCropHealth, computeFieldConditions } from '@/utils/formatAgronomic';

interface FieldHealthBarsProps {
  day: DayRecord;
  compact?: boolean;
}

export function FieldHealthBars({ day, compact = false }: FieldHealthBarsProps) {
  const crop = computeCropHealth(day);
  const field = computeFieldConditions(day);

  return (
    <div className={compact ? 'space-y-3' : 'dash-card space-y-4 p-5'}>
      {!compact && (
        <div>
          <p className="text-sm font-semibold text-slate-800">Field status</p>
          <p className="text-xs text-slate-500">
            Live from DSSAT — before today&apos;s water and nitrogen decisions
          </p>
        </div>
      )}
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <ProgressBar
            label={`Crop health · ${crop.status}`}
            value={crop.pct}
            max={100}
            unit="%"
            colorClass={crop.fill}
            trackClass={crop.track}
          />
          <p className="mt-1.5 text-xs leading-relaxed text-slate-500">{crop.hint}</p>
        </div>
        <div>
          <ProgressBar
            label={`Field conditions · ${field.status}`}
            value={field.pct}
            max={100}
            unit="%"
            colorClass={field.fill}
            trackClass={field.track}
          />
          <p className="mt-1.5 text-xs leading-relaxed text-slate-500">{field.hint}</p>
        </div>
      </div>
    </div>
  );
}
