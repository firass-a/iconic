import { AlertTriangle } from 'lucide-react';
import type { SeasonResult } from '@/types';
import { EXPECTED_SEASON_LENGTH, isSeasonComplete, seasonProgressPct } from '@/utils/seasonDays';

interface SeasonStatusBannerProps {
  result: SeasonResult;
  advisoryDone?: boolean;
  advisoryError?: string | null;
  atLastDay?: boolean;
}

export function SeasonStatusBanner({
  result,
  advisoryDone,
  advisoryError,
  atLastDay = false,
}: SeasonStatusBannerProps) {
  const complete = isSeasonComplete(result, { advisoryDone });
  const pct = seasonProgressPct(result);
  const isAlgiers = (result.weather_id ?? result.summary.weather_id ?? '').includes('algiers');

  if (complete && !advisoryError) return null;

  let title = 'Season not finished';
  let detail =
    'Final grain yield is only measured at harvest (~season day 165). Totals below are so far this season, not the final crop.';

  if (advisoryError) {
    title = 'Simulation stopped';
    detail = advisoryError;
  } else if (atLastDay && !complete && advisoryDone === false) {
    title = `Stopped at season day ${result.ep_length}`;
    detail =
      'Click Next day on Daily run to continue. If nothing happens, DSSAT may have timed out — try Gainesville weather or a different seed.';
  } else if (!complete && result.summary.yield_kg_ha < 1) {
    title = `In progress — ${pct}% of season (${result.ep_length} / ~${EXPECTED_SEASON_LENGTH} days)`;
  }

  return (
    <div className="flex items-start rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
      <AlertTriangle className="mr-2 mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
      <div>
        <p className="font-medium">{title}</p>
        <p className="mt-1 text-xs leading-relaxed text-amber-900/90">{detail}</p>
        {isAlgiers && !complete && advisoryError && (
          <p className="mt-2 text-xs text-amber-800">
            If this keeps happening, restart the API so Algiers measured weather files rebuild,
            or switch to <strong>Gainesville WGEN</strong>.
          </p>
        )}
      </div>
    </div>
  );
}
