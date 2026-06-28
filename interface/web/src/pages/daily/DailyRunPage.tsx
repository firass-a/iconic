import { Link } from 'react-router-dom';
import { ChevronRight, Loader2, Play, RotateCcw } from 'lucide-react';
import { FieldSimView } from '@/components/FieldSimView';
import { ManagementOutlook } from '@/components/ManagementOutlook';
import { PreferencePanel } from '@/components/PreferencePanel';
import { RunContextBanner } from '@/components/RunContextBanner';
import { Button } from '@/components/ui';
import { useApp } from '@/hooks/useAppContext';
import { managementOutlook } from '@/utils/managementInsights';
import {
  formatSimDayDate,
  findPlantingSimDay,
  isCropActiveDay,
  isSeasonComplete,
  PLANTING_DATE,
} from '@/utils/seasonDays';

export function DailyRunPage() {
  const {
    error,
    advisorySession,
    advisoryBusy,
    advisoryError,
    startAdvisory,
    stepAdvisory,
    endAdvisory,
    result,
    config,
    preference,
  } = useApp();

  const busy = advisoryBusy;
  const current = advisorySession?.current;
  const days = result?.days ?? [];
  const planting = config?.planting_date ?? PLANTING_DATE;
  const plantingSimDay = findPlantingSimDay(days);
  const dateLabel = current
    ? formatSimDayDate(current.day, planting, plantingSimDay)
    : '';
  const seasonComplete = result
    ? isSeasonComplete(result, { advisoryDone: advisorySession?.done })
    : false;
  const outlook =
    current && days.length
      ? managementOutlook(days, current, plantingSimDay, seasonComplete)
      : [];

  return (
    <div className="mx-auto max-w-2xl space-y-5">
      <div>
        <h2 className="page-title">Daily run</h2>
        <p className="page-subtitle">Step through the season one day at a time.</p>
      </div>

      <RunContextBanner
        weatherId={preference.weather_id}
        weatherLabel={result?.weather_label ?? result?.summary.weather_label}
        config={config}
      />

      {(error || advisoryError) && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          <p className="font-medium">Simulation stopped</p>
          <p className="mt-1">{advisoryError || error}</p>
          {(preference.weather_id ?? '').includes('algiers') && (
            <p className="mt-2 text-xs">
              Algiers weather often hangs in DSSAT. Switch to Gainesville WGEN or UFGA 1982, end
              this season, and start again.
            </p>
          )}
        </div>
      )}

      <PreferencePanel disabled={busy} lockSeed={!!advisorySession} />

      {current ? (
        <>
          {current.dap === 0 && !isCropActiveDay(current, plantingSimDay) && (
            <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
              Pre-plant day — keep stepping until DAP 1; field browse will focus on crop days.
            </p>
          )}
          <FieldSimView focusDay={current} dateLabel={dateLabel} />
          {isCropActiveDay(current, plantingSimDay) && <ManagementOutlook hints={outlook} />}
        </>
      ) : (
        <div className="rounded-xl border border-slate-200 bg-white p-5 text-sm text-slate-600">
          {!advisorySession ? (
            <p>
              Choose your priorities below, then click <strong>Start season</strong>.
            </p>
          ) : (
            <p>
              Season is ready. Click <strong>Next day</strong> for the first recommendation.
            </p>
          )}
        </div>
      )}

      <div className="flex flex-wrap gap-3">
        {!advisorySession ? (
          <Button onClick={() => void startAdvisory()} disabled={busy}>
            {advisoryBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Start season
          </Button>
        ) : (
          <>
            <Button onClick={() => void stepAdvisory()} disabled={busy || advisorySession.done}>
              {advisoryBusy ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
              Next day
            </Button>
            <Button variant="secondary" onClick={() => void endAdvisory()} disabled={busy}>
              <RotateCcw className="h-4 w-4" />
              End season
            </Button>
            {current && (
              <Link
                to="/daily/field"
                className="inline-flex items-center rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-brand-800 hover:bg-brand-50"
              >
                Field status
              </Link>
            )}
          </>
        )}
      </div>

      {advisoryBusy && (
        <p className="text-sm text-slate-500">Running DSSAT — may take a moment on the first day.</p>
      )}
    </div>
  );
}
