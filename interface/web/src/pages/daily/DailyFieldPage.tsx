import { useEffect, useMemo } from 'react';

import { Link } from 'react-router-dom';

import { ChevronLeft, ChevronRight, Loader2, Sprout } from 'lucide-react';

import { FieldDashboard } from '@/components/FieldDashboard';

import { ManagementOutlook } from '@/components/ManagementOutlook';

import { PrePlantBanner } from '@/components/PrePlantBanner';

import { RunContextBanner } from '@/components/RunContextBanner';

import { WeeklySummaryCard } from '@/components/WeeklySummaryCard';

import { Button } from '@/components/ui';

import { useApp } from '@/hooks/useAppContext';

import { managementOutlook, weeklySummary } from '@/utils/managementInsights';

import {

  cropActiveBounds,

  filterCropActiveDays,

  findDayBySimDay,

  findSimDayIndex,

  formatSimDayDate,

  findPlantingSimDay,

  isCropActiveDay,

  isSeasonComplete,

  PLANTING_DATE,

  stepSimDayCropActive,

} from '@/utils/seasonDays';



export function DailyFieldPage() {

  const {

    result,

    advisorySession,

    advisoryBusy,

    advisoryError,

    dapSelection,

    setDapSelection,

    stepAdvisory,

    config,

  } = useApp();



  const days = result?.days ?? [];

  const planting = config?.planting_date ?? PLANTING_DATE;

  const plantingSimDay = useMemo(() => findPlantingSimDay(days), [days]);

  const cropDays = useMemo(() => filterCropActiveDays(days, plantingSimDay), [days, plantingSimDay]);



  useEffect(() => {

    if (!days.length || !dapSelection) return;

    if (!cropDays.length) return;

    const focus = findDayBySimDay(days, dapSelection.focus);

    if (focus && !isCropActiveDay(focus, plantingSimDay)) {

      const latest = cropDays[cropDays.length - 1];

      setDapSelection({ start: latest.day, end: latest.day, focus: latest.day });

    }

  }, [days, dapSelection, cropDays, plantingSimDay, setDapSelection]);



  const focusDay = useMemo(() => {

    if (!days.length) return null;

    if (dapSelection) {

      return findDayBySimDay(days, dapSelection.focus) ?? days[days.length - 1];

    }

    return days[days.length - 1];

  }, [days, dapSelection]);



  const focusIdx = focusDay ? findSimDayIndex(cropDays, focusDay.day) : -1;

  const { min: cropMin } = cropActiveBounds(days, plantingSimDay);

  const canGoBack = cropDays.length > 0 && focusIdx > 0;

  const canGoForward = cropDays.length > 0 && focusIdx >= 0 && focusIdx < cropDays.length - 1;



  const goDay = (delta: number) => {

    if (!focusDay || !days.length) return;

    const next = stepSimDayCropActive(days, focusDay.day, delta, plantingSimDay);

    setDapSelection({ start: next, end: next, focus: next });

  };



  const seasonComplete = result

    ? isSeasonComplete(result, { advisoryDone: advisorySession?.done })

    : false;



  const week = focusDay ? weeklySummary(days, focusDay.day, plantingSimDay) : null;

  const outlook = focusDay

    ? managementOutlook(days, focusDay, plantingSimDay, seasonComplete)

    : [];



  if (!advisorySession && !days.length) {

    return (

      <div className="dash-card mx-auto max-w-lg p-8 text-center">

        <Sprout className="mx-auto h-10 w-10 text-emerald-600" />

        <h2 className="mt-4 text-lg font-semibold text-slate-900">No active daily run</h2>

        <p className="mt-2 text-sm text-slate-500">

          Start a daily simulation and step through at least one day.

        </p>

        <Link

          to="/daily/run"

          className="mt-6 inline-flex rounded-xl bg-emerald-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-emerald-700"

        >

          Go to daily run

        </Link>

      </div>

    );

  }



  if (!focusDay || !result) {

    return (

      <div className="dash-card mx-auto max-w-lg p-8 text-center">

        <Sprout className="mx-auto h-10 w-10 text-emerald-600" />

        <h2 className="mt-4 text-lg font-semibold text-slate-900">Waiting for first day</h2>

        <p className="mt-2 text-sm text-slate-500">

          Season is started. Use <strong>Next day</strong> on Daily run to simulate the first step.

        </p>

        <Link

          to="/daily/run"

          className="mt-6 inline-flex rounded-xl bg-emerald-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-emerald-700"

        >

          Daily run

        </Link>

      </div>

    );

  }



  const dateLabel = formatSimDayDate(focusDay.day, planting, plantingSimDay);

  const showPrePlant = !isCropActiveDay(focusDay, plantingSimDay);



  return (

    <div className="space-y-5">

      <RunContextBanner

        weatherId={result.weather_id ?? result.summary.weather_id}

        weatherLabel={result.weather_label ?? result.summary.weather_label}

        config={config}

      />



      <div className="flex flex-wrap items-end justify-between gap-3">

        <div>

          <h2 className="page-title">Field status</h2>

          <p className="page-subtitle">

            {dateLabel} · Season day {focusDay.day} · DAP {focusDay.dap}

            {advisorySession?.done ? ' · complete' : ''}

          </p>

        </div>

        <div className="flex flex-wrap gap-2">

          <Button variant="secondary" className="px-3 py-2" disabled={!canGoBack} onClick={() => goDay(-1)}>

            <ChevronLeft className="h-4 w-4" />

            Previous

          </Button>

          {canGoForward ? (

            <Button variant="secondary" className="px-3 py-2" onClick={() => goDay(1)}>

              Next

              <ChevronRight className="h-4 w-4" />

            </Button>

          ) : !advisorySession?.done ? (

            <Button className="px-3 py-2" disabled={advisoryBusy} onClick={() => void stepAdvisory()}>

              {advisoryBusy ? (

                <Loader2 className="h-4 w-4 animate-spin" />

              ) : (

                <ChevronRight className="h-4 w-4" />

              )}

              Next day

            </Button>

          ) : (

            <Button variant="secondary" className="px-3 py-2" disabled>

              Season complete

            </Button>

          )}

        </div>

      </div>



      {showPrePlant && (

        <PrePlantBanner

          seasonDay={focusDay.day}

          plantingSimDay={plantingSimDay}

          daysUntilPlanting={Math.max(0, cropMin - focusDay.day)}

        />

      )}



      {week && isCropActiveDay(focusDay, plantingSimDay) && <WeeklySummaryCard summary={week} />}

      {isCropActiveDay(focusDay, plantingSimDay) && <ManagementOutlook hints={outlook} />}



      <FieldDashboard

        result={result}

        focusDay={focusDay}

        plantingDate={planting}

        plantingSimDay={plantingSimDay}

        advisoryDone={advisorySession?.done}

        advisoryError={advisoryError}

        atLastDay={!canGoForward}

      />

    </div>

  );

}
