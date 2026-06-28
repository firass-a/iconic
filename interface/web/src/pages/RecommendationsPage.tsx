import { Link } from 'react-router-dom';

import { useMemo } from 'react';

import { motion } from 'framer-motion';

import { Sprout } from 'lucide-react';

import { FieldDashboard } from '@/components/FieldDashboard';

import { ManagementOutlook } from '@/components/ManagementOutlook';

import { RunContextBanner } from '@/components/RunContextBanner';

import { WeeklySummaryCard } from '@/components/WeeklySummaryCard';

import { DayNavButtons, SeasonDayPicker } from '@/components/SeasonDayPicker';

import { useApp } from '@/hooks/useAppContext';

import { analyzeManagementDay } from '@/utils/managementDay';

import { managementOutlook, weeklySummary } from '@/utils/managementInsights';

import {

  findDayBySimDay,

  findPlantingSimDay,

  findSimDayIndex,

  formatSimDayDate,

  isCropActiveDay,

  isSeasonComplete,

  PLANTING_DATE,

  stepSimDayCropActive,

} from '@/utils/seasonDays';



export function RecommendationsPage() {

  const { result, dapSelection, setDapSelection, config } = useApp();



  const planting = config?.planting_date ?? PLANTING_DATE;

  const plantingSimDay = useMemo(

    () => findPlantingSimDay(result?.days ?? []),

    [result?.days],

  );



  const focusDay = useMemo(() => {

    if (!result || !dapSelection) return null;

    return findDayBySimDay(result.days, dapSelection.focus) ?? result.days[0];

  }, [result, dapSelection]);

  const cropDays = useMemo(() => {
    if (!result) return [];
    return result.days.filter((d) => isCropActiveDay(d, plantingSimDay));
  }, [result, plantingSimDay]);

  if (!result || !dapSelection || !focusDay) {

    return (

      <div className="dash-card mx-auto max-w-lg p-8 text-center">

        <Sprout className="mx-auto h-10 w-10 text-emerald-600" />

        <p className="mt-4 text-sm text-slate-500">Run a full season first.</p>

        <Link

          to="/demo/run"

          className="mt-6 inline-flex rounded-xl bg-emerald-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-emerald-700"

        >

          Run full season

        </Link>

      </div>

    );

  }



  const mgmt = analyzeManagementDay(focusDay);

  const focusIdx = findSimDayIndex(cropDays.length ? cropDays : result.days, focusDay.day);
  const browseDays = cropDays.length ? cropDays : result.days;
  const dateLabel = formatSimDayDate(focusDay.day, planting, plantingSimDay);
  const seasonComplete = isSeasonComplete(result);
  const week = weeklySummary(result.days, focusDay.day, plantingSimDay);
  const outlook = managementOutlook(result.days, focusDay, plantingSimDay, seasonComplete);

  const goDay = (delta: number) => {
    const simDay = stepSimDayCropActive(result.days, focusDay.day, delta, plantingSimDay);
    setDapSelection({ start: simDay, end: simDay, focus: simDay });
  };



  const selectDay = (simDay: number) => {

    setDapSelection({ start: simDay, end: simDay, focus: simDay });

  };



  return (

    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5">
      <RunContextBanner
        weatherId={result.weather_id ?? result.summary.weather_id}
        weatherLabel={result.weather_label ?? result.summary.weather_label}
        config={config}
      />

      <div className="flex flex-wrap items-end justify-between gap-3">

        <div>

          <h2 className="page-title">Season log</h2>

          <p className="page-subtitle">
            {dateLabel} · Season day {focusDay.day} · DAP {focusDay.dap}
          </p>

        </div>

        <DayNavButtons

          onPrev={() => goDay(-1)}

          onNext={() => goDay(1)}

          disablePrev={focusIdx <= 0}

          disableNext={focusIdx >= browseDays.length - 1}

        />

      </div>



      <SeasonDayPicker

        days={result.days}

        focusDay={dapSelection.focus}

        onFocusDay={selectDay}

        plantingDate={planting}

      />

      {week && isCropActiveDay(focusDay, plantingSimDay) && <WeeklySummaryCard summary={week} />}
      {isCropActiveDay(focusDay, plantingSimDay) && <ManagementOutlook hints={outlook} />}

      <FieldDashboard

        result={result}

        focusDay={focusDay}

        plantingDate={planting}

        plantingSimDay={plantingSimDay}

      />



      {(focusDay.irrigation_mm < 0.1 || focusDay.nitrogen_kg_ha < 0.1) && (

        <div className="dash-card px-4 py-3 text-sm leading-relaxed text-slate-600">

          {focusDay.irrigation_mm < 0.1 && <p>{mgmt.irrigationNote}</p>}

          {focusDay.nitrogen_kg_ha < 0.1 && (

            <p className={focusDay.irrigation_mm < 0.1 ? 'mt-2' : ''}>{mgmt.nitrogenNote}</p>

          )}

        </div>

      )}

    </motion.div>

  );

}

