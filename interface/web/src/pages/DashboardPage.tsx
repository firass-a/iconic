import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Sprout } from 'lucide-react';
import { FieldDashboard } from '@/components/FieldDashboard';
import { DayAgeHelp } from '@/components/DayAgeHelp';
import { useApp } from '@/hooks/useAppContext';
import { findPlantingSimDay, formatSimDayDate, PLANTING_DATE } from '@/utils/seasonDays';

export function DashboardPage() {
  const { result, config, loading, dapSelection, advisorySession, advisoryError } = useApp();
  const planting = config?.planting_date ?? PLANTING_DATE;

  const cur = useMemo(() => {
    if (!result) return null;
    if (dapSelection) {
      return result.days.find((d) => d.day === dapSelection.focus) ?? result.days[0];
    }
    return result.current;
  }, [result, dapSelection]);

  const plantingSimDay = useMemo(
    () => findPlantingSimDay(result?.days ?? []),
    [result?.days],
  );

  if (loading) {
    return <p className="text-sm text-muted">Connecting…</p>;
  }

  if (!result || !cur) {
    return (
      <div className="dash-card mx-auto max-w-lg p-10 text-center">
        <Sprout className="mx-auto h-10 w-10 text-emerald-600" />
        <h2 className="mt-4 text-lg font-semibold text-slate-900">No simulation yet</h2>
        <p className="mt-2 text-sm text-slate-500">Run a season to see field status.</p>
      </div>
    );
  }

  const dateLabel = formatSimDayDate(cur.day, planting, plantingSimDay);

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">
      <div>
        <h2 className="page-title">Dashboard</h2>
        <p className="page-subtitle">
          {dateLabel} · Season day {cur.day} · DAP {cur.dap} (crop age)
        </p>
      </div>

      <DayAgeHelp compact />

      <FieldDashboard
        result={result}
        focusDay={cur}
        plantingDate={planting}
        plantingSimDay={plantingSimDay}
        showSeasonTotals
        advisoryDone={advisorySession?.done}
        advisoryError={advisoryError}
        atLastDay={cur.day === result.days[result.days.length - 1]?.day}
      />
    </motion.div>
  );
}
