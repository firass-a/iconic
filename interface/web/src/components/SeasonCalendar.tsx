import { useEffect, useMemo, useState } from 'react';
import { Calendar, ChevronLeft, ChevronRight, Droplets, FlaskConical, Sprout } from 'lucide-react';
import type { DayRecord } from '@/types';
import { Button } from '@/components/ui';
import { analyzeManagementDay, levelBarWidth } from '@/utils/managementDay';
import {
  defaultDapInterval,
  findPlantingSimDay,
  formatPlantingDate,
  formatSimDayDate,
  PLANTING_DATE,
  simDayBounds,
  simDayToCalendarDate,
} from '@/utils/seasonDays';
export type SeasonViewMode = 'day' | 'period';

interface SeasonCalendarProps {
  days: DayRecord[];
  startDay: number;
  endDay: number;
  focusDay: number;
  viewMode: SeasonViewMode;
  onViewModeChange: (mode: SeasonViewMode) => void;
  onChangeInterval: (start: number, end: number) => void;
  onFocusDay: (simDay: number) => void;
  plantingDate?: string;
}

const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

export function SeasonCalendar({
  days,
  startDay,
  endDay,
  focusDay,
  viewMode,
  onViewModeChange,
  onChangeInterval,
  onFocusDay,
  plantingDate = PLANTING_DATE,
}: SeasonCalendarProps) {
  const { min, max } = simDayBounds(days);
  const plantingSimDay = useMemo(() => findPlantingSimDay(days), [days]);
  const lo = Math.min(startDay, endDay);
  const hi = Math.max(startDay, endDay);
  const [monthIdx, setMonthIdx] = useState(0);
  const [rangePick, setRangePick] = useState<number | null>(null);

  const months = useMemo(() => {
    const groups = new Map<string, DayRecord[]>();
    for (const d of days) {
      const key = simDayToCalendarDate(d.day, plantingDate, plantingSimDay).toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'long',
      });
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(d);
    }
    return [...groups.entries()];
  }, [days, plantingDate, plantingSimDay]);

  const plantingMonthIdx = useMemo(() => {
    const label = simDayToCalendarDate(plantingSimDay, plantingDate, plantingSimDay).toLocaleDateString(
      undefined,
      { year: 'numeric', month: 'long' },
    );
    const idx = months.findIndex(([name]) => name === label);
    return idx >= 0 ? idx : 0;
  }, [months, plantingDate, plantingSimDay]);

  const focusMonthIdx = useMemo(() => {
    const label = simDayToCalendarDate(focusDay, plantingDate, plantingSimDay).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'long',
    });
    const idx = months.findIndex(([name]) => name === label);
    return idx >= 0 ? idx : plantingMonthIdx;
  }, [focusDay, months, plantingDate, plantingSimDay, plantingMonthIdx]);

  useEffect(() => {
    setMonthIdx(plantingMonthIdx);
  }, [plantingMonthIdx, days.length]);

  const currentMonth = months[monthIdx] ?? months[focusMonthIdx] ?? ['', []];
  const monthDays = currentMonth[1];
  const firstDate = monthDays.length
    ? simDayToCalendarDate(monthDays[0].day, plantingDate, plantingSimDay)
    : new Date();
  const pad = (firstDate.getDay() + 6) % 7;

  const monthIndexForSimDay = (simDay: number) => {
    const label = simDayToCalendarDate(simDay, plantingDate, plantingSimDay).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'long',
    });
    const idx = months.findIndex(([name]) => name === label);
    return idx >= 0 ? idx : plantingMonthIdx;
  };

  const goToSimDay = (simDay: number, mode: SeasonViewMode = viewMode) => {
    onFocusDay(simDay);
    onChangeInterval(simDay, simDay);
    setRangePick(null);
    onViewModeChange(mode);
    setMonthIdx(monthIndexForSimDay(simDay));
  };

  const selectDay = (simDay: number) => {
    onFocusDay(simDay);
    if (viewMode === 'day') {
      onChangeInterval(simDay, simDay);
      setRangePick(null);
      return;
    }
    if (rangePick === null) {
      setRangePick(simDay);
      onChangeInterval(simDay, simDay);
      return;
    }
    onChangeInterval(rangePick, simDay);
    setRangePick(null);
  };

  const applyPreset = (start: number, end: number, focus: number) => {
    onChangeInterval(start, end);
    onFocusDay(focus);
    setRangePick(null);
    onViewModeChange('period');
    setMonthIdx(monthIndexForSimDay(focus));
  };

  const peak = defaultDapInterval(days);
  const busiestDate = formatSimDayDate(peak.focus, plantingDate, plantingSimDay);

  return (
    <div className="panel flex flex-col gap-4 p-5">
      <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
        <Calendar className="h-4 w-4 text-brand-700" />
        Browse season
      </div>

      <div className="rounded-xl border border-brand-200 bg-brand-50/70 px-3 py-2.5">
        <div className="flex items-start gap-2">
          <Sprout className="mt-0.5 h-4 w-4 shrink-0 text-brand-700" />
          <div>
            <p className="text-xs font-medium text-brand-900">Planting date</p>
            <p className="text-sm text-slate-800">{formatPlantingDate(plantingDate)}</p>
            <p className="mt-0.5 text-[11px] text-slate-500">
              Calendar starts here · DAP 0
            </p>
          </div>
        </div>
      </div>

      <div className="flex rounded-xl border border-brand-200 bg-brand-50/80 p-1">
        <button
          type="button"
          onClick={() => {
            onViewModeChange('day');
            onChangeInterval(focusDay, focusDay);
            setRangePick(null);
          }}
          className={`flex-1 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
            viewMode === 'day'
              ? 'bg-white text-brand-900 shadow-sm ring-1 ring-brand-200'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          Single day
        </button>
        <button
          type="button"
          onClick={() => {
            onViewModeChange('period');
            setRangePick(null);
          }}
          className={`flex-1 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
            viewMode === 'period'
              ? 'bg-white text-brand-900 shadow-sm ring-1 ring-brand-200'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          Date range
        </button>
      </div>

      <p className="text-xs leading-relaxed text-slate-600">
        {viewMode === 'day'
          ? 'See what CAPQL applied on one day of the replay. Most days are hold (0 mm, 0 N).'
          : 'Select two dates to sum up applications and browse the day-by-day log.'}
      </p>

      <div className="space-y-2">
        {viewMode === 'period' && (
          <p className="text-xs text-slate-500">
            {rangePick === null
              ? 'Step 1: click the first day. Step 2: click the last day.'
              : `Start: ${formatSimDayDate(rangePick, plantingDate, plantingSimDay)} — now click the last day`}
          </p>
        )}
        <div className="flex flex-wrap gap-2">
          <Button
            variant="secondary"
            className="px-3 py-1.5 text-xs"
            onClick={() => goToSimDay(peak.focus, 'day')}
          >
            Busiest day ({busiestDate})
          </Button>
          {viewMode === 'period' && (
            <Button
              variant="secondary"
              className="px-3 py-1.5 text-xs"
              onClick={() => applyPreset(min, max, peak.focus)}
            >
              Entire season
            </Button>
          )}
        </div>
        <p className="text-[11px] leading-relaxed text-slate-500">
          Shortcut: <strong className="font-medium text-slate-700">Busiest day</strong> skips empty
          hold days and opens {busiestDate}, when the model applied the most irrigation or N.
        </p>
        {viewMode === 'period' && (
          <p className="text-sm text-slate-700">
            <span className="font-medium text-slate-900">
              {formatSimDayDate(lo, plantingDate, plantingSimDay)} –{' '}
              {formatSimDayDate(hi, plantingDate, plantingSimDay)}
            </span>
            <span className="text-slate-500"> · {hi - lo + 1} days</span>
          </p>
        )}
      </div>

      <div className="flex items-center justify-between">
        <button
          type="button"
          className="rounded-lg p-1.5 text-slate-600 hover:bg-brand-100 disabled:opacity-40"
          disabled={monthIdx <= 0}
          onClick={() => setMonthIdx((i) => Math.max(0, i - 1))}
          aria-label="Previous month"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <div className="text-center">
          <p className="text-sm font-semibold text-slate-900">{currentMonth[0]}</p>
          <button
            type="button"
            className="text-xs text-brand-700 hover:underline"
            onClick={() => setMonthIdx(plantingMonthIdx)}
          >
            Jump to planting
          </button>
          <span className="text-slate-300">·</span>
          <button
            type="button"
            className="text-xs text-brand-700 hover:underline"
            onClick={() => setMonthIdx(focusMonthIdx)}
          >
            Jump to selected day
          </button>
        </div>
        <button
          type="button"
          className="rounded-lg p-1.5 text-slate-600 hover:bg-brand-100 disabled:opacity-40"
          disabled={monthIdx >= months.length - 1}
          onClick={() => setMonthIdx((i) => Math.min(months.length - 1, i + 1))}
          aria-label="Next month"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>

      <div className="grid grid-cols-7 gap-1">
        {WEEKDAYS.map((w) => (
          <div key={w} className="text-center text-[10px] font-medium uppercase text-slate-500">
            {w}
          </div>
        ))}
        {Array.from({ length: pad }).map((_, i) => (
          <div key={`pad-${i}`} />
        ))}
        {monthDays.map((d) => {
          const inRange = viewMode === 'period' && d.day >= lo && d.day <= hi;
          const focused = d.day === focusDay;
          const pending = rangePick === d.day;
          const isPlantingDay = d.day === plantingSimDay;
          const mgmt = analyzeManagementDay(d);
          const dateLabel = formatSimDayDate(d.day, plantingDate, plantingSimDay);
          const tooltip = [
            `${dateLabel} · Day ${d.day} · DAP ${d.dap}`,
            mgmt.calendarTitle,
            mgmt.irrigationNote,
            mgmt.nitrogenNote,
          ].join('\n');
          return (
            <button
              key={d.day}
              type="button"
              title={tooltip}
              onClick={() => selectDay(d.day)}
              className={`relative flex h-11 flex-col items-center justify-center rounded-lg text-xs transition-all ${
                focused
                  ? 'bg-brand-700 font-semibold text-white shadow-sm'
                  : pending
                    ? 'bg-brand-200 font-medium text-brand-900 ring-2 ring-brand-500'
                    : inRange
                      ? 'bg-brand-100 font-medium text-brand-900'
                      : isPlantingDay
                        ? 'bg-white font-medium text-brand-900 ring-2 ring-brand-400'
                        : mgmt.irrigationApplied || mgmt.nitrogenApplied
                          ? 'bg-white font-medium text-slate-800 ring-1 ring-brand-100'
                          : 'text-slate-700 hover:bg-brand-50'
              }`}
            >
              <span>{simDayToCalendarDate(d.day, plantingDate, plantingSimDay).getDate()}</span>
              <div className="mt-0.5 flex w-full max-w-[2rem] items-end justify-center gap-0.5 px-0.5">
                <span
                  className={`flex h-1.5 flex-1 items-end justify-center rounded-sm ${
                    focused ? 'bg-white/20' : 'bg-sky-100'
                  }`}
                  aria-hidden
                >
                  {mgmt.irrigationApplied ? (
                    <span
                      className={`rounded-sm ${focused ? 'bg-sky-200' : 'bg-sky-500'} ${levelBarWidth(mgmt.irrigationLevel)} h-full min-w-[2px]`}
                    />
                  ) : (
                    <span className={`h-px w-full ${focused ? 'bg-white/40' : 'bg-sky-200'}`} />
                  )}
                </span>
                <span
                  className={`flex h-1.5 flex-1 items-end justify-center rounded-sm ${
                    focused ? 'bg-white/20' : 'bg-brand-100'
                  }`}
                  aria-hidden
                >
                  {mgmt.nitrogenApplied ? (
                    <span
                      className={`rounded-sm ${focused ? 'bg-brand-200' : 'bg-brand-600'} ${levelBarWidth(mgmt.nitrogenLevel)} h-full min-w-[2px]`}
                    />
                  ) : (
                    <span className={`h-px w-full ${focused ? 'bg-white/40' : 'bg-brand-200'}`} />
                  )}
                </span>
              </div>
              {isPlantingDay && !focused && (
                <Sprout className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 text-brand-600" />
              )}
            </button>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] text-slate-600">
        <span className="flex items-center gap-1.5">
          <Sprout className="h-3 w-3 text-brand-600" />
          Planting
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-flex h-2 w-4 overflow-hidden rounded-sm bg-sky-100">
            <span className="h-full w-2/3 bg-sky-500" />
          </span>
          <Droplets className="h-3 w-3 text-sky-600" />
          Irrigation (bar height = amount)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-flex h-2 w-4 overflow-hidden rounded-sm bg-brand-100">
            <span className="h-full w-2/3 bg-brand-600" />
          </span>
          <FlaskConical className="h-3 w-3 text-brand-700" />
          Nitrogen
        </span>
        <span className="text-slate-500">Flat line = 0 (hold) · hover a day for why</span>
      </div>

      <p className="text-[11px] leading-relaxed text-slate-500">
        One cell per simulated day. Click a day to see irrigation and nitrogen details with field
        indicators (moisture, rain, stress).
      </p>
    </div>
  );
}
