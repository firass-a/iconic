import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import type { DayRecord } from '@/types';
import { analyzeManagementDay } from '@/utils/managementDay';
import {
  findPlantingSimDay,
  formatSimDayDate,
  PLANTING_DATE,
  simDayToCalendarDate,
} from '@/utils/seasonDays';

interface SeasonDayPickerProps {
  days: DayRecord[];
  focusDay: number;
  onFocusDay: (simDay: number) => void;
  plantingDate?: string;
}

function hasApplication(d: DayRecord) {
  return d.irrigation_mm > 0.1 || d.nitrogen_kg_ha > 0.1;
}

export function SeasonDayPicker({
  days,
  focusDay,
  onFocusDay,
  plantingDate = PLANTING_DATE,
}: SeasonDayPickerProps) {
  const [hideHold, setHideHold] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const plantingSimDay = useMemo(() => findPlantingSimDay(days), [days]);

  const visibleDays = useMemo(
    () => (hideHold ? days.filter(hasApplication) : days),
    [days, hideHold],
  );

  const segments = useMemo(() => {
    const groups: { label: string; short: string; days: DayRecord[] }[] = [];
    let currentKey = '';
    for (const d of visibleDays) {
      const cal = simDayToCalendarDate(d.day, plantingDate, plantingSimDay);
      const key = cal.toLocaleDateString(undefined, { year: 'numeric', month: 'short' });
      const short = cal.toLocaleDateString(undefined, { month: 'short' });
      if (key !== currentKey) {
        groups.push({ label: key, short, days: [d] });
        currentKey = key;
      } else {
        groups[groups.length - 1].days.push(d);
      }
    }
    return groups;
  }, [visibleDays, plantingDate, plantingSimDay]);

  useEffect(() => {
    const el = scrollRef.current?.querySelector(`[data-day="${focusDay}"]`);
    el?.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
  }, [focusDay]);

  const scrollBy = (dir: number) => {
    scrollRef.current?.scrollBy({ left: dir * 160, behavior: 'smooth' });
  };

  const applicationCount = days.filter(hasApplication).length;

  return (
    <div className="dash-card">
      <div className="flex items-center justify-between gap-3 border-b border-slate-100 px-3 py-2">
        <div className="min-w-0">
          <p className="truncate text-xs font-medium text-slate-800">
            {formatSimDayDate(focusDay, plantingDate, plantingSimDay)} · day {focusDay}
          </p>
          <p className="text-[10px] text-slate-500">
            {applicationCount} active · {days.length} days
          </p>
        </div>
        <label className="flex shrink-0 cursor-pointer items-center gap-1.5 rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-[10px] text-slate-600">
          <input
            type="checkbox"
            checked={hideHold}
            onChange={(e) => setHideHold(e.target.checked)}
            className="h-3 w-3 rounded border-slate-300 text-brand-600"
          />
          Active only
        </label>
      </div>

      <div className="relative flex items-center">
        <button
          type="button"
          aria-label="Scroll left"
          onClick={() => scrollBy(-1)}
          className="absolute left-0 z-10 flex h-full items-center bg-gradient-to-r from-white via-white to-transparent pl-1 pr-2 text-slate-400 hover:text-slate-700"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>

        <div
          ref={scrollRef}
          className="scrollbar-thin flex gap-3 overflow-x-auto scroll-smooth px-8 py-2.5"
          style={{ scrollbarWidth: 'thin' }}
        >
          {segments.map((seg) => (
            <div key={seg.label} className="flex shrink-0 items-end gap-0.5">
              <span className="mb-1 mr-1 w-7 shrink-0 text-center text-[9px] font-semibold uppercase tracking-wide text-slate-400">
                {seg.short}
              </span>
              {seg.days.map((d) => {
                const mgmt = analyzeManagementDay(d);
                const focused = d.day === focusDay;
                const cal = simDayToCalendarDate(d.day, plantingDate, plantingSimDay);
                const active = mgmt.irrigationApplied || mgmt.nitrogenApplied;
                return (
                  <button
                    key={d.day}
                    type="button"
                    data-day={d.day}
                    title={`DAP ${d.dap} · ${mgmt.calendarTitle}`}
                    onClick={() => onFocusDay(d.day)}
                    className={`group relative flex h-9 w-7 shrink-0 flex-col items-center justify-center rounded-md text-[10px] font-medium transition-all ${
                      focused
                        ? 'bg-brand-700 text-white shadow-md ring-2 ring-brand-500/40 ring-offset-1'
                        : active
                          ? 'bg-white text-slate-800 shadow-sm ring-1 ring-slate-200 hover:ring-brand-300'
                          : 'text-slate-400 hover:bg-slate-50 hover:text-slate-600'
                    }`}
                  >
                    <span>{cal.getDate()}</span>
                    <span className="mt-0.5 flex h-[3px] w-4 gap-px overflow-hidden rounded-full bg-black/5">
                      <span
                        className={`h-full flex-1 rounded-full ${
                          mgmt.irrigationApplied
                            ? focused
                              ? 'bg-sky-300'
                              : 'bg-sky-500'
                            : 'bg-transparent'
                        }`}
                      />
                      <span
                        className={`h-full flex-1 rounded-full ${
                          mgmt.nitrogenApplied
                            ? focused
                              ? 'bg-brand-300'
                              : 'bg-brand-600'
                            : 'bg-transparent'
                        }`}
                      />
                    </span>
                  </button>
                );
              })}
            </div>
          ))}
          {segments.length === 0 && (
            <p className="px-2 py-1 text-xs text-slate-500">No application days.</p>
          )}
        </div>

        <button
          type="button"
          aria-label="Scroll right"
          onClick={() => scrollBy(1)}
          className="absolute right-0 z-10 flex h-full items-center bg-gradient-to-l from-white via-white to-transparent pl-2 pr-1 text-slate-400 hover:text-slate-700"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

export function DayNavButtons({
  onPrev,
  onNext,
  disablePrev,
  disableNext,
}: {
  onPrev: () => void;
  onNext: () => void;
  disablePrev?: boolean;
  disableNext?: boolean;
}) {
  return (
    <div className="inline-flex overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
      <button
        type="button"
        disabled={disablePrev}
        onClick={onPrev}
        className="inline-flex items-center px-2.5 py-1.5 text-slate-600 hover:bg-slate-50 disabled:opacity-30"
        aria-label="Previous day"
      >
        <ChevronLeft className="h-4 w-4" />
      </button>
      <div className="w-px bg-slate-200" />
      <button
        type="button"
        disabled={disableNext}
        onClick={onNext}
        className="inline-flex items-center px-2.5 py-1.5 text-slate-600 hover:bg-slate-50 disabled:opacity-30"
        aria-label="Next day"
      >
        <ChevronRight className="h-4 w-4" />
      </button>
    </div>
  );
}
