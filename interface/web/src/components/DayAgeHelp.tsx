import { Info } from 'lucide-react';

/** Short explainer — season day (sim step) vs DAP (crop age). */
export function DayAgeHelp({ compact = false }: { compact?: boolean } = {}) {
  if (compact) {
    return (
      <p className="text-xs text-slate-500">
        <strong className="font-medium text-slate-600">Season day</strong> = step in the simulation
        (1…165). <strong className="font-medium text-slate-600">DAP</strong> = days after planting
        (crop age). They differ early on — e.g. season day 40 can still be DAP 13.
      </p>
    );
  }

  return (
    <div className="flex items-start justify-between rounded-xl border border-sky-100 bg-sky-50/80 px-4 py-3 text-sm text-sky-900">
      <div className="flex min-w-0 items-start justify-between">
        <Info className="mr-2 mt-0.5 h-4 w-4 shrink-0 text-sky-600" />
        <div className="min-w-0 space-y-1">
          <p className="font-medium text-sky-950">Season day vs DAP</p>
          <p className="text-xs leading-relaxed text-sky-800/90">
            <strong>Season day</strong> counts every simulation step from season start (what the
            calendar and charts use). <strong>DAP</strong> (days after planting) is crop age — it
            stays at 0 until planting, then increases by 1 each day. Season day 10 is not the same as
            DAP 10.
          </p>
        </div>
      </div>
    </div>
  );
}
