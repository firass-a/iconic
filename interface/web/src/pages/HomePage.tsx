import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowRight, CalendarDays, Leaf, Zap } from 'lucide-react';
import { useApp } from '@/hooks/useAppContext';

export function HomePage() {
  const { config } = useApp();

  return (
    <div className="mx-auto flex min-h-[calc(100vh-8rem)] max-w-4xl flex-col justify-center space-y-10 py-8">
      <div className="text-center">
        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-600 text-white shadow-md">
          <Leaf className="h-7 w-7" />
        </div>
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900">Smart Farm DSS</h1>
        <p className="mx-auto mt-3 max-w-xl text-base leading-relaxed text-slate-600">
          CAPQL on DSSAT maize — choose how you want to explore the season. Each mode uses the same
          model; the workflow is different.
        </p>
        <p className="mt-2 text-sm text-slate-500">{config?.crop ?? 'Maize'} · {config?.location ?? 'UFGA benchmark'}</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
          <Link
            to="/daily/run"
            className="group flex h-full flex-col rounded-2xl border border-slate-200/70 bg-white p-8 shadow-[0_1px_3px_rgba(0,0,0,0.04),0_4px_12px_rgba(0,0,0,0.03)] transition-all hover:border-emerald-300 hover:shadow-md"
          >
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-sky-100 text-sky-800">
              <CalendarDays className="h-6 w-6" />
            </div>
            <h2 className="mt-5 text-xl font-semibold text-slate-900">Daily simulation</h2>
            <p className="mt-3 flex-1 text-sm leading-relaxed text-slate-600">
              Step through the crop year one day at a time. Weather updates DSSAT, CAPQL reads the
              field, you get today&apos;s water and N decision — then move to the next day.
            </p>
            <ul className="mt-4 space-y-1 text-xs text-slate-500">
              <li>· Live advisory loop</li>
              <li>· Field status after each step</li>
              <li>· Best for understanding daily decisions</li>
            </ul>
            <span className="mt-6 inline-flex items-center gap-2 text-sm font-medium text-emerald-700 group-hover:gap-3">
              Open daily mode
              <ArrowRight className="h-4 w-4" />
            </span>
          </Link>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.12 }}>
          <Link
            to="/demo/run"
            className="group flex h-full flex-col rounded-2xl border border-slate-200/70 bg-white p-8 shadow-[0_1px_3px_rgba(0,0,0,0.04),0_4px_12px_rgba(0,0,0,0.03)] transition-all hover:border-emerald-300 hover:shadow-md"
          >
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-100 text-emerald-800">
              <Zap className="h-6 w-6" />
            </div>
            <h2 className="mt-5 text-xl font-semibold text-slate-900">Season demo</h2>
            <p className="mt-3 flex-1 text-sm leading-relaxed text-slate-600">
              Run a full season in one go, then browse the complete management replay — dashboard,
              day-by-day log, reports, and trade-offs vs eval data.
            </p>
            <ul className="mt-4 space-y-1 text-xs text-slate-500">
              <li>· Full-season CAPQL replay</li>
              <li>· Calendar &amp; reports</li>
              <li>· Best for thesis eval &amp; comparisons</li>
            </ul>
            <span className="mt-6 inline-flex items-center gap-2 text-sm font-medium text-emerald-700 group-hover:gap-3">
              Open season demo
              <ArrowRight className="h-4 w-4" />
            </span>
          </Link>
        </motion.div>
      </div>
    </div>
  );
}
