import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import {
  ArrowLeft,
  BarChart3,
  Bell,
  CalendarDays,
  Droplets,
  LayoutDashboard,
  Leaf,
  Loader2,
  Play,
  Search,
  SlidersHorizontal,
  Sprout,
} from 'lucide-react';
import { useApp } from '@/hooks/useAppContext';
import { formatSimProgress } from '@/utils/simProgress';

export type AppMode = 'demo' | 'daily';

const demoNav = [
  { to: '/demo/run', icon: Play, label: 'Run season' },
  { to: '/demo/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/demo/log', icon: Droplets, label: 'Season log' },
  { to: '/demo/reports', icon: BarChart3, label: 'Reports' },
  { to: '/demo/tradeoffs', icon: SlidersHorizontal, label: 'Trade-offs' },
];

const dailyNav = [
  { to: '/daily/run', icon: CalendarDays, label: 'Daily run' },
  { to: '/daily/field', icon: Sprout, label: 'Field status' },
];

const demoTitles: Record<string, string> = {
  '/demo/run': 'Run full season',
  '/demo/dashboard': 'Dashboard',
  '/demo/log': 'Season log',
  '/demo/reports': 'Reports',
  '/demo/tradeoffs': 'Trade-offs',
};

const dailyTitles: Record<string, string> = {
  '/daily/run': 'Daily run',
  '/daily/field': 'Field status',
};

function greeting(): string {
  const h = new Date().getHours();
  if (h < 12) return 'Good morning';
  if (h < 17) return 'Good afternoon';
  return 'Good evening';
}

function headerDate(): string {
  return new Date().toLocaleDateString(undefined, {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });
}

interface ModeShellProps {
  mode: AppMode;
}

export function ModeShell({ mode }: ModeShellProps) {
  const location = useLocation();
  const { error, simRunning, simJob, advisoryBusy } = useApp();
  const nav = mode === 'demo' ? demoNav : dailyNav;
  const titles = mode === 'demo' ? demoTitles : dailyTitles;
  const title = titles[location.pathname] ?? (mode === 'demo' ? 'Season demo' : 'Daily simulation');
  const simProgress = formatSimProgress(simJob);
  const modeLabel = mode === 'demo' ? 'Season demo' : 'Daily simulation';

  return (
    <div className="flex min-h-screen bg-canvas">
      <aside className="flex w-64 shrink-0 flex-col border-r border-slate-200/80 bg-white">
        <div className="border-b border-slate-100 px-5 py-5">
          <Link
            to="/"
            className="mb-4 inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-emerald-700"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Change mode
          </Link>
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-600 text-white shadow-sm">
              <Leaf className="h-5 w-5" />
            </div>
            <div>
              <p className="text-base font-bold text-slate-900">Smart Farm</p>
              <p className="text-xs text-slate-400">{modeLabel}</p>
            </div>
          </div>
        </div>

        <nav className="flex-1 px-3 pb-4">
          <p className="nav-section">Main</p>
          <div className="space-y-0.5">
            {nav.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `nav-link ${isActive ? 'nav-link-active' : 'nav-link-idle'}`
                }
              >
                <Icon className="h-[18px] w-[18px] shrink-0" />
                <span>{label}</span>
                {mode === 'demo' && to === '/demo/run' && simRunning && (
                  <Loader2 className="ml-auto h-3.5 w-3.5 animate-spin text-emerald-600" />
                )}
                {mode === 'daily' && to === '/daily/run' && advisoryBusy && (
                  <Loader2 className="ml-auto h-3.5 w-3.5 animate-spin text-sky-600" />
                )}
              </NavLink>
            ))}
          </div>
        </nav>

        <div className="mx-3 mb-4 rounded-2xl border border-emerald-100 bg-gradient-to-br from-emerald-50 to-white p-4">
          <p className="text-sm font-semibold text-slate-800">CAPQL v2</p>
          <p className="mt-1 text-xs leading-relaxed text-slate-500">
            AI-managed irrigation & nitrogen for maize.
          </p>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="border-b border-slate-200/80 bg-white">
          <div className="flex items-center justify-between gap-4 px-6 py-4">
            <div>
              <p className="text-xs text-slate-400">{headerDate()}</p>
              <h1 className="text-lg font-bold text-slate-900">
                {greeting()} · {title}
              </h1>
            </div>
            <div className="flex items-center gap-2">
              {error && (
                <span className="hidden rounded-lg border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs text-amber-900 sm:inline">
                  {error}
                </span>
              )}
              <button
                type="button"
                className="rounded-xl p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
                aria-label="Search"
              >
                <Search className="h-5 w-5" />
              </button>
              <button
                type="button"
                className="rounded-xl p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
                aria-label="Notifications"
              >
                <Bell className="h-5 w-5" />
              </button>
              <div className="ml-1 flex h-9 w-9 items-center justify-center rounded-full bg-emerald-100 text-sm font-semibold text-emerald-700">
                SF
              </div>
            </div>
          </div>
          {mode === 'demo' && simRunning && (
            <div className="border-t border-slate-100 bg-emerald-50/50 px-6 py-2.5">
              <div className="flex items-center gap-3 text-xs text-slate-700">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-emerald-600" />
                <span className="font-medium">{simProgress.label}</span>
                <span className="hidden text-slate-500 sm:inline">{simProgress.detail}</span>
                <div className="ml-auto h-1.5 w-32 overflow-hidden rounded-full bg-emerald-100">
                  <div
                    className="h-full bg-emerald-600 transition-all duration-300"
                    style={{ width: `${simProgress.percent}%` }}
                  />
                </div>
              </div>
            </div>
          )}
        </header>

        <main className="flex-1 overflow-auto p-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.18 }}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
