import { Outlet } from 'react-router-dom';
import { Leaf } from 'lucide-react';
import { useApp } from '@/hooks/useAppContext';

export function HomeShell() {
  const { config } = useApp();

  return (
    <div className="min-h-screen bg-gradient-to-b from-brand-50/80 to-slate-50">
      <header className="border-b border-brand-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-5xl items-center gap-3 px-6">
          <div className="rounded-lg bg-brand-700 p-1.5 text-white">
            <Leaf className="h-4 w-4" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-900">Smart Farm DSS</p>
            <p className="text-xs text-slate-500">{config?.model ?? 'CAPQL v2'}</p>
          </div>
        </div>
      </header>
      <main className="px-6 py-8">
        <Outlet />
      </main>
      <footer className="border-t border-brand-100 py-6 text-center text-xs text-slate-500">
        Choose a mode to begin · DSSAT live simulation
      </footer>
    </div>
  );
}
