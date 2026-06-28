import { Link } from 'react-router-dom';
import { Sprout } from 'lucide-react';
import { DashboardPage } from '@/pages/DashboardPage';
import { useApp } from '@/hooks/useAppContext';

export function DemoDashboardPage() {
  const { result } = useApp();

  if (!result) {
    return (
      <div className="mx-auto max-w-lg rounded-2xl border border-dashed border-brand-300 bg-white p-10 text-center">
        <Sprout className="mx-auto h-10 w-10 text-brand-700" />
        <h2 className="mt-4 text-lg font-semibold text-slate-900">No season data yet</h2>
        <p className="mt-2 text-sm text-slate-600">
          Run a full season first, then field overview and charts appear here.
        </p>
        <Link
          to="/demo/run"
          className="mt-6 inline-flex rounded-xl bg-brand-700 px-5 py-2.5 text-sm font-medium text-white hover:bg-brand-800"
        >
          Run full season
        </Link>
      </div>
    );
  }

  return <DashboardPage />;
}
