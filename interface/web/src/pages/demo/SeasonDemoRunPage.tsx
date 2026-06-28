import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, Play, RotateCcw } from 'lucide-react';
import { PreferencePanel } from '@/components/PreferencePanel';
import { Button, Card } from '@/components/ui';
import { useApp } from '@/hooks/useAppContext';
import { formatSimProgress } from '@/utils/simProgress';

export function SeasonDemoRunPage() {
  const { error, simRunning, simJob, simError, runSimulation, clearSimulation } = useApp();
  const simProgress = formatSimProgress(simJob);

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h2 className="page-title">Run full season</h2>
        <p className="page-subtitle">
          One background run (~2–5 min) produces the complete season replay for dashboard, log, and
          reports.
        </p>
      </div>

      {(error || simError) && (
        <div className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-700">
          {simError || error}
        </div>
      )}

      <PreferencePanel disabled={simRunning} />

      <Card title="Full-season run" className="space-y-4">
        <p className="text-sm leading-relaxed text-slate-600">
          CAPQL manages irrigation and nitrogen for every day inside DSSAT. When finished, browse
          results under Dashboard, Season log, and Reports.
        </p>
        <div className="flex flex-wrap gap-3">
          <Button onClick={() => void runSimulation()} disabled={simRunning}>
            {simRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run full season
          </Button>
          <Button variant="secondary" onClick={clearSimulation} disabled={simRunning}>
            <RotateCcw className="h-4 w-4" />
            Clear progress
          </Button>
        </div>
      </Card>

      <AnimatePresence>
        {simRunning && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="panel overflow-hidden p-6"
          >
            <p className="text-sm font-medium text-slate-900">{simProgress.label}</p>
            <p className="mt-1 text-xs text-slate-500">{simProgress.detail}</p>
            <div className="mt-4 h-2 overflow-hidden rounded-full bg-brand-200">
              <motion.div
                className="h-full bg-brand-700"
                animate={{ width: `${simProgress.percent}%` }}
                transition={{ duration: 0.4 }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
