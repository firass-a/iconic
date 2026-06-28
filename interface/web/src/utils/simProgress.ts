import type { SimulationJob } from '@/types';

export function formatSimProgress(job: SimulationJob | null): {
  label: string;
  detail: string;
  percent: number;
} {
  const p = job?.progress;
  const total = p?.total_days ?? 165;
  const phase = p?.phase ?? 'simulating';
  const percent = p?.percent ?? 2;

  if (phase === 'loading_model') {
    return {
      label: 'Loading CAPQL model…',
      detail: 'First run loads the neural network weights (once per API session).',
      percent,
    };
  }
  if (phase === 'starting_dssat') {
    return {
      label: 'Starting DSSAT…',
      detail: 'Spinning up the crop model — the first step often takes 2–5 minutes.',
      percent,
    };
  }

  const dap = p?.current_dap ?? 0;
  const step = p?.current_day ?? 0;
  return {
    label: `Simulating · step ${step} · DAP ${dap}`,
    detail: `~${total} days total · each DSSAT step takes a few seconds`,
    percent,
  };
}
