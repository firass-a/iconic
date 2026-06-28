import { useApp } from '@/hooks/useAppContext';
import { Card } from '@/components/ui';

export function SettingsPage() {
  const { config } = useApp();

  return (
    <div className="mx-auto max-w-xl space-y-6">
      <div>
        <h2 className="page-title">Settings</h2>
        <p className="page-subtitle">System configuration (DSSAT experiment context)</p>
      </div>
      <Card className="space-y-3 text-sm text-slate-800">
        <p><span className="text-slate-500">Model</span> · {config?.model ?? 'CAPQL v2'}</p>
        <p><span className="text-slate-500">Backend</span> · FastAPI + DSSAT (Docker)</p>
        <p><span className="text-slate-500">Objectives</span> · Yield, N efficiency, Water saving</p>
        <p className="text-xs leading-relaxed text-slate-500">{config?.notes}</p>
      </Card>
    </div>
  );
}
