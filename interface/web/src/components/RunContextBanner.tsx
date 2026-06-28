import { Info } from 'lucide-react';
import type { SimConfig } from '@/types';

interface RunContextBannerProps {
  weatherId?: string;
  weatherLabel?: string;
  config?: SimConfig | null;
}

export function RunContextBanner({ weatherId, weatherLabel, config }: RunContextBannerProps) {
  const trainingId = config?.training_weather_id ?? 'wgen-ufga';
  const trainingLabel =
    config?.training_weather_label ?? 'Gainesville WGEN (UFGA — training climate)';
  const guardsOn = config?.agronomic_guards !== false;
  const isTransfer = Boolean(weatherId && weatherId !== trainingId);

  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
      <div className="flex gap-2">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" />
        <div>
          <p className="font-medium text-slate-900">About this run</p>
          <ul className="mt-1.5 list-disc space-y-1 pl-4 text-slate-600">
            <li>
              Policy trained on <strong>{trainingLabel}</strong>; soil, crop, and planting are fixed
              UFGA DSSAT settings.
            </li>
            <li>
              Current weather: <strong>{weatherLabel ?? weatherId ?? 'default'}</strong>.
              {isTransfer && ' Climate transfer — use as illustration, not local calibration.'}
            </li>
            <li>
              Recommendations are learned pulses (not fixed rules). Many days show zero inputs.
              {guardsOn &&
                ' Simple agronomic guards reduce irrigation on wet soil and early N when demand is low.'}
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
