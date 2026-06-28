import { useState } from 'react';
import { Loader2 } from 'lucide-react';
import { useApp } from '@/hooks/useAppContext';
import { usePreferenceSliders } from '@/hooks/usePreferenceSliders';
import { Button } from '@/components/ui';
import { api } from '@/services/api';

interface CityClimatePanelProps {
  disabled?: boolean;
  lockSeed?: boolean;
}

export function CityClimatePanel({ disabled = false, lockSeed = false }: CityClimatePanelProps) {
  const { config, refreshConfig } = useApp();
  const { setWeatherId } = usePreferenceSliders();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const algiers = config?.city_presets?.find((c) => c.city_id === 'algiers');
  if (!algiers) return null;

  const locked = disabled || lockSeed;

  const buildAlgiers = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await api.buildCityClimate('algiers');
      await refreshConfig();
      setWeatherId(res.weather_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not download NASA climate');
    } finally {
      setBusy(false);
    }
  };

  if (algiers.built) return null;

  return (
    <div className="rounded-xl border border-sky-200 bg-sky-50/60 p-4 text-sm text-slate-700">
      <p className="font-medium text-slate-900">{algiers.name} climate (NASA)</p>
      <p className="mt-1 leading-relaxed">
        Download {algiers.period} rain and temperature for Algiers, then use it as a synthetic
        season. Crop and soil stay on the Gainesville benchmark — good for exploring a drier
        Mediterranean pattern.
      </p>
      {error && <p className="mt-2 text-red-700">{error}</p>}
      <Button
        className="mt-3"
        disabled={locked || busy}
        onClick={() => void buildAlgiers()}
      >
        {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
        Add Algiers climate to weather list
      </Button>
    </div>
  );
}
