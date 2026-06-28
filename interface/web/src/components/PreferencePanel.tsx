import { useApp } from '@/hooks/useAppContext';
import { formatPreference, preferenceVector, usePreferenceSliders } from '@/hooks/usePreferenceSliders';
import { CityClimatePanel } from '@/components/CityClimatePanel';
import { Button, Card } from '@/components/ui';

function SliderRow({
  label,
  value,
  onChange,
  color,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  color: string;
  disabled?: boolean;
}) {
  return (
    <div>
      <div className="mb-2 flex justify-between text-sm">
        <span className="font-medium text-slate-800">{label}</span>
        <span className="text-slate-500">{(value * 100).toFixed(0)}%</span>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="h-2 w-full cursor-pointer appearance-none rounded-full bg-brand-200 disabled:opacity-50"
        style={{ accentColor: color === 'green' ? '#16a34a' : color === 'blue' ? '#0284c7' : '#71717a' }}
      />
    </div>
  );
}

interface PreferencePanelProps {
  disabled?: boolean;
  lockSeed?: boolean;
}

export function PreferencePanel({ disabled = false, lockSeed = false }: PreferencePanelProps) {
  const { config } = useApp();
  const { pref, setYield, setNeff, setWater, applyPreset, setSeed, setWeatherId } = usePreferenceSliders();

  const presets = config?.presets ?? {
    maximum_yield: [1, 0, 0] as [number, number, number],
    n_efficiency: [0, 1, 0],
    water_saver: [0, 0, 1],
    balanced: [1 / 3, 1 / 3, 1 / 3],
  };

  const weatherOptions = config?.weather_options ?? [];
  const selectedWeather = weatherOptions.find((o) => o.id === pref.weather_id) ?? weatherOptions[0];
  const isNasaCityWeather = Boolean(pref.weather_id?.startsWith('wgen-nasa-'));
  const usesSeed = selectedWeather?.mode !== 'measured' || isNasaCityWeather;
  const seedHint = isNasaCityWeather
    ? 'Different seed → different NASA year replayed (2004–2023)'
    : 'Different seed → different synthetic rain pattern';

  const w = preferenceVector(pref);
  const locked = disabled || lockSeed;

  return (
    <>
      <CityClimatePanel disabled={disabled} lockSeed={lockSeed} />

      <Card title="Scenario" className="space-y-3 text-sm text-slate-800">
        <p>
          <span className="text-slate-500">Crop</span> · {config?.crop ?? 'Maize'}
        </p>
        <p>
          <span className="text-slate-500">Location</span> · {config?.location ?? 'DSSAT benchmark'}
        </p>
        <label className="block space-y-1.5">
          <span className="text-slate-500">Weather</span>
          <select
            value={pref.weather_id ?? config?.defaults.weather_id ?? ''}
            disabled={disabled || lockSeed}
            onChange={(e) => setWeatherId(e.target.value)}
            className="w-full rounded-lg border border-brand-200 bg-white px-3 py-2 text-sm text-slate-900 disabled:opacity-50"
          >
            {weatherOptions.length === 0 ? (
              <option value="">Loading weather options…</option>
            ) : (
              weatherOptions.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))
            )}
          </select>
          {selectedWeather?.notes && (
            <span className="block text-xs leading-relaxed text-slate-500">{selectedWeather.notes}</span>
          )}
        </label>
        {usesSeed && (
          <label className="flex flex-wrap items-center gap-3">
            <span className="text-slate-500">Weather seed</span>
            <input
              type="number"
              min={1}
              max={99999}
              value={pref.seed}
              disabled={disabled || lockSeed}
              onChange={(e) => setSeed(parseInt(e.target.value, 10) || 123)}
              className="w-28 rounded-lg border border-brand-200 bg-white px-3 py-1.5 text-sm text-slate-900 disabled:opacity-50"
            />
            <span className="text-xs text-slate-500">{seedHint}</span>
          </label>
        )}
      </Card>

      <Card title="Preference vector" className="space-y-5">
        <p className="font-mono text-sm text-brand-800">{formatPreference(w)}</p>
        <SliderRow label="Yield" value={pref.w_yield} onChange={setYield} color="green" disabled={disabled} />
        <SliderRow label="Nitrogen efficiency" value={pref.w_neff} onChange={setNeff} color="zinc" disabled={disabled} />
        <SliderRow label="Water saving" value={pref.w_water} onChange={setWater} color="blue" disabled={disabled} />
        <div className="flex flex-wrap gap-2 pt-2">
          {(
            [
              ['Maximum Yield', presets.maximum_yield],
              ['Balanced', presets.balanced],
              ['Water Saver', presets.water_saver],
              ['N-Efficiency', presets.n_efficiency],
            ] as const
          ).map(([label, vec]) => (
            <Button key={label} variant="secondary" disabled={locked} onClick={() => applyPreset(vec)}>
              {label}
            </Button>
          ))}
        </div>
      </Card>
    </>
  );
}
