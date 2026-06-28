import { useCallback } from 'react';
import { useApp } from '@/hooks/useAppContext';
import type { PreferenceInput } from '@/types';

/** Preference sliders — single source of truth in AppContext. */
export function usePreferenceSliders() {
  const { preference, replacePreference } = useApp();

  const setYield = useCallback(
    (w_yield: number) => {
      const p = preference;
      const rest = Math.max(0.001, 1 - w_yield);
      const ratio = p.w_neff + p.w_water || 1;
      replacePreference({
        ...p,
        w_yield,
        w_neff: (p.w_neff / ratio) * rest,
        w_water: (p.w_water / ratio) * rest,
      });
    },
    [preference, replacePreference],
  );

  const setNeff = useCallback(
    (w_neff: number) => {
      const p = preference;
      const rest = Math.max(0.001, 1 - w_neff);
      const ratio = p.w_yield + p.w_water || 1;
      replacePreference({
        ...p,
        w_neff,
        w_yield: (p.w_yield / ratio) * rest,
        w_water: (p.w_water / ratio) * rest,
      });
    },
    [preference, replacePreference],
  );

  const setWater = useCallback(
    (w_water: number) => {
      const p = preference;
      const rest = Math.max(0.001, 1 - w_water);
      const ratio = p.w_yield + p.w_neff || 1;
      replacePreference({
        ...p,
        w_water,
        w_yield: (p.w_yield / ratio) * rest,
        w_neff: (p.w_neff / ratio) * rest,
      });
    },
    [preference, replacePreference],
  );

  const applyPreset = useCallback(
    (w: [number, number, number]) => {
      replacePreference({
        ...preference,
        w_yield: w[0],
        w_neff: w[1],
        w_water: w[2],
      });
    },
    [preference, replacePreference],
  );

  const setSeed = useCallback(
    (seed: number) => {
      replacePreference({ ...preference, seed: Math.max(1, Math.min(99999, seed)) });
    },
    [preference, replacePreference],
  );

  const setWeatherId = useCallback(
    (weather_id: string) => {
      replacePreference({ ...preference, weather_id });
    },
    [preference, replacePreference],
  );

  return { pref: preference, setYield, setNeff, setWater, setSeed, setWeatherId, applyPreset };
}
export function formatPreference(w: [number, number, number]) {
  return `w = [${w.map((x) => x.toFixed(2)).join(', ')}]`;
}

export function preferenceVector(p: PreferenceInput): [number, number, number] {
  return [p.w_yield, p.w_neff, p.w_water];
}
