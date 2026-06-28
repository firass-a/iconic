import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react';
import { api, pollSimulation } from '@/services/api';
import type { AdvisorySessionState, PreferenceInput, SeasonResult, SimConfig, SimulationJob } from '@/types';
import { defaultDapInterval } from '@/utils/seasonDays';

export interface DapSelection {
  start: number;
  end: number;
  focus: number;
}

interface AppState {
  config: SimConfig | null;
  preference: PreferenceInput;
  setPreference: (p: Partial<PreferenceInput>) => void;
  replacePreference: (p: PreferenceInput) => void;
  result: SeasonResult | null;
  setResult: (r: SeasonResult | null) => void;
  dapSelection: DapSelection | null;
  setDapSelection: (s: DapSelection) => void;
  loading: boolean;
  error: string | null;
  simJob: SimulationJob | null;
  simRunning: boolean;
  simError: string | null;
  runSimulation: () => Promise<void>;
  clearSimulation: () => void;
  advisorySession: AdvisorySessionState | null;
  advisoryBusy: boolean;
  advisoryError: string | null;
  startAdvisory: () => Promise<void>;
  stepAdvisory: () => Promise<void>;
  endAdvisory: () => Promise<void>;
  refreshLatest: () => Promise<void>;
  refreshConfig: () => Promise<void>;
}

const defaultPref: PreferenceInput = {
  w_yield: 1 / 3,
  w_neff: 1 / 3,
  w_water: 1 / 3,
  seed: 123,
  weather_id: 'wgen-ufga',
};

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<SimConfig | null>(null);
  const [preference, setPref] = useState<PreferenceInput>(defaultPref);
  const [result, setResult] = useState<SeasonResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [simJob, setSimJob] = useState<SimulationJob | null>(null);
  const [simRunning, setSimRunning] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [dapSelection, setDapSelection] = useState<DapSelection | null>(null);
  const [advisorySession, setAdvisorySession] = useState<AdvisorySessionState | null>(null);
  const [advisoryBusy, setAdvisoryBusy] = useState(false);
  const [advisoryError, setAdvisoryError] = useState<string | null>(null);

  const applyResult = useCallback((r: SeasonResult) => {
    setResult(r);
    setDapSelection(defaultDapInterval(r.days));
  }, []);

  const applyAdvisorySnapshot = useCallback((snap: AdvisorySessionState) => {
    setAdvisorySession(snap);
    if (snap.result) {
      setResult(snap.result);
      const day = snap.current?.day ?? snap.result.current?.day;
      if (day) {
        setDapSelection({ start: day, end: day, focus: day });
      }
    }
  }, []);

  const refreshConfig = useCallback(async () => {
    const cfg = await api.getConfig();
    setConfig(cfg);
    if (cfg.defaults.weather_id) {
      setPref((prev) => ({ ...prev, weather_id: prev.weather_id ?? cfg.defaults.weather_id }));
    }
  }, []);

  const refreshLatest = useCallback(async () => {
    try {
      const { result: r } = await api.getLatestResult();
      if (r) applyResult(r);
    } catch {
      /* no prior run */
    }
  }, [applyResult]);

  useEffect(() => {
    (async () => {
      try {
        const cfg = await api.getConfig();
        setConfig(cfg);
        if (cfg.defaults.weather_id) {
          setPref((prev) => ({ ...prev, weather_id: cfg.defaults.weather_id }));
        }
        setError(null);
        await refreshLatest();
      } catch (e) {
        setError(e instanceof Error ? e.message : 'API unavailable — start Docker API');
      } finally {
        setLoading(false);
      }
    })();
  }, [refreshLatest]);

  const setPreference = useCallback((p: Partial<PreferenceInput>) => {
    setPref((prev) => {
      const next = { ...prev, ...p };
      const same =
        prev.w_yield === next.w_yield &&
        prev.w_neff === next.w_neff &&
        prev.w_water === next.w_water &&
        prev.seed === next.seed &&
        prev.weather_id === next.weather_id;
      return same ? prev : next;
    });
  }, []);

  const replacePreference = useCallback((p: PreferenceInput) => {
    setPref(p);
  }, []);

  const clearSimulation = useCallback(() => {
    setSimJob(null);
    setSimError(null);
  }, []);

  const runSimulation = useCallback(async () => {
    if (simRunning) return;
    setSimError(null);
    setSimRunning(true);
    setSimJob(null);
    try {
      const { job_id } = await api.startSimulation(preference);
      const final = await pollSimulation(job_id, setSimJob);
      if (final.status === 'failed') {
        throw new Error(final.error ?? 'Simulation failed');
      }
      if (final.result) applyResult(final.result);
      setAdvisorySession(null);
    } catch (e) {
      setSimError(e instanceof Error ? e.message : 'Run failed');
    } finally {
      setSimRunning(false);
    }
  }, [preference, simRunning, applyResult]);

  const endAdvisory = useCallback(async () => {
    const id = advisorySession?.session_id;
    if (id) {
      try {
        await api.closeAdvisorySession(id);
      } catch {
        /* session may already be gone */
      }
    }
    setAdvisorySession(null);
  }, [advisorySession?.session_id]);

  const startAdvisory = useCallback(async () => {
    if (advisoryBusy || simRunning) return;
    setAdvisoryError(null);
    setAdvisoryBusy(true);
    try {
      if (advisorySession?.session_id) {
        await api.closeAdvisorySession(advisorySession.session_id);
      }
      const snap = await api.startAdvisorySession(preference);
      setResult(null);
      setDapSelection(null);
      setAdvisorySession(snap);
    } catch (e) {
      setAdvisoryError(
        e instanceof Error && e.message.includes('Not Found')
          ? 'Advisory API not loaded — restart the API container (Ctrl+C, then .\\run_api_docker.ps1)'
          : e instanceof Error
            ? e.message
            : 'Could not start advisory session',
      );
    } finally {
      setAdvisoryBusy(false);
    }
  }, [advisoryBusy, simRunning, preference, advisorySession?.session_id]);

  const stepAdvisory = useCallback(async () => {
    const id = advisorySession?.session_id;
    if (!id || advisoryBusy || advisorySession.done) return;
    setAdvisoryError(null);
    setAdvisoryBusy(true);
    try {
      const snap = await api.stepAdvisorySession(id);
      applyAdvisorySnapshot(snap);
    } catch (e) {
      setAdvisoryError(e instanceof Error ? e.message : 'Step failed');
    } finally {
      setAdvisoryBusy(false);
    }
  }, [advisorySession, advisoryBusy, applyAdvisorySnapshot]);

  return (
    <AppContext.Provider
      value={{
        config,
        preference,
        setPreference,
        replacePreference,
        result,
        setResult,
        dapSelection,
        setDapSelection,
        loading,
        error,
        simJob,
        simRunning,
        simError,
        runSimulation,
        clearSimulation,
        advisorySession,
        advisoryBusy,
        advisoryError,
        startAdvisory,
        stepAdvisory,
        endAdvisory,
        refreshLatest,
        refreshConfig,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}
