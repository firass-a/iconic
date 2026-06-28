import type {
  AdvisorySessionState,
  ParetoPoint,
  PreferenceInput,
  SeasonResult,
  SimConfig,
  SimulationJob,
} from '@/types';

const BASE = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ ok: boolean }>('/health'),

  getConfig: () => request<SimConfig>('/config'),

  buildCityClimate: (cityId: string) =>
    request<{
      ok: boolean;
      city_id: string;
      weather_id: string;
      meta: Record<string, unknown>;
      weather_options: SimConfig['weather_options'];
    }>(`/weather/cities/${cityId}/build`, { method: 'POST' }),

  getPareto: () => request<{ points: ParetoPoint[] }>('/pareto'),

  startSimulation: (preference: PreferenceInput) =>
    request<{ job_id: string }>('/simulate', {
      method: 'POST',
      body: JSON.stringify({ preference }),
    }),

  getSimulation: (jobId: string) => request<SimulationJob>(`/simulate/${jobId}`),

  getLatestResult: () =>
    request<{ result: SeasonResult | null; job_id?: string }>('/simulate/latest/result'),

  startAdvisorySession: (preference: PreferenceInput) =>
    request<AdvisorySessionState>('/advisory/session', {
      method: 'POST',
      body: JSON.stringify({ preference }),
    }),

  getAdvisorySession: (sessionId: string) =>
    request<AdvisorySessionState>(`/advisory/session/${sessionId}`),

  stepAdvisorySession: (sessionId: string) =>
    request<AdvisorySessionState>(`/advisory/session/${sessionId}/step`, {
      method: 'POST',
    }),

  closeAdvisorySession: (sessionId: string) =>
    request<{ ok: boolean }>(`/advisory/session/${sessionId}`, {
      method: 'DELETE',
    }),
};

export async function pollSimulation(
  jobId: string,
  onProgress?: (job: SimulationJob) => void,
  intervalMs = 1500,
): Promise<SimulationJob> {
  for (;;) {
    const job = await api.getSimulation(jobId);
    onProgress?.(job);
    if (job.status === 'completed' || job.status === 'failed') {
      return job;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}
