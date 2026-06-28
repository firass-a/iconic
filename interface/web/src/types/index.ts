export type PreferenceVector = [number, number, number];

export interface WeatherOption {
  id: string;
  label: string;
  mode: 'wgen' | 'measured';
  path?: string | null;
  available: boolean;
  year?: number | null;
  recommended?: boolean;
  notes?: string;
}

export interface CityPreset {
  city_id: string;
  name: string;
  lat: number;
  lon: number;
  weather_id: string;
  built: boolean;
  period: string;
}

export interface SimConfig {
  model: string;
  crop: string;
  location: string;
  objectives: string[];
  preference_labels: Record<string, string>;
  presets: Record<string, PreferenceVector>;
  defaults: { seed: number; weather_id?: string };
  weather_options?: WeatherOption[];
  city_presets?: CityPreset[];
  planting_date?: string;
  training_weather_id?: string;
  training_weather_label?: string;
  agronomic_guards?: boolean;
  read_only_fields: string[];
  notes: string;
}

export interface DayRecord {
  day: number;
  dap: number;
  rain_mm: number;
  tmax_c: number;
  srad: number;
  irrigation_mm: number;
  nitrogen_kg_ha: number;
  soil_moisture: number;
  soil_moisture_fc?: boolean;
  soil_moisture_before?: number;
  soil_moisture_fc_before?: boolean;
  swfac: number;
  swfac_before?: number;
  nstres?: number;
  nstres_before?: number;
  cumulative_irrigation_mm?: number;
  growth_stage: number;
  xlai: number;
  grnwt: number;
  topwt: number;
  n_uptake: number;
  n_uptake_cumulative_kg_ha?: number;
  recommendation: string;
  irrigation_note?: string;
  nitrogen_note?: string;
  agronomic_guard?: string;
}

export interface SeasonSummary {
  yield_kg_ha: number;
  total_n_kg_ha: number;
  total_water_mm: number;
  total_rain_mm: number;
  harvest_index: number;
  ane: number;
  water_productivity: number;
  r_yield: number;
  r_ane: number;
  r_water_eff: number;
  cum_reward: number;
  weather_cli?: string | null;
  weather_id?: string;
  weather_label?: string;
  weather_path?: string | null;
}

export interface SeasonResult {
  preference: PreferenceVector;
  seed: number;
  weather_id?: string;
  weather_label?: string;
  ep_length: number;
  days: DayRecord[];
  summary: SeasonSummary;
  current: DayRecord;
}

export interface SimulationJob {
  id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  preference: PreferenceVector;
  seed: number;
  progress: {
    current_day: number;
    current_dap?: number;
    total_days: number;
    percent: number;
    phase?: 'queued' | 'loading_model' | 'starting_dssat' | 'simulating' | 'done';
  };
  result: SeasonResult | null;
  error: string | null;
}

export interface AdvisorySessionState {
  session_id: string;
  seed: number;
  weather_id?: string;
  weather_label?: string;
  preference: PreferenceVector;
  started: boolean;
  done: boolean;
  day_count: number;
  current: DayRecord | null;
  result: SeasonResult | null;
}

export interface ParetoPoint {
  id: string;
  corner: string;
  episode: number;
  preference: PreferenceVector;
  yield_kg_ha: number;
  total_n_kg_ha: number;
  total_water_mm: number;
  r_yield: number;
  r_ane: number;
  r_water_eff: number;
}

export interface PreferenceInput {
  w_yield: number;
  w_neff: number;
  w_water: number;
  seed: number;
  weather_id?: string;
}
