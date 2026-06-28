import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import { api } from '@/services/api';
import { useApp } from '@/hooks/useAppContext';
import { formatPreference, preferenceVector } from '@/hooks/usePreferenceSliders';
import type { ParetoPoint } from '@/types';

export function TradeoffsPage() {
  const { preference, result } = useApp();
  const [points, setPoints] = useState<ParetoPoint[]>([]);

  useEffect(() => {
    api.getPareto().then((d) => setPoints(d.points)).catch(() => setPoints([]));
  }, []);

  const highlight = result
    ? {
        yield_kg_ha: result.summary.yield_kg_ha,
        total_water_mm: result.summary.total_water_mm,
        total_n_kg_ha: result.summary.total_n_kg_ha,
      }
    : null;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <div>
        <h2 className="page-title">Trade-offs</h2>
        <p className="page-subtitle">
          Pareto frontier from CAPQL v2 evaluation (real DSSAT runs). Your latest simulation is highlighted.
        </p>
        <p className="mt-1 font-mono text-xs text-slate-500">
          Current {formatPreference(preferenceVector(preference))}
        </p>
      </div>

      <div className="panel p-5">
        <p className="mb-4 text-sm font-medium text-slate-900">Yield vs total irrigation water</p>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#adc9ba" />
              <XAxis type="number" dataKey="total_water_mm" name="Water" unit=" mm" tick={{ fontSize: 11 }} />
              <YAxis type="number" dataKey="yield_kg_ha" name="Yield" unit=" kg/ha" tick={{ fontSize: 11 }} />
              <ZAxis type="number" dataKey="total_n_kg_ha" range={[40, 400]} name="N" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter name="Eval runs" data={points} fill="#86efac" fillOpacity={0.8} />
              {highlight && (
                <Scatter
                  name="Your run"
                  data={[highlight]}
                  fill="#16a34a"
                  shape="star"
                />
              )}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="panel p-5">
        <p className="mb-4 text-sm font-medium text-slate-900">Yield vs total nitrogen</p>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#adc9ba" />
              <XAxis type="number" dataKey="total_n_kg_ha" name="N" unit=" kg/ha" tick={{ fontSize: 11 }} />
              <YAxis type="number" dataKey="yield_kg_ha" name="Yield" tick={{ fontSize: 11 }} />
              <Tooltip />
              <Scatter data={points} fill="#86efac" fillOpacity={0.8} />
              {highlight && <Scatter data={[highlight]} fill="#16a34a" />}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </motion.div>
  );
}
