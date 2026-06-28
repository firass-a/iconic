import { motion } from 'framer-motion';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useApp } from '@/hooks/useAppContext';
import { seasonChartTooltipLabel } from '@/components/SeasonChartTooltip';
import { SeasonStatusBanner } from '@/components/SeasonStatusBanner';
import { Card, ProgressBar, StatCard } from '@/components/ui';
import { SeasonTimeline } from '@/components/SeasonTimeline';
import { BarChart3, Droplets, FlaskConical, Wheat } from 'lucide-react';
import { isSeasonComplete } from '@/utils/seasonDays';

export function ReportsPage() {
  const { result, advisorySession, advisoryError } = useApp();

  if (!result) {
    return <p className="text-sm text-muted">Run a simulation to generate season reports.</p>;
  }

  const s = result.summary;
  const days = result.days;
  const cur = result.current;
  const complete = isSeasonComplete(result, { advisoryDone: advisorySession?.done });

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <div>
        <h2 className="page-title">Reports</h2>
        <p className="page-subtitle">
          End-of-season KPIs from your CAPQL + DSSAT run. Charts use{' '}
          <strong className="font-medium text-slate-700">season day</strong> (simulation step) — not
          DAP (crop age).
        </p>
      </div>

      <SeasonStatusBanner
        result={result}
        advisoryDone={advisorySession?.done}
        advisoryError={advisoryError}
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <StatCard
          label={complete ? 'Expected yield' : 'Yield (harvest)'}
          value={complete ? s.yield_kg_ha.toLocaleString() : '—'}
          unit={complete ? 'kg/ha' : undefined}
          sub={complete ? undefined : 'Run to harvest (~day 165) for final yield'}
          icon={<Wheat className="h-5 w-5" />}
        />
        <StatCard label="Total water (irrigation)" value={s.total_water_mm.toFixed(0)} unit="mm" icon={<Droplets className="h-5 w-5" />} />
        <StatCard label="Total nitrogen" value={s.total_n_kg_ha.toFixed(0)} unit="kg/ha" icon={<FlaskConical className="h-5 w-5" />} />
        <StatCard label="Harvest index" value={s.harvest_index.toFixed(3)} icon={<BarChart3 className="h-5 w-5" />} />
        <StatCard label="N efficiency (ANE)" value={s.ane.toFixed(1)} unit="kg/kg" icon={<FlaskConical className="h-5 w-5" />} />
        <StatCard label="Water productivity" value={s.water_productivity.toFixed(1)} unit="kg/mm" icon={<Droplets className="h-5 w-5" />} />
      </div>

      <Card title="Crop status (final day)">
        <div className="space-y-4">
          <ProgressBar label="Soil moisture" value={cur.soil_moisture} />
          <ProgressBar label="Leaf area index" value={cur.xlai} max={7} />
          <ProgressBar label="Grain fill progress" value={cur.grnwt} max={12000} />
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Daily soil moisture">
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={days}>
                <CartesianGrid strokeDasharray="3 3" stroke="#adc9ba" />
                <XAxis dataKey="day" tick={{ fontSize: 10 }} label={{ value: 'Season day', position: 'insideBottom', offset: -2, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} domain={[0, 1]} />
                <Tooltip labelFormatter={seasonChartTooltipLabel} />
                <Area type="monotone" dataKey="soil_moisture" stroke="#0284c7" fill="#0284c722" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Daily irrigation">
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={days}>
                <CartesianGrid strokeDasharray="3 3" stroke="#adc9ba" />
                <XAxis dataKey="day" tick={{ fontSize: 10 }} label={{ value: 'Season day', position: 'insideBottom', offset: -2, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip labelFormatter={seasonChartTooltipLabel} />
                <Line type="stepAfter" dataKey="irrigation_mm" stroke="#16a34a" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Daily nitrogen">
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={days}>
                <XAxis dataKey="day" tick={{ fontSize: 10 }} label={{ value: 'Season day', position: 'insideBottom', offset: -2, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip labelFormatter={seasonChartTooltipLabel} />
                <Line type="stepAfter" dataKey="nitrogen_kg_ha" stroke="#15803d" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card title="Biomass & yield progress">
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={days}>
                <XAxis dataKey="day" tick={{ fontSize: 10 }} label={{ value: 'Season day', position: 'insideBottom', offset: -2, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip labelFormatter={seasonChartTooltipLabel} />
                <Line dataKey="topwt" stroke="#86efac" dot={false} name="Biomass" />
                <Line dataKey="grnwt" stroke="#16a34a" dot={false} name="Grain" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <div>
        <h3 className="mb-3 text-sm font-medium text-slate-800">Season timeline</h3>
        <SeasonTimeline days={days} />
      </div>
    </motion.div>
  );
}
