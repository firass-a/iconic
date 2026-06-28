import type { DayRecord } from '@/types';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { formatDecisionSwfac, formatSwfacReading } from '@/utils/formatAgronomic';

export function MoistureChart({
  days,
  title = 'Water status',
}: {
  days: DayRecord[];
  title?: string;
}) {
  const data = days.map((d) => {
    const moistRaw = d.soil_moisture_before ?? d.soil_moisture;
    const isFc = (d.soil_moisture_fc_before ?? d.soil_moisture_fc) !== false;
    const swfacDec = formatDecisionSwfac(d);
    return {
      day: d.day,
      dap: d.dap,
      swfac: +(swfacDec.value * 100).toFixed(1),
      moisture: isFc ? +(moistRaw * 100).toFixed(1) : null,
      irrig: d.irrigation_mm > 0.1 ? d.irrigation_mm : null,
    };
  });

  if (!data.length) {
    return (
      <div className="dash-card p-5">
        <p className="text-sm font-semibold text-slate-800">{title}</p>
        <p className="mt-4 text-sm text-slate-500">No data yet.</p>
      </div>
    );
  }

  return (
    <div className="dash-card p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm font-semibold text-slate-800">{title}</p>
        <span className="text-xs text-slate-400">
          Green = soil wetness before decision · orange = plant water supply (SWFAC)
        </span>
      </div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
            <XAxis dataKey="day" tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
            <YAxis
              yAxisId="left"
              tick={{ fontSize: 11, fill: '#94a3b8' }}
              axisLine={false}
              tickLine={false}
              domain={[50, 100]}
            />
            <Tooltip
              contentStyle={{
                borderRadius: 12,
                border: '1px solid #e2e8f0',
                fontSize: 12,
                boxShadow: '0 4px 12px rgba(0,0,0,0.06)',
              }}
              formatter={(v, name) => {
                const num = typeof v === 'number' ? v : null;
                if (num == null) return ['—', String(name)];
                if (name === 'swfac') return [`${num}% (${formatSwfacReading(num / 100).status})`, 'SWFAC'];
                if (name === 'moisture') return [`${num}% FC`, 'Moisture before'];
                return [`${num} mm`, 'Irrigation'];
              }}
              labelFormatter={(_, payload) => {
                const row = payload?.[0]?.payload as { day?: number; dap?: number } | undefined;
                if (row?.day != null) return `Season day ${row.day} · DAP ${row.dap ?? '—'}`;
                return '';
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
              formatter={(value) =>
                value === 'swfac' ? 'SWFAC (plant)' : value === 'moisture' ? 'Soil % FC (before)' : 'Irrigation'
              }
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="swfac"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
              connectNulls
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="moisture"
              stroke="#10b981"
              strokeWidth={2}
              strokeDasharray="4 3"
              dot={false}
              activeDot={{ r: 4 }}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
