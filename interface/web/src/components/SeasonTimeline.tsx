import type { DayRecord } from '@/types';
import { Badge } from '@/components/ui';
import { formatDecisionSwfac, formatGrowthStage, formatSoilMoisture } from '@/utils/formatAgronomic';

export function SeasonTimeline({ days }: { days: DayRecord[] }) {
  const slice = days.slice(-20);

  return (
    <div className="overflow-x-auto panel">
      <table className="w-full min-w-[720px] text-left text-sm">
        <thead className="bg-brand-100 text-xs uppercase text-slate-600">
          <tr>
            <th className="px-4 py-3">Season day</th>
            <th className="px-4 py-3">DAP</th>
            <th className="px-4 py-3">Temp</th>
            <th className="px-4 py-3">Irrigation</th>
            <th className="px-4 py-3">Nitrogen</th>
            <th className="px-4 py-3">Moisture</th>
            <th className="px-4 py-3">Stage (V)</th>
            <th className="px-4 py-3">Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-brand-200">
          {slice.map((d) => {
            const growth = formatGrowthStage(d.growth_stage);
            return (
            <tr key={d.day} className="bg-white">
              <td className="px-4 py-2.5 font-medium text-slate-900">{d.day}</td>
              <td className="px-4 py-2.5 text-slate-600">{d.dap}</td>
              <td className="px-4 py-2.5 text-slate-600">{d.tmax_c.toFixed(0)}°C max</td>
              <td className="px-4 py-2.5 text-slate-800">{d.irrigation_mm.toFixed(1)} mm</td>
              <td className="px-4 py-2.5 text-slate-800">{d.nitrogen_kg_ha.toFixed(1)} kg/ha</td>
              <td className="px-4 py-2.5 text-slate-800">
                {formatSoilMoisture(d).value}
                {d.soil_moisture_fc !== false ? '%' : ''}
              </td>
              <td className="px-4 py-2.5 align-top text-slate-800">
                <span className="font-medium">{growth.label}</span>
                <span className="mt-0.5 block text-xs leading-snug text-slate-500">
                  {growth.explanation}
                </span>
              </td>
              <td className="px-4 py-2.5">
                {(d.swfac_before ?? formatDecisionSwfac(d).value) < 0.85 ? (
                  <Badge tone="amber">Water stress</Badge>
                ) : d.nitrogen_kg_ha > 5 ? (
                  <Badge tone="green">N applied</Badge>
                ) : (
                  <Badge>Stable</Badge>
                )}
              </td>
            </tr>
            );
          })}
        </tbody>
      </table>
      <p className="border-t border-brand-200 px-4 py-2 text-xs text-slate-500">
        Showing last 20 days of {days.length} total
      </p>
    </div>
  );
}
