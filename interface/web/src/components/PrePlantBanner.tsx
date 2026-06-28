import { Sprout } from 'lucide-react';

interface PrePlantBannerProps {
  seasonDay: number;
  plantingSimDay: number;
  daysUntilPlanting: number;
}

export function PrePlantBanner({ seasonDay, plantingSimDay, daysUntilPlanting }: PrePlantBannerProps) {
  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
      <div className="flex gap-2">
        <Sprout className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
        <div>
          <p className="font-medium">Pre-plant period</p>
          <p className="mt-1 text-amber-900/80">
            Season day {seasonDay} is before crop activity (planting at season day {plantingSimDay}).
            {daysUntilPlanting > 0 &&
              ` About ${daysUntilPlanting} simulation day${daysUntilPlanting === 1 ? '' : 's'} until DAP 1.`}{' '}
            Field browse skips these days once the crop is established.
          </p>
        </div>
      </div>
    </div>
  );
}
