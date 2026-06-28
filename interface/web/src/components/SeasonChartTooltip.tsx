import type { DayRecord } from '@/types';
import { formatSeasonDayLabel } from '@/utils/seasonDays';

/** Recharts tooltip label — season day + DAP so charts match the season log. */
export function seasonChartTooltipLabel(
  _: unknown,
  payload: ReadonlyArray<{ payload?: DayRecord }>,
): string {
  const day = payload[0]?.payload;
  return day ? formatSeasonDayLabel(day) : '';
}
