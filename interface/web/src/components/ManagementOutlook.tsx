import { ArrowRight, Droplets, FlaskConical, Lightbulb } from 'lucide-react';
import type { ManagementHint } from '@/utils/managementInsights';

interface ManagementOutlookProps {
  hints: ManagementHint[];
}

function HintIcon({ kind }: { kind: ManagementHint['kind'] }) {
  if (kind === 'irrigation') return <Droplets className="h-4 w-4 text-sky-500" />;
  if (kind === 'nitrogen') return <FlaskConical className="h-4 w-4 text-violet-500" />;
  return <Lightbulb className="h-4 w-4 text-amber-500" />;
}

export function ManagementOutlook({ hints }: ManagementOutlookProps) {
  if (!hints.length) return null;

  return (
    <div className="rounded-xl border border-emerald-100 bg-emerald-50/60 p-4">
      <div className="flex items-center gap-2 text-sm font-medium text-emerald-900">
        <ArrowRight className="h-4 w-4" />
        What&apos;s next
      </div>
      <ul className="mt-2 space-y-2">
        {hints.map((hint, i) => (
          <li key={i} className="flex gap-2 text-sm text-emerald-900/90">
            <HintIcon kind={hint.kind} />
            <span>
              {hint.message}
              {hint.source === 'heuristic' && (
                <span className="ml-1 text-xs text-emerald-700/70">(estimate)</span>
              )}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
