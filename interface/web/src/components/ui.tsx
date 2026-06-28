import type { ReactNode } from 'react';

export function Card({
  children,
  className = '',
  title,
  icon,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
  icon?: ReactNode;
}) {
  return (
    <div className={`panel p-5 ${className}`}>
      {title && (
        <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-800">
          {icon}
          {title}
        </div>
      )}
      {!title && icon}
      {children}
    </div>
  );
}

export function StatCard({
  label,
  value,
  unit,
  icon,
  sub,
}: {
  label: string;
  value: string | number;
  unit?: string;
  icon: ReactNode;
  sub?: string;
}) {
  return (
    <Card>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-slate-600">{label}</p>
          <p className="mt-1 text-2xl font-semibold tracking-tight text-slate-900">
            {value}
            {unit && <span className="ml-1 text-base font-normal text-slate-500">{unit}</span>}
          </p>
          {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
        </div>
        <div className="rounded-xl border border-brand-200 bg-brand-50 p-2.5 text-brand-700">{icon}</div>
      </div>
    </Card>
  );
}

export function MetricTile({
  label,
  value,
  unit,
  hint,
}: {
  label: string;
  value: string | number;
  unit?: string;
  hint?: string;
}) {
  return (
    <div className="rounded-xl border border-brand-200 bg-white px-4 py-3 shadow-sm shadow-slate-200/40">
      <p className="text-label">{label}</p>
      <p className="mt-1 text-xl font-semibold text-slate-900">
        {value}
        {unit && <span className="ml-1 text-sm font-medium text-slate-600">{unit}</span>}
      </p>
      {hint && <p className="mt-1.5 text-xs leading-relaxed text-slate-500">{hint}</p>}
    </div>
  );
}

export function Badge({
  children,
  tone = 'default',
}: {
  children: ReactNode;
  tone?: 'default' | 'green' | 'blue' | 'amber';
}) {
  const tones = {
    default: 'bg-brand-100 text-brand-900 border border-brand-200',
    green: 'bg-brand-100 text-brand-900 border border-brand-200',
    blue: 'bg-sky-100 text-sky-900 border border-sky-200',
    amber: 'bg-amber-100 text-amber-900 border border-amber-200',
  };
  return (
    <span className={`inline-flex rounded-full px-2.5 py-0.5 text-xs font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function ProgressBar({
  value,
  max = 1,
  label,
}: {
  value: number;
  max?: number;
  label?: string;
}) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div>
      {label && (
        <div className="mb-1 flex justify-between text-xs text-slate-600">
          <span>{label}</span>
          <span className="font-medium text-slate-800">{pct.toFixed(0)}%</span>
        </div>
      )}
      <div className="h-2 overflow-hidden rounded-full bg-brand-100">
        <div
          className="h-full rounded-full bg-gradient-to-r from-brand-600 to-brand-800 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = 'primary',
  disabled,
  className = '',
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'ghost';
  disabled?: boolean;
  className?: string;
}) {
  const variants = {
    primary:
      'bg-emerald-600 text-white hover:bg-emerald-700 shadow-sm shadow-slate-400/20 disabled:opacity-50',
    secondary: 'border border-slate-200 bg-white text-slate-800 hover:bg-slate-50',
    ghost: 'text-slate-700 hover:bg-slate-100',
  };
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={`inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-colors ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  );
}
