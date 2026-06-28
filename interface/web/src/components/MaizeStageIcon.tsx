/** Simple maize plant sketch scaled by DSSAT v-stage (0–18). */
export function MaizeStageIcon({ vstage, className = 'h-16 w-16' }: { vstage: number; className?: string }) {
  const v = Math.max(0, Math.min(18, vstage));
  const leaves = Math.min(8, Math.max(1, Math.floor(v) || (v > 0 ? 1 : 0)));
  const stemH = 12 + Math.min(v, 14) * 2.2;
  const showTassel = v >= 14;

  return (
    <svg
      viewBox="0 0 48 56"
      className={className}
      aria-hidden
      role="img"
    >
      <ellipse cx="24" cy="52" rx="14" ry="3" fill="#a8a29e" opacity="0.35" />
      <rect x="22" y={52 - stemH} width="4" height={stemH} rx="2" fill="#65a30d" />
      {Array.from({ length: leaves }).map((_, i) => {
        const side = i % 2 === 0 ? -1 : 1;
        const y = 52 - stemH + 8 + i * (stemH / (leaves + 1));
        const len = 10 + Math.min(v, 12) * 0.4;
        return (
          <path
            key={i}
            d={`M24 ${y} Q${24 + side * len} ${y - 4} ${24 + side * (len + 4)} ${y + 2}`}
            fill="none"
            stroke="#84cc16"
            strokeWidth="2.5"
            strokeLinecap="round"
          />
        );
      })}
      {showTassel && (
        <>
          <line x1="24" y1={52 - stemH - 2} x2="24" y2={52 - stemH - 8} stroke="#65a30d" strokeWidth="2" />
          <circle cx="24" cy={52 - stemH - 10} r="3" fill="#eab308" />
        </>
      )}
      {v < 0.5 && (
        <path d="M24 48 Q20 42 24 36 Q28 42 24 48" fill="#84cc16" opacity="0.9" />
      )}
    </svg>
  );
}
