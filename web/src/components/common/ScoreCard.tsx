import { fmtScore } from '@/lib/format';
import { deriveStatus, STATUS_META } from '@/lib/status';
import { StatusChip } from './StatusChip';

export function ScoreCard({
  score,
  isFitted,
  isAnomaly,
  label = 'Anomaly Score',
}: {
  score: number;
  isFitted: boolean;
  isAnomaly: boolean;
  label?: string;
}) {
  const status = deriveStatus(isFitted, isAnomaly);
  const meta = STATUS_META[status];
  return (
    <div
      className={`relative overflow-hidden rounded-lg border border-line bg-surface p-6 ${meta.glow}`}
    >
      <div className="flex items-center justify-between">
        <span className="text-[0.62rem] font-semibold uppercase tracking-[0.18em] text-muted">
          {label}
        </span>
        <StatusChip status={status} />
      </div>

      <div
        className={`mt-5 font-mono text-[5.5rem] font-bold leading-none tnum ${meta.accent} ${
          isAnomaly ? 'animate-glowPulse' : ''
        }`}
      >
        {fmtScore(score)}
      </div>

      {/* Barrido tipo osciloscopio bajo el número. */}
      <div className={`relative mt-5 h-px w-full overflow-hidden bg-line ${meta.accent}`}>
        <div
          className="absolute inset-y-0 w-1/3 animate-sweep"
          style={{
            background: 'linear-gradient(90deg, transparent, currentColor, transparent)',
          }}
        />
      </div>
    </div>
  );
}
