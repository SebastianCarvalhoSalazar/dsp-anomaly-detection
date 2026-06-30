import { fmtScore } from '@/lib/format';
import { bigScoreColor, deriveStatus } from '@/lib/status';
import { StatusChip } from './StatusChip';

export function ScoreCard({
  score,
  isFitted,
  isAnomaly,
}: {
  score: number;
  isFitted: boolean;
  isAnomaly: boolean;
}) {
  const status = deriveStatus(isFitted, isAnomaly);
  return (
    <div className="rounded-2xl bg-surface p-8 text-center shadow-sm">
      <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted">
        Anomaly Score
      </div>
      <div
        className={`text-7xl font-extrabold leading-none tracking-tight ${bigScoreColor(score, isFitted)}`}
        aria-live="polite"
      >
        {fmtScore(score)}
      </div>
      <div className="mt-4 flex justify-center">
        <StatusChip status={status} />
      </div>
    </div>
  );
}
