import type { BoundingBox } from '@/api/types';
import { fmtScore } from '@/lib/format';

export function BBoxChip({ boxes }: { boxes: BoundingBox[] }) {
  if (!boxes || boxes.length === 0) return null;
  const best = boxes.reduce((a, b) => (b.source_score > a.source_score ? b : a));
  return (
    <div className="rounded-xl border-l-4 border-anomaly bg-surface p-4 shadow-sm">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">
        Fuente probable
      </div>
      <span className="rounded-md bg-red-100 px-3 py-1 text-sm font-bold text-red-700">
        ({best.x},{best.y}) {best.w}×{best.h} — score {fmtScore(best.source_score)}
      </span>
    </div>
  );
}
