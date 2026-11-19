import type { BoundingBox } from '@/api/types';
import { fmtScore } from '@/lib/format';

/** Caja de movimiento principal del frame (área × novedad; no indica la fuente del sonido). */
export function BBoxChip({ boxes }: { boxes: BoundingBox[] }) {
  if (!boxes || boxes.length === 0) return null;
  const best = boxes.reduce((a, b) => (b.motion_score > a.motion_score ? b : a));
  return (
    <div className="rounded-lg border border-line border-l-4 border-l-anomaly bg-surface p-4 shadow-panel">
      <div className="mb-2 text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-muted">
        Movimiento principal
      </div>
      <span className="rounded-md bg-anomaly/15 px-3 py-1 font-mono text-sm font-semibold text-anomaly">
        ({best.x},{best.y}) {best.w}×{best.h} · mov {fmtScore(best.motion_score)}
      </span>
    </div>
  );
}
