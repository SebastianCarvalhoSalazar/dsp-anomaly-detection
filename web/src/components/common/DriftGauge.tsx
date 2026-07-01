import { fmtScore } from '@/lib/format';

/** Medidor de drift (C2ST AUC): 0.5 = sin drift, 1.0 = drift total. */
export function DriftGauge({ auc }: { auc: number }) {
  const t = Math.max(0, Math.min(1, (auc - 0.5) / 0.5));
  const tone = auc >= 0.85 ? 'anomaly' : auc >= 0.7 ? 'warning' : 'primary';
  const fill =
    tone === 'anomaly' ? 'bg-anomaly' : tone === 'warning' ? 'bg-warning' : 'bg-primary';
  const text =
    tone === 'anomaly' ? 'text-anomaly' : tone === 'warning' ? 'text-warning' : 'text-primary';

  return (
    <div className="rounded-lg border border-line bg-surface px-4 py-3 shadow-panel">
      <div className="flex items-baseline justify-between">
        <span className="text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-muted">
          Drift AUC
        </span>
        <span className={`font-mono text-xl font-semibold tnum ${text}`}>{fmtScore(auc)}</span>
      </div>
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-line">
        <div
          className={`h-full rounded-full ${fill} transition-[width] duration-500`}
          style={{ width: `${t * 100}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[0.6rem] text-dim">
        <span>0.5</span>
        <span>drift</span>
        <span>1.0</span>
      </div>
    </div>
  );
}
