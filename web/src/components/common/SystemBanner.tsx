import type { ReactNode } from 'react';

import { STATUS_META, type DetectorStatus } from '@/lib/status';

/** Estado global del sistema, imposible de no ver. */
export function SystemBanner({
  status,
  children,
}: {
  status: DetectorStatus;
  children?: ReactNode;
}) {
  const meta = STATUS_META[status];
  return (
    <div
      className={`mb-6 flex items-center justify-between gap-4 rounded-lg border border-line border-l-4 bg-surface px-5 py-3 shadow-panel ${meta.accent} ${meta.glow}`}
      style={{ borderLeftColor: 'currentColor' }}
    >
      <div className="flex items-center gap-3">
        <span className={`h-3 w-3 rounded-full ${meta.dot} animate-pulseDot`} aria-hidden />
        <span className="font-display text-lg font-bold uppercase tracking-[0.16em]">
          {meta.system}
        </span>
      </div>
      {children && (
        <div className="hidden items-center gap-5 font-mono text-xs text-muted sm:flex">
          {children}
        </div>
      )}
    </div>
  );
}
