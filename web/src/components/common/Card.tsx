import type { ReactNode } from 'react';

export function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`rounded-lg border border-line bg-surface p-4 shadow-panel sm:p-5 ${className}`}
    >
      {children}
    </div>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="mb-3 flex items-center gap-2 text-[0.65rem] font-semibold uppercase tracking-[0.16em] text-muted">
      <span className="inline-block h-1.5 w-1.5 rounded-[1px] bg-primary shadow-glow-primary" />
      {children}
    </div>
  );
}
