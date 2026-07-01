import type { ReactNode } from 'react';

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="mb-6 flex flex-col gap-3 border-b border-line pb-4 sm:flex-row sm:items-end sm:justify-between">
      <div className="flex items-center gap-3">
        <span className="h-9 w-1 rounded-sm bg-primary shadow-glow-primary" aria-hidden />
        <div>
          <h1 className="font-display text-2xl font-bold uppercase tracking-[0.06em] text-ink">
            {title}
          </h1>
          {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
        </div>
      </div>
      {actions && <div className="flex items-center gap-3">{actions}</div>}
    </header>
  );
}
