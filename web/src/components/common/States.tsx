import type { ReactNode } from 'react';

export function EmptyState({ icon = '∅', children }: { icon?: string; children: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-line py-14 text-center text-muted">
      <div className="mb-3 text-5xl opacity-70" aria-hidden>
        {icon}
      </div>
      <p className="mx-auto max-w-sm text-sm">{children}</p>
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      className="rounded-lg border border-anomaly/40 bg-anomaly/10 p-4 text-sm text-anomaly"
      role="alert"
    >
      <p className="font-display font-bold uppercase tracking-wide">Error</p>
      <p className="mt-1 break-words text-anomaly/90">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded-md bg-anomaly px-3 py-1 text-xs font-bold uppercase tracking-wide text-bg hover:brightness-110"
        >
          Reintentar
        </button>
      )}
    </div>
  );
}

export function LoadingSkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-3" aria-busy="true" aria-label="Cargando">
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="h-16 animate-pulse rounded-lg border border-line bg-surface/60"
        />
      ))}
    </div>
  );
}
