import type { ReactNode } from 'react';

export function EmptyState({ icon = '∅', children }: { icon?: string; children: ReactNode }) {
  return (
    <div className="py-12 text-center text-muted">
      <div className="mb-2 text-5xl" aria-hidden>
        {icon}
      </div>
      <p className="text-sm">{children}</p>
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="rounded-xl border border-anomaly/30 bg-red-50 p-4 text-sm text-red-700" role="alert">
      <p className="font-medium">Ocurrió un error</p>
      <p className="mt-1 break-words">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded-md bg-anomaly px-3 py-1 text-xs font-semibold text-white hover:bg-red-600"
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
        <div key={i} className="h-16 animate-pulse rounded-xl bg-line/60" />
      ))}
    </div>
  );
}
