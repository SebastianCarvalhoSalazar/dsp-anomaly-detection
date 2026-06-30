import type { StreamStatus } from '@/hooks/useAnomalyStream';

const META: Record<StreamStatus, { label: string; dot: string; text: string }> = {
  connecting: { label: 'Conectando…', dot: 'bg-warning', text: 'text-warning' },
  live: { label: 'En vivo', dot: 'bg-normal', text: 'text-normal' },
  stale: { label: 'Sin datos', dot: 'bg-warning', text: 'text-warning' },
  reconnecting: { label: 'Reconectando…', dot: 'bg-warning', text: 'text-warning' },
  closed: { label: 'Desconectado', dot: 'bg-muted', text: 'text-muted' },
};

export function ConnectionIndicator({
  status,
  onReconnect,
}: {
  status: StreamStatus;
  onReconnect?: () => void;
}) {
  const m = META[status];
  const showReconnect = status === 'closed' || status === 'stale';
  return (
    <div className="inline-flex items-center gap-2 text-sm" role="status">
      <span className={`h-2.5 w-2.5 rounded-full ${m.dot}`} aria-hidden />
      <span className={`font-medium ${m.text}`}>{m.label}</span>
      {showReconnect && onReconnect && (
        <button
          type="button"
          onClick={onReconnect}
          className="ml-1 rounded-md px-2 py-0.5 text-xs font-medium text-primary hover:bg-primary/10"
        >
          Reconectar
        </button>
      )}
    </div>
  );
}
