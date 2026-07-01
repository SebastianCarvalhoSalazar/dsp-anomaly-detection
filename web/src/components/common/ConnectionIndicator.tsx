import type { StreamStatus } from '@/hooks/useAnomalyStream';

const META: Record<StreamStatus, { label: string; dot: string; text: string }> = {
  connecting: { label: 'CONECTANDO', dot: 'bg-warning', text: 'text-warning' },
  live: { label: 'EN VIVO', dot: 'bg-normal', text: 'text-normal' },
  stale: { label: 'SIN DATOS', dot: 'bg-warning', text: 'text-warning' },
  reconnecting: { label: 'RECONECTANDO', dot: 'bg-warning', text: 'text-warning' },
  closed: { label: 'DESCONECTADO', dot: 'bg-dim', text: 'text-muted' },
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
  const pulse = status === 'live' ? 'animate-pulseDot' : '';
  return (
    <div
      className="inline-flex items-center gap-2 rounded-md border border-line bg-surface-2 px-2.5 py-1 font-mono text-xs"
      role="status"
    >
      <span className={`h-2 w-2 rounded-full ${m.dot} ${pulse}`} aria-hidden />
      <span className={`font-semibold tracking-wider ${m.text}`}>{m.label}</span>
      {showReconnect && onReconnect && (
        <button
          type="button"
          onClick={onReconnect}
          className="ml-1 rounded px-2 py-0.5 text-[0.7rem] font-semibold text-primary hover:bg-primary/10"
        >
          RECONECTAR
        </button>
      )}
    </div>
  );
}
