import { useState } from 'react';

import type { EventResponse } from '@/api/types';
import { Heatmap } from '@/components/charts/Heatmap';
import { AnnotatedFrame, AudioPlayer } from '@/components/common/Media';
import { ErrorState, LoadingSkeleton } from '@/components/common/States';
import { useSpectrogram } from '@/hooks/useSpectrogram';
import { fmtScore, fmtTimestamp } from '@/lib/format';
import { scoreColorClass } from '@/lib/status';

export function EventCard({
  event,
  onDelete,
}: {
  event: EventResponse;
  onDelete: (id: number) => void;
}) {
  const [showSpec, setShowSpec] = useState(false);
  const spec = useSpectrogram(event.id, showSpec && event.has_audio);
  return (
    <div className="rounded-lg border border-line bg-surface p-5 shadow-panel transition-colors hover:border-primary/40">
      <div className="flex items-start justify-between">
        <div>
          <div className="font-display text-sm font-bold uppercase tracking-wide text-ink">
            EVT·{String(event.id).padStart(4, '0')}
          </div>
          <div className="font-mono text-xs text-muted">{fmtTimestamp(event.timestamp)}</div>
        </div>
        <div className={`font-mono text-2xl font-bold tnum ${scoreColorClass(event.anomaly_score)}`}>
          {fmtScore(event.anomaly_score)}
        </div>
      </div>

      <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
        {event.has_frame && (
          <AnnotatedFrame eventId={event.id} alt={`Frame del evento ${event.id}`} />
        )}
        {event.has_audio && (
          <div className="flex items-center">
            <AudioPlayer eventId={event.id} />
          </div>
        )}
      </div>

      {showSpec && (
        <div className="mt-3">
          {spec.isLoading && <LoadingSkeleton rows={2} />}
          {spec.isError && <ErrorState message={(spec.error as Error).message} />}
          {spec.data && <Heatmap z={spec.data.spectrogram} height={160} />}
        </div>
      )}

      <div className="mt-4 flex items-center justify-between font-mono text-xs">
        <span className="text-dim">embedding {event.has_embedding ? '✓' : '—'}</span>
        {event.has_audio && (
          <button
            type="button"
            aria-expanded={showSpec}
            onClick={() => setShowSpec((v) => !v)}
            className="rounded-md px-3 py-1.5 font-semibold uppercase tracking-wide text-primary hover:bg-primary/10"
          >
            {showSpec ? 'Ocultar espectrograma' : 'Espectrograma'}
          </button>
        )}
        <button
          type="button"
          onClick={() => onDelete(event.id)}
          className="rounded-md px-3 py-1.5 font-semibold uppercase tracking-wide text-anomaly hover:bg-anomaly/10"
        >
          Eliminar
        </button>
      </div>
    </div>
  );
}
