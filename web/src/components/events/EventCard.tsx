import type { EventResponse } from '@/api/types';
import { AnnotatedFrame, AudioPlayer } from '@/components/common/Media';
import { fmtScore, fmtTimestamp } from '@/lib/format';
import { scoreColorClass } from '@/lib/status';

export function EventCard({
  event,
  onDelete,
}: {
  event: EventResponse;
  onDelete: (id: number) => void;
}) {
  return (
    <div className="rounded-2xl bg-surface p-5 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-bold text-ink">Evento #{event.id}</div>
          <div className="text-xs text-muted">{fmtTimestamp(event.timestamp)}</div>
        </div>
        <div className={`text-2xl font-extrabold ${scoreColorClass(event.anomaly_score)}`}>
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

      <div className="mt-4 flex items-center justify-between">
        <span className="text-xs text-muted">
          embedding {event.has_embedding ? '✓' : '—'}
        </span>
        <button
          type="button"
          onClick={() => onDelete(event.id)}
          className="rounded-lg px-3 py-1.5 text-sm font-medium text-anomaly hover:bg-red-50"
        >
          Eliminar
        </button>
      </div>
    </div>
  );
}
