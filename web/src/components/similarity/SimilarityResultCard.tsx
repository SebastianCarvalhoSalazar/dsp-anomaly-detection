import type { SimilarEventResponse } from '@/api/types';
import { AnnotatedFrame, AudioPlayer } from '@/components/common/Media';
import { fmtScore, fmtTimestamp } from '@/lib/format';

export function SimilarityResultCard({ result }: { result: SimilarEventResponse }) {
  const { event, cosine_similarity } = result;
  const strong = cosine_similarity >= 0.7;
  return (
    <div className="rounded-2xl bg-surface p-4 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-sm font-bold text-ink">Evento #{event.id}</div>
          <div className="text-xs text-muted">{fmtTimestamp(event.timestamp)}</div>
        </div>
        <div className="text-right">
          <div className={`text-xl font-extrabold ${strong ? 'text-normal' : 'text-warning'}`}>
            {fmtScore(cosine_similarity)}
          </div>
          <div className="text-[0.65rem] uppercase tracking-wider text-muted">similitud</div>
        </div>
      </div>
      <div className="mt-3 space-y-2">
        {event.has_frame && (
          <AnnotatedFrame eventId={event.id} alt={`Frame del evento ${event.id}`} />
        )}
        {event.has_audio && <AudioPlayer eventId={event.id} />}
      </div>
    </div>
  );
}
