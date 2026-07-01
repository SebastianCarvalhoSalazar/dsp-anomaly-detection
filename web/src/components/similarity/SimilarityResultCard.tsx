import type { SimilarEventResponse } from '@/api/types';
import { AnnotatedFrame, AudioPlayer } from '@/components/common/Media';
import { fmtScore, fmtTimestamp } from '@/lib/format';

export function SimilarityResultCard({ result }: { result: SimilarEventResponse }) {
  const { event, cosine_similarity } = result;
  const strong = cosine_similarity >= 0.7;
  return (
    <div className="rounded-lg border border-line bg-surface p-4 shadow-panel">
      <div className="flex items-start justify-between">
        <div>
          <div className="font-display text-sm font-bold uppercase tracking-wide text-ink">
            EVT·{String(event.id).padStart(4, '0')}
          </div>
          <div className="font-mono text-xs text-muted">{fmtTimestamp(event.timestamp)}</div>
        </div>
        <div className="text-right">
          <div className={`font-mono text-xl font-bold tnum ${strong ? 'text-normal' : 'text-warning'}`}>
            {fmtScore(cosine_similarity)}
          </div>
          <div className="text-[0.6rem] uppercase tracking-[0.14em] text-dim">similitud</div>
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
