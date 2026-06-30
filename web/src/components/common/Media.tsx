import { useState } from 'react';

import { audioUrl, frameUrl } from '@/api/mediaUrls';

export function AudioPlayer({ eventId }: { eventId: number }) {
  return (
    <audio controls preload="none" className="w-full" src={audioUrl(eventId)}>
      Tu navegador no soporta audio.
    </audio>
  );
}

export function AnnotatedFrame({ eventId, alt }: { eventId: number; alt: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <div className="flex h-40 items-center justify-center rounded-xl bg-bg text-sm text-muted">
        Sin frame disponible
      </div>
    );
  }
  return (
    <img
      src={frameUrl(eventId, true)}
      alt={alt}
      loading="lazy"
      onError={() => setFailed(true)}
      className="w-full rounded-xl"
    />
  );
}
