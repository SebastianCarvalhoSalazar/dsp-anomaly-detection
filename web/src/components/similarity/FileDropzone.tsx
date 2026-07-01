import { useRef, useState } from 'react';

import { MAX_UPLOAD_BYTES } from '@/lib/constants';
import { fmtBytes } from '@/lib/format';

export function FileDropzone({
  accept,
  file,
  onFile,
}: {
  accept: string[];
  file: File | null;
  onFile: (file: File | null) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  const handle = (f: File | null) => {
    setError(null);
    if (f && f.size > MAX_UPLOAD_BYTES) {
      setError(`El archivo (${fmtBytes(f.size)}) supera el límite de 10 MB.`);
      onFile(null);
      return;
    }
    onFile(f);
  };

  return (
    <div>
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="flex w-full flex-col items-center gap-1 rounded-lg border-2 border-dashed border-line bg-surface-2 px-4 py-8 font-mono text-sm text-muted transition-colors hover:border-primary/60"
      >
        <span className="text-2xl" aria-hidden>
          ⬆
        </span>
        {file ? (
          <span className="text-ink">
            {file.name} · {fmtBytes(file.size)}
          </span>
        ) : (
          <span>clic para elegir archivo · {accept.join(' ')}</span>
        )}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={accept.join(',')}
        className="sr-only"
        onChange={(e) => handle(e.target.files?.[0] ?? null)}
      />
      {error && (
        <p className="mt-2 text-sm text-anomaly" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
