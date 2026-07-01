import { useState } from 'react';

import type { Modality, SimilarEventResponse } from '@/api/types';
import { EmptyState, ErrorState, LoadingSkeleton } from '@/components/common/States';
import { PageHeader } from '@/components/common/PageHeader';
import { Card } from '@/components/common/Card';
import { FileDropzone } from '@/components/similarity/FileDropzone';
import { SimilarityResultCard } from '@/components/similarity/SimilarityResultCard';
import { useEvents } from '@/hooks/useEvents';
import { useSimilarByEvent, useSimilarUpload } from '@/hooks/useSimilarity';
import { ACCEPTED_AUDIO, ACCEPTED_IMAGE, MODALITIES } from '@/lib/constants';
import { fmtTimestamp } from '@/lib/format';

type Mode = 'upload' | 'event';

export default function SimilaritySearch() {
  const [mode, setMode] = useState<Mode>('upload');
  const [k, setK] = useState(5);

  return (
    <>
      <PageHeader
        title="Búsqueda por similitud"
        subtitle="Encuentra eventos similares usando embeddings multimodales (Wav2Vec2 + DINOv2)"
      />

      <div className="mb-4 flex flex-wrap items-center gap-4">
        <div role="tablist" aria-label="Modo de búsqueda" className="flex gap-2">
          <TabButton active={mode === 'upload'} onClick={() => setMode('upload')}>
            📁 Subir archivo
          </TabButton>
          <TabButton active={mode === 'event'} onClick={() => setMode('event')}>
            📋 Evento existente
          </TabButton>
        </div>
        <label className="text-sm">
          <span className="mr-2 font-medium text-muted">Resultados (k): {k}</span>
          <input
            type="range"
            min={1}
            max={20}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            className="align-middle accent-primary"
          />
        </label>
      </div>

      {mode === 'upload' ? <UploadMode k={k} /> : <EventMode k={k} />}
    </>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={`rounded-lg px-4 py-2 text-sm font-medium ${
        active ? 'bg-primary font-semibold text-bg' : 'border border-line bg-surface text-muted hover:text-ink'
      }`}
    >
      {children}
    </button>
  );
}

function Results({ results }: { results: SimilarEventResponse[] }) {
  if (results.length === 0) {
    return <EmptyState icon="🔍">No se encontraron resultados. El índice puede estar vacío.</EmptyState>;
  }
  return (
    <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {results.map((r) => (
        <SimilarityResultCard key={r.event.id} result={r} />
      ))}
    </div>
  );
}

function UploadMode({ k }: { k: number }) {
  const [modality, setModality] = useState<Modality>('audio');
  const [file, setFile] = useState<File | null>(null);
  const search = useSimilarUpload();
  const accept = modality === 'audio' ? ACCEPTED_AUDIO : ACCEPTED_IMAGE;

  return (
    <>
      <Card>
        <div className="mb-3 flex gap-2">
          {MODALITIES.map((m) => (
            <button
              key={m.value}
              type="button"
              onClick={() => {
                setModality(m.value);
                setFile(null);
              }}
              className={`rounded-lg px-3 py-1.5 text-sm ${
                modality === m.value ? 'bg-primary font-semibold text-bg' : 'border border-line bg-surface-2 text-muted'
              }`}
            >
              {m.icon} {m.label}
            </button>
          ))}
        </div>
        <FileDropzone accept={accept} file={file} onFile={setFile} />
        <button
          type="button"
          disabled={!file || search.isPending}
          onClick={() => file && search.mutate({ file, modality, k })}
          className="mt-3 rounded-md bg-primary px-4 py-2 text-sm font-semibold uppercase tracking-wide text-bg hover:brightness-110 disabled:opacity-50"
        >
          Buscar eventos similares
        </button>
      </Card>

      {search.isPending && (
        <p className="mt-4 text-sm text-muted">
          Codificando y buscando… la primera búsqueda carga los modelos (~60 s).
        </p>
      )}
      {search.isError && (
        <div className="mt-4">
          <ErrorState message={(search.error as Error).message} />
        </div>
      )}
      {search.data && <Results results={search.data} />}
    </>
  );
}

function EventMode({ k }: { k: number }) {
  const { data: events, isLoading } = useEvents({ limit: 50 });
  const [selected, setSelected] = useState<number | null>(null);
  const [run, setRun] = useState(false);
  const id = selected ?? events?.[0]?.id ?? null;

  const search = useSimilarByEvent(id, k, run);

  if (isLoading) return <LoadingSkeleton rows={2} />;
  if (!events || events.length === 0) {
    return <EmptyState icon="📋">No hay eventos guardados todavía.</EmptyState>;
  }

  return (
    <>
      <Card>
        <label className="text-sm">
          <span className="mb-1 block font-medium text-muted">Selecciona un evento</span>
          <select
            value={id ?? ''}
            onChange={(e) => {
              setSelected(Number(e.target.value));
              setRun(false);
            }}
            className="w-full rounded-lg border border-line bg-surface px-3 py-2"
          >
            {events.map((e) => (
              <option key={e.id} value={e.id}>
                #{e.id} — {fmtTimestamp(e.timestamp)} (score {e.anomaly_score.toFixed(3)})
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => setRun(true)}
          className="mt-3 rounded-md bg-primary px-4 py-2 text-sm font-semibold uppercase tracking-wide text-bg hover:brightness-110"
        >
          Buscar eventos similares
        </button>
      </Card>

      {search.isFetching && (
        <div className="mt-4">
          <LoadingSkeleton rows={2} />
        </div>
      )}
      {search.isError && (
        <div className="mt-4">
          <ErrorState message={(search.error as Error).message} />
        </div>
      )}
      {run && search.data && <Results results={search.data} />}
    </>
  );
}
