import { useState } from 'react';

import { Card } from '@/components/common/Card';
import { EmptyState, ErrorState, LoadingSkeleton } from '@/components/common/States';
import { MetricTile } from '@/components/common/MetricTile';
import { PageHeader } from '@/components/common/PageHeader';
import { AudioPlayer } from '@/components/common/Media';
import { Heatmap } from '@/components/charts/Heatmap';
import { ImfChart } from '@/components/charts/ImfChart';
import { useEvents } from '@/hooks/useEvents';
import { useOfflineAnalysis } from '@/hooks/useOfflineAnalysis';
import { fmtTimestamp } from '@/lib/format';

type Tab = 'emd' | 'spec';

export default function OfflineAnalysis() {
  const { data: events, isLoading } = useEvents({ limit: 100 });
  const [selected, setSelected] = useState<number | null>(null);
  const [run, setRun] = useState(false);
  const [tab, setTab] = useState<Tab>('emd');

  const id = selected ?? events?.[0]?.id ?? null;
  const analysis = useOfflineAnalysis(id, run);
  const selectedEvent = events?.find((e) => e.id === id) ?? null;

  if (isLoading) return <LoadingSkeleton rows={3} />;
  if (!events || events.length === 0) {
    return (
      <>
        <PageHeader title="Análisis offline" subtitle="Descomposición EMD · Mel-spectrogram" />
        <EmptyState icon="📊">No hay eventos disponibles todavía.</EmptyState>
      </>
    );
  }

  return (
    <>
      <PageHeader
        title="Análisis offline"
        subtitle="Descomposición EMD e IMFs · Mel-spectrogram por evento"
      />

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
                #{e.id} · score {e.anomaly_score.toFixed(3)} · {fmtTimestamp(e.timestamp)}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => setRun(true)}
          className="mt-3 rounded-md bg-primary px-4 py-2 text-sm font-semibold uppercase tracking-wide text-bg hover:brightness-110"
        >
          Ejecutar análisis
        </button>
      </Card>

      {run && analysis.isLoading && (
        <div className="mt-4">
          <LoadingSkeleton rows={3} />
        </div>
      )}
      {run && analysis.isError && (
        <div className="mt-4">
          <ErrorState message={(analysis.error as Error).message} />
        </div>
      )}

      {run && analysis.data && (
        <div className="mt-4 space-y-4">
          <div className="flex gap-2">
            <MetricTile label="IMFs" value={String(analysis.data.n_imfs)} />
            <MetricTile label="Sample rate" value={`${analysis.data.sample_rate} Hz`} />
          </div>

          <div role="tablist" className="flex gap-2">
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'emd'}
              onClick={() => setTab('emd')}
              className={`rounded-lg px-4 py-2 text-sm font-medium ${
                tab === 'emd' ? 'bg-primary font-semibold text-bg' : 'border border-line bg-surface text-muted'
              }`}
            >
              📈 Descomposición EMD
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'spec'}
              onClick={() => setTab('spec')}
              className={`rounded-lg px-4 py-2 text-sm font-medium ${
                tab === 'spec' ? 'bg-primary font-semibold text-bg' : 'border border-line bg-surface text-muted'
              }`}
            >
              🎨 Mel-Spectrogram
            </button>
          </div>

          <Card>
            {tab === 'emd' ? (
              <ImfChart imfs={analysis.data.imfs} sampleRate={analysis.data.sample_rate} />
            ) : (
              <Heatmap z={analysis.data.spectrogram} />
            )}
          </Card>

          {selectedEvent?.has_audio && (
            <Card>
              <AudioPlayer eventId={selectedEvent.id} />
            </Card>
          )}
        </div>
      )}
    </>
  );
}
