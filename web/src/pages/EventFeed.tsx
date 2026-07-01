import { useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';

import { EmptyState, ErrorState, LoadingSkeleton } from '@/components/common/States';
import { ConfirmDialog } from '@/components/common/ConfirmDialog';
import { MetricTile } from '@/components/common/MetricTile';
import { PageHeader } from '@/components/common/PageHeader';
import { EventCard } from '@/components/events/EventCard';
import { EventFilters, type FilterValues } from '@/components/events/EventFilters';
import { useDeleteAllEvents, useDeleteEvent, useEvents } from '@/hooks/useEvents';
import { ANOMALY_THRESHOLD } from '@/lib/constants';
import { fmtScore } from '@/lib/format';

export default function EventFeed() {
  const [sp, setSp] = useSearchParams();
  const filters: FilterValues = {
    minScore: Number(sp.get('min') ?? 0),
    limit: Number(sp.get('limit') ?? 25),
    sort: (sp.get('sort') as FilterValues['sort']) ?? 'recent',
  };

  const patch = (p: Partial<FilterValues>) => {
    const next = new URLSearchParams(sp);
    if (p.minScore !== undefined) next.set('min', String(p.minScore));
    if (p.limit !== undefined) next.set('limit', String(p.limit));
    if (p.sort !== undefined) next.set('sort', p.sort);
    setSp(next, { replace: true });
  };

  const { data, isLoading, isError, error, refetch } = useEvents({
    limit: filters.limit,
    min_score: filters.minScore,
  });
  const delEvent = useDeleteEvent();
  const delAll = useDeleteAllEvents();

  const [toDelete, setToDelete] = useState<number | null>(null);
  const [confirmAll, setConfirmAll] = useState(false);

  const events = useMemo(() => {
    const list = data ?? [];
    return filters.sort === 'score'
      ? [...list].sort((a, b) => b.anomaly_score - a.anomaly_score)
      : list;
  }, [data, filters.sort]);

  const summary = useMemo(() => {
    if (events.length === 0) return null;
    const anomalies = events.filter((e) => e.anomaly_score >= ANOMALY_THRESHOLD).length;
    const avg = events.reduce((s, e) => s + e.anomaly_score, 0) / events.length;
    const max = Math.max(...events.map((e) => e.anomaly_score));
    return { total: events.length, anomalies, avg, max };
  }, [events]);

  return (
    <>
      <PageHeader
        title="Eventos detectados"
        subtitle="Historial de anomalías con audio y evidencia visual"
      />

      <EventFilters values={filters} onChange={patch} onClearAll={() => setConfirmAll(true)} />

      {isLoading && (
        <div className="mt-4">
          <LoadingSkeleton rows={4} />
        </div>
      )}
      {isError && (
        <div className="mt-4">
          <ErrorState message={(error as Error).message} onRetry={() => refetch()} />
        </div>
      )}

      {!isLoading && !isError && summary && (
        <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
          <MetricTile label="Total eventos" value={String(summary.total)} />
          <MetricTile label="Anomalías" value={String(summary.anomalies)} />
          <MetricTile label="Score promedio" value={fmtScore(summary.avg)} />
          <MetricTile label="Score máximo" value={fmtScore(summary.max)} />
        </div>
      )}

      {!isLoading && !isError && events.length === 0 && (
        <EmptyState icon="🎙️">
          No hay eventos todavía. Ejecuta el pipeline y genera algún sonido cerca del micrófono.
        </EmptyState>
      )}

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        {events.map((e) => (
          <EventCard key={e.id} event={e} onDelete={setToDelete} />
        ))}
      </div>

      <ConfirmDialog
        open={toDelete !== null}
        title="Eliminar evento"
        description={`Se eliminará el evento #${toDelete} (DB + archivos). Esta acción no se puede deshacer.`}
        confirmLabel="Eliminar"
        danger
        onCancel={() => setToDelete(null)}
        onConfirm={() => {
          if (toDelete !== null) delEvent.mutate(toDelete);
          setToDelete(null);
        }}
      />
      <ConfirmDialog
        open={confirmAll}
        title="Borrar todos los eventos"
        description="Se eliminarán TODOS los eventos y se reseteará el índice FAISS. Esta acción no se puede deshacer."
        confirmLabel="Borrar todo"
        danger
        onCancel={() => setConfirmAll(false)}
        onConfirm={() => {
          delAll.mutate();
          setConfirmAll(false);
        }}
      />
    </>
  );
}
