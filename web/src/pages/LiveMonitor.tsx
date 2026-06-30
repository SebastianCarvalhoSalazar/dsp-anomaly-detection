import { useState } from 'react';

import { Card, SectionLabel } from '@/components/common/Card';
import { BBoxChip } from '@/components/common/BBoxChip';
import { ConfirmDialog } from '@/components/common/ConfirmDialog';
import { ConnectionIndicator } from '@/components/common/ConnectionIndicator';
import { MetricTile } from '@/components/common/MetricTile';
import { PageHeader } from '@/components/common/PageHeader';
import { ScoreCard } from '@/components/common/ScoreCard';
import { KdeChart } from '@/components/charts/KdeChart';
import { RealtimeChart } from '@/components/charts/RealtimeChart';
import { RmsBarChart } from '@/components/charts/RmsBarChart';
import { FusionControls } from '@/components/fusion/FusionControls';
import { useAnomalyStream } from '@/hooks/useAnomalyStream';
import { useResetDetector } from '@/hooks/useFusionConfig';
import { fmtScore, fmtTimestamp } from '@/lib/format';

export default function LiveMonitor() {
  const stream = useAnomalyStream();
  const msg = stream.lastMessage;
  const resetDetector = useResetDetector();
  const [confirmReset, setConfirmReset] = useState(false);

  const score = msg?.anomaly_score ?? 0;
  const isFitted = msg?.is_fitted ?? false;
  const isAnomaly = msg?.is_anomaly ?? false;
  const lastThreshold = stream.thresholdSeries.at(-1) ?? 0.5;
  const topDrift = msg?.top_drift_features ?? [];
  const topAudio = msg?.top_audio_features ?? [];
  const topVideo = msg?.top_video_features ?? [];

  return (
    <>
      <PageHeader
        title="Monitor en vivo"
        subtitle="Scores de anomalía en tiempo real vía WebSocket"
        actions={
          <ConnectionIndicator status={stream.status} onReconnect={stream.reconnectNow} />
        }
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <ScoreCard score={score} isFitted={isFitted} isAnomaly={isAnomaly} />
        </div>

        <div className="lg:col-span-2">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <MetricTile label="Ventana" value={`#${msg?.window_index ?? 0}`} />
            <MetricTile label="Última detección" value={fmtTimestamp(msg?.timestamp ?? '')} />
            <MetricTile label="Detector" value={isFitted ? 'Listo ✓' : 'Calentando…'} />
            <MetricTile
              label="Motion energy"
              value={fmtScore(msg?.motion_energy ?? 0)}
            />
            <MetricTile label="Refits" value={String(msg?.refit_count ?? 0)} />
            <MetricTile
              label="Drift AUC"
              value={fmtScore(msg?.drift_auc ?? 0.5)}
              help="C2ST: 0.5 = sin drift, 1.0 = drift total"
            />
          </div>
          {topDrift.length > 0 && (
            <p className="mt-2 text-xs text-warning">⚠️ Top drift: {topDrift.slice(0, 3).join(', ')}</p>
          )}
          <div className="mt-3 flex gap-2">
            <button
              type="button"
              onClick={stream.clearHistory}
              className="rounded-lg border border-line bg-surface px-3 py-2 text-sm font-medium hover:bg-bg"
            >
              Reiniciar historial
            </button>
            <button
              type="button"
              onClick={() => setConfirmReset(true)}
              className="rounded-lg border border-line bg-surface px-3 py-2 text-sm font-medium hover:bg-bg"
            >
              Reiniciar detector
            </button>
          </div>
        </div>
      </div>

      <Card className="mt-4">
        <FusionControls
          audioScore={msg?.audio_score ?? 0}
          videoScore={msg?.video_score ?? 0}
        />
        <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
          <MetricTile label="Audio rápido" value={fmtScore(msg?.fast_audio_score ?? 0)} />
          <MetricTile label="Audio lento" value={fmtScore(msg?.slow_audio_score ?? 0)} />
          <MetricTile label="Video rápido" value={fmtScore(msg?.fast_video_score ?? 0)} />
          <MetricTile label="Video lento" value={fmtScore(msg?.slow_video_score ?? 0)} />
        </div>
        {(topAudio.length > 0 || topVideo.length > 0) && (
          <p className="mt-2 text-xs text-muted">
            Top contributors —{' '}
            {topAudio.length > 0 && <span>🔊 {topAudio.join(', ')}</span>}
            {topAudio.length > 0 && topVideo.length > 0 && '  ·  '}
            {topVideo.length > 0 && <span>🎥 {topVideo.join(', ')}</span>}
          </p>
        )}
      </Card>

      <Card className="mt-4">
        <SectionLabel>Historial de anomaly score</SectionLabel>
        <RealtimeChart score={stream.scoreSeries} threshold={stream.thresholdSeries} />
      </Card>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <SectionLabel>Amplitud RMS</SectionLabel>
          <RmsBarChart rms={stream.rmsSeries} />
        </Card>
        <Card>
          <SectionLabel>Distribución de scores (KDE)</SectionLabel>
          <KdeChart scores={stream.scoreSeries} threshold={lastThreshold} />
        </Card>
      </div>

      {msg?.bounding_boxes && msg.bounding_boxes.length > 0 && (
        <div className="mt-4">
          <BBoxChip boxes={msg.bounding_boxes} />
        </div>
      )}

      <ConfirmDialog
        open={confirmReset}
        title="Reiniciar detector"
        description="El pipeline reiniciará su detector y volverá a la fase de warmup. ¿Continuar?"
        confirmLabel="Reiniciar"
        onCancel={() => setConfirmReset(false)}
        onConfirm={() => {
          resetDetector.mutate();
          setConfirmReset(false);
        }}
      />
    </>
  );
}
