import { useState } from 'react';

import { Card, SectionLabel } from '@/components/common/Card';
import { BBoxChip } from '@/components/common/BBoxChip';
import { ConfirmDialog } from '@/components/common/ConfirmDialog';
import { ConnectionIndicator } from '@/components/common/ConnectionIndicator';
import { DriftGauge } from '@/components/common/DriftGauge';
import { MetricTile } from '@/components/common/MetricTile';
import { PageHeader } from '@/components/common/PageHeader';
import { ScoreCard } from '@/components/common/ScoreCard';
import { SystemBanner } from '@/components/common/SystemBanner';
import { KdeChart } from '@/components/charts/KdeChart';
import { RealtimeChart } from '@/components/charts/RealtimeChart';
import { RmsBarChart } from '@/components/charts/RmsBarChart';
import { FusionControls } from '@/components/fusion/FusionControls';
import { useAnomalyStream } from '@/hooks/useAnomalyStream';
import { useResetDetector } from '@/hooks/useFusionConfig';
import { fmtScore, fmtTimestamp } from '@/lib/format';
import { deriveStatus } from '@/lib/status';

export default function LiveMonitor() {
  const stream = useAnomalyStream();
  const msg = stream.lastMessage;
  const resetDetector = useResetDetector();
  const [confirmReset, setConfirmReset] = useState(false);

  const score = msg?.anomaly_score ?? 0;
  const isFitted = msg?.is_fitted ?? false;
  const isAnomaly = msg?.is_anomaly ?? false;
  const status = deriveStatus(isFitted, isAnomaly);
  const lastThreshold = stream.thresholdSeries.at(-1) ?? 0.5;
  const topDrift = msg?.top_drift_features ?? [];
  const topAudio = msg?.top_audio_features ?? [];
  const topVideo = msg?.top_video_features ?? [];

  return (
    <>
      <PageHeader
        title="Monitor en vivo"
        subtitle="Telemetría de anomalías en tiempo real · WebSocket"
        actions={<ConnectionIndicator status={stream.status} onReconnect={stream.reconnectNow} />}
      />

      <SystemBanner status={status}>
        <span>
          WIN <span className="text-ink">#{msg?.window_index ?? 0}</span>
        </span>
        <span>
          REFITS <span className="text-ink">{msg?.refit_count ?? 0}</span>
        </span>
        <span>
          T <span className="text-ink">{fmtTimestamp(msg?.timestamp ?? '')}</span>
        </span>
      </SystemBanner>

      {/* Fila héroe: score dominante + traza + telemetría */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <ScoreCard score={score} isFitted={isFitted} isAnomaly={isAnomaly} />

        <Card className="lg:col-span-2">
          <SectionLabel>Traza · anomaly score / umbral</SectionLabel>
          <RealtimeChart score={stream.scoreSeries} threshold={stream.thresholdSeries} />
        </Card>
      </div>

      {/* Telemetría */}
      <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <DriftGauge auc={msg?.drift_auc ?? 0.5} />
        <MetricTile label="Motion energy" value={fmtScore(msg?.motion_energy ?? 0)} />
        <MetricTile label="Detector" value={isFitted ? 'LISTO' : 'CALIBRANDO'} valueClass={isFitted ? 'text-normal' : 'text-warning'} />
        <MetricTile label="Refit reason" value={(msg?.refit_reason ?? '—').toUpperCase()} />
      </div>
      {topDrift.length > 0 && (
        <p className="mt-2 font-mono text-xs text-warning">
          ⚠ drift features: {topDrift.slice(0, 3).join(' · ')}
        </p>
      )}

      <div className="mt-3 flex gap-2">
        <button
          type="button"
          onClick={stream.clearHistory}
          className="rounded-md border border-line bg-surface px-3 py-2 font-mono text-sm text-muted hover:bg-white/5 hover:text-ink"
        >
          Reiniciar historial
        </button>
        <button
          type="button"
          onClick={() => setConfirmReset(true)}
          className="rounded-md border border-line bg-surface px-3 py-2 font-mono text-sm text-muted hover:bg-white/5 hover:text-ink"
        >
          Reiniciar detector
        </button>
      </div>

      {/* Consola de fusión */}
      <Card className="mt-4">
        <FusionControls audioScore={msg?.audio_score ?? 0} videoScore={msg?.video_score ?? 0} />
        <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">
          <MetricTile label="Audio · rápido" value={fmtScore(msg?.fast_audio_score ?? 0)} />
          <MetricTile label="Audio · lento" value={fmtScore(msg?.slow_audio_score ?? 0)} />
          <MetricTile label="Video · rápido" value={fmtScore(msg?.fast_video_score ?? 0)} />
          <MetricTile label="Video · lento" value={fmtScore(msg?.slow_video_score ?? 0)} />
        </div>
        {(topAudio.length > 0 || topVideo.length > 0) && (
          <p className="mt-2 font-mono text-xs text-muted">
            {topAudio.length > 0 && <span>🔊 {topAudio.join(' · ')}</span>}
            {topAudio.length > 0 && topVideo.length > 0 && '   '}
            {topVideo.length > 0 && <span>🎥 {topVideo.join(' · ')}</span>}
          </p>
        )}
      </Card>

      {/* Señales secundarias */}
      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <SectionLabel>Amplitud RMS</SectionLabel>
          <RmsBarChart rms={stream.rmsSeries} />
        </Card>
        <Card>
          <SectionLabel>Distribución de scores · KDE</SectionLabel>
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
        description="El pipeline reiniciará su detector y volverá a la fase de calibración (warmup). ¿Continuar?"
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
