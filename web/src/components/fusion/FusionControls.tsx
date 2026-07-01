import { useEffect, useRef } from 'react';

import type { FusionConfig } from '@/api/types';
import { MetricTile } from '@/components/common/MetricTile';
import { SectionLabel } from '@/components/common/Card';
import { useFusionConfig, useSetFusionConfig } from '@/hooks/useFusionConfig';
import { FUSION_STRATEGIES } from '@/lib/constants';
import { fmtScore } from '@/lib/format';
import { combine, DOMINANT_LABEL } from '@/lib/fusion';
import { useFusionDraft } from '@/store/fusionDraftStore';

function sameConfig(a: FusionConfig, b: FusionConfig): boolean {
  return a.strategy === b.strategy && a.audio_weight === b.audio_weight && a.gates === b.gates;
}

const labelCls = 'mb-1 block text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-muted';

export function FusionControls({
  audioScore,
  videoScore,
}: {
  audioScore: number;
  videoScore: number;
}) {
  const draft = useFusionDraft();
  const { strategy, audio_weight, gates, setStrategy, setAudioWeight, setGates, hydrate } = draft;

  const serverCfg = useFusionConfig();
  const setCfg = useSetFusionConfig();

  const hydrated = useRef(false);
  useEffect(() => {
    if (!hydrated.current && serverCfg.data) {
      hydrate(serverCfg.data);
      hydrated.current = true;
    }
  }, [serverCfg.data, hydrate]);

  const lastSent = useRef<FusionConfig | null>(null);
  useEffect(() => {
    const cfg: FusionConfig = { strategy, audio_weight, gates };
    if (!hydrated.current) return;
    if (lastSent.current && sameConfig(lastSent.current, cfg)) return;
    lastSent.current = cfg;
    setCfg.mutate(cfg);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategy, audio_weight, gates]);

  const preview = combine(strategy, audioScore, videoScore, audio_weight);

  return (
    <section aria-label="Controles de fusión multimodal" className="space-y-4">
      <SectionLabel>Consola de fusión</SectionLabel>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <label className="text-sm">
          <span className={labelCls}>Estrategia</span>
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value as FusionConfig['strategy'])}
            className="w-full rounded-md border border-line bg-surface-2 px-3 py-2 font-mono text-sm text-ink"
          >
            {FUSION_STRATEGIES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
        </label>

        <label className="text-sm">
          <span className={labelCls}>Audio weight · {audio_weight.toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={audio_weight}
            onChange={(e) => setAudioWeight(Number(e.target.value))}
            className="w-full accent-primary"
            disabled={strategy !== 'weighted'}
          />
          <span className="font-mono text-[0.65rem] text-dim">video = 1 − audio</span>
        </label>

        <label className="flex cursor-pointer items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={gates}
            onChange={(e) => setGates(e.target.checked)}
            className="h-4 w-4 accent-primary"
          />
          <span className="font-medium text-muted">La fusión decide anomalías</span>
        </label>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricTile label="Audio" value={fmtScore(audioScore)} valueClass="text-primary" />
        <MetricTile label="Video" value={fmtScore(videoScore)} valueClass="text-primary" />
        <MetricTile label="Combined · preview" value={fmtScore(preview.combinedScore)} />
        <MetricTile label="Dominante" value={DOMINANT_LABEL[preview.dominantModality]} />
      </div>
      {strategy === 'weighted' && (
        <p className="font-mono text-[0.65rem] text-dim">
          combined = {audio_weight.toFixed(2)}·audio + {(1 - audio_weight).toFixed(2)}·video ·
          valor autoritativo desde el backend
        </p>
      )}
    </section>
  );
}
