import type { FusionStrategy } from '@/api/types';
import { ANOMALY_THRESHOLD } from './constants';

export type DominantModality = 'audio-driven' | 'video-driven' | 'multimodal';

export interface FusionResult {
  combinedScore: number;
  isAnomaly: boolean;
  dominantModality: DominantModality;
}

const DOMINANCE_EPS = 0.1;

function dominant(audio: number, video: number): DominantModality {
  if (Math.abs(audio - video) <= DOMINANCE_EPS) return 'multimodal';
  return audio > video ? 'audio-driven' : 'video-driven';
}

/**
 * Reimplementación en TS de las 4 estrategias de fusión del backend
 * (src/fusion/strategies.py). Es un PREVIEW local para feedback instantáneo;
 * el `combined_score` del backend sigue siendo la fuente autoritativa.
 *
 * Verificado por tests de paridad en fusion.test.ts.
 */
export function combine(
  strategy: FusionStrategy,
  audioScore: number,
  videoScore: number,
  audioWeight = 0.5,
  threshold = ANOMALY_THRESHOLD,
): FusionResult {
  let combinedScore: number;
  switch (strategy) {
    case 'weighted':
      combinedScore = audioWeight * audioScore + (1 - audioWeight) * videoScore;
      break;
    case 'max':
      combinedScore = Math.max(audioScore, videoScore);
      break;
    case 'and':
      // Ambas modalidades deben superar el umbral; score = mínimo.
      combinedScore = Math.min(audioScore, videoScore);
      break;
    case 'or':
      // Cualquiera basta; score = máximo.
      combinedScore = Math.max(audioScore, videoScore);
      break;
  }

  let isAnomaly: boolean;
  if (strategy === 'and') {
    isAnomaly = audioScore >= threshold && videoScore >= threshold;
  } else if (strategy === 'or') {
    isAnomaly = audioScore >= threshold || videoScore >= threshold;
  } else {
    isAnomaly = combinedScore >= threshold;
  }

  return { combinedScore, isAnomaly, dominantModality: dominant(audioScore, videoScore) };
}

export const DOMINANT_LABEL: Record<DominantModality, string> = {
  'audio-driven': 'Audio-driven',
  'video-driven': 'Video-driven',
  multimodal: 'Multimodal',
};
