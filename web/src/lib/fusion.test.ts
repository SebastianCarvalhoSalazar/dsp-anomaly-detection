import { describe, expect, it } from 'vitest';

import { combine } from './fusion';

// Paridad con las 4 estrategias del backend (src/fusion/strategies.py, ADR-0005).
describe('combine (paridad de fusión)', () => {
  it('weighted: combinación lineal por audio_weight', () => {
    expect(combine('weighted', 0.8, 0.2, 0.5).combinedScore).toBeCloseTo(0.5);
    expect(combine('weighted', 0.8, 0.2, 1).combinedScore).toBeCloseTo(0.8);
    expect(combine('weighted', 0.8, 0.2, 0).combinedScore).toBeCloseTo(0.2);
  });

  it('max: máximo de las modalidades', () => {
    expect(combine('max', 0.3, 0.7).combinedScore).toBe(0.7);
  });

  it('and: score = mínimo; anomalía solo si ambas superan el umbral', () => {
    expect(combine('and', 0.9, 0.4).combinedScore).toBe(0.4);
    expect(combine('and', 0.9, 0.4).isAnomaly).toBe(false);
    expect(combine('and', 0.9, 0.6).isAnomaly).toBe(true);
  });

  it('or: score = máximo; anomalía si cualquiera supera el umbral', () => {
    expect(combine('or', 0.9, 0.1).combinedScore).toBe(0.9);
    expect(combine('or', 0.9, 0.1).isAnomaly).toBe(true);
    expect(combine('or', 0.2, 0.1).isAnomaly).toBe(false);
  });

  it('dominante: multimodal cuando los scores están cerca', () => {
    expect(combine('weighted', 0.5, 0.52).dominantModality).toBe('multimodal');
    expect(combine('weighted', 0.9, 0.2).dominantModality).toBe('audio-driven');
    expect(combine('weighted', 0.2, 0.9).dominantModality).toBe('video-driven');
  });
});
