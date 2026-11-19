import { describe, expect, it } from 'vitest';

import { deriveStatus } from './status';

describe('deriveStatus', () => {
  it('warmup cuando el detector no está entrenado', () => {
    expect(deriveStatus(false, true)).toBe('warmup');
    expect(deriveStatus(false, false)).toBe('warmup');
  });

  it('anomaly / normal cuando está entrenado', () => {
    expect(deriveStatus(true, true)).toBe('anomaly');
    expect(deriveStatus(true, false)).toBe('normal');
  });
});
