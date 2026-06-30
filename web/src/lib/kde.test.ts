import { describe, expect, it } from 'vitest';

import { gaussianKde } from './kde';

describe('gaussianKde', () => {
  it('devuelve densidad cero con menos de 2 muestras válidas', () => {
    const out = gaussianKde([0.5]);
    expect(out.every((p) => p.y === 0)).toBe(true);
  });

  it('produce un pico cerca de la media de las muestras', () => {
    const out = gaussianKde([0.6, 0.61, 0.59, 0.6]);
    const peak = out.reduce((a, b) => (b.y > a.y ? b : a));
    expect(peak.x).toBeGreaterThan(0.5);
    expect(peak.x).toBeLessThan(0.7);
  });
});
