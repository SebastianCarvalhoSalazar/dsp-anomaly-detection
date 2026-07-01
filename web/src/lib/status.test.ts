import { describe, expect, it } from 'vitest';

import { deriveStatus, slowTileDisplay } from './status';

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

describe('slowTileDisplay', () => {
  it('apagado cuando el detector lento no está habilitado', () => {
    expect(slowTileDisplay(false, false, 0).text).toBe('— off');
  });

  it('calibrando cuando está habilitado pero no entrenado', () => {
    expect(slowTileDisplay(true, false, 0).text).toBe('calibrando');
  });

  it('muestra el score cuando está habilitado y entrenado', () => {
    expect(slowTileDisplay(true, true, 0.42).text).toBe('0.420');
  });
});
