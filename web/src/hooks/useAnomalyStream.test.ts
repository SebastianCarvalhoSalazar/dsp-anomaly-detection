import { Server } from 'mock-socket';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AnomalyStreamStore } from './useAnomalyStream';

const URL = 'ws://localhost:8000/ws/stream';

const sampleMessage = {
  anomaly_score: 0.42,
  is_anomaly: false,
  is_fitted: true,
  timestamp: '2026-06-30T12:00:00',
  window_index: 7,
  bounding_boxes: [],
  motion_energy: 0.1,
  rms: 0.05,
  adaptive_threshold: 0.5,
  drift_auc: 0.6,
};

let server: Server | null = null;

afterEach(() => {
  server?.stop();
  server = null;
});

describe('AnomalyStreamStore', () => {
  it('transiciona a "live" y acumula la serie al recibir un mensaje', async () => {
    server = new Server(URL);
    server.on('connection', (socket) => {
      socket.send(JSON.stringify(sampleMessage));
    });

    const store = new AnomalyStreamStore();
    store.start();

    await vi.waitFor(() => {
      expect(store.getSnapshot().status).toBe('live');
    });

    const snap = store.getSnapshot();
    expect(snap.lastMessage?.window_index).toBe(7);
    expect(snap.scoreSeries.at(-1)).toBeCloseTo(0.42);
    expect(snap.thresholdSeries.at(-1)).toBeCloseTo(0.5);

    store.destroy();
    expect(store.getSnapshot().status).toBe('closed');
  });
});
