import {
  createContext,
  useContext,
  useEffect,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from 'react';

import { API_BASE_URL } from '@/api/client';
import type { AnomalyScoreMessage } from '@/api/types';
import { HISTORY_CAPACITY } from '@/lib/constants';
import { RingBuffer } from '@/lib/ringBuffer';

export type StreamStatus = 'connecting' | 'live' | 'stale' | 'reconnecting' | 'closed';

export interface StreamSnapshot {
  status: StreamStatus;
  lastMessage: AnomalyScoreMessage | null;
  lastMessageAt: number | null;
  scoreSeries: number[];
  thresholdSeries: number[];
  rmsSeries: number[];
  reconnectNow: () => void;
  clearHistory: () => void;
}

type Lifecycle = 'connecting' | 'open' | 'reconnecting' | 'closed';

const STALE_THRESHOLD_MS = 5_000;
const HEARTBEAT_MS = 10_000;
const MAX_BACKOFF_MS = 30_000;

function streamUrl(): string {
  return `${API_BASE_URL.replace(/^http/, 'ws')}/ws/stream`;
}

/**
 * Posee un único WebSocket a /ws/stream con:
 *  - reconexión con backoff exponencial + jitter (B2),
 *  - heartbeat (ping periódico),
 *  - detección de staleness por recencia de mensajes (B3),
 *  - ring buffers acotados para las series en tiempo real (B1/B13).
 * Se expone vía useSyncExternalStore para re-render selectivo.
 */
export class AnomalyStreamStore {
  private ws: WebSocket | null = null;
  private lifecycle: Lifecycle = 'closed';
  private running = false;
  private attempt = 0;
  private lastMessage: AnomalyScoreMessage | null = null;
  private lastMessageAt: number | null = null;

  private readonly scoreBuf = new RingBuffer<number>(HISTORY_CAPACITY);
  private readonly threshBuf = new RingBuffer<number>(HISTORY_CAPACITY);
  private readonly rmsBuf = new RingBuffer<number>(HISTORY_CAPACITY);

  private readonly listeners = new Set<() => void>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private stalenessTimer: ReturnType<typeof setInterval> | null = null;

  private snapshot: StreamSnapshot = this.build();

  // ── Ciclo de vida ────────────────────────────────────────────────────────
  start(): void {
    if (this.running) return;
    this.running = true;
    this.connect();
    this.stalenessTimer = setInterval(() => this.tick(), 1_000);
  }

  destroy(): void {
    this.running = false;
    if (this.stalenessTimer) clearInterval(this.stalenessTimer);
    this.clearReconnect();
    this.clearHeartbeat();
    this.closeSocket();
    this.setLifecycle('closed');
  }

  reconnectNow = (): void => {
    this.attempt = 0;
    this.closeSocket();
    this.connect();
  };

  clearHistory = (): void => {
    this.scoreBuf.clear();
    this.threshBuf.clear();
    this.rmsBuf.clear();
    this.rebuild();
  };

  // ── Suscripción (useSyncExternalStore) ───────────────────────────────────
  subscribe = (cb: () => void): (() => void) => {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  };

  getSnapshot = (): StreamSnapshot => this.snapshot;

  // ── Internos ─────────────────────────────────────────────────────────────
  private connect(): void {
    this.clearReconnect();
    this.setLifecycle(this.attempt === 0 ? 'connecting' : 'reconnecting');
    let ws: WebSocket;
    try {
      ws = new WebSocket(streamUrl());
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;
    ws.onopen = () => {
      this.attempt = 0;
      this.setLifecycle('open');
      this.startHeartbeat();
    };
    ws.onmessage = (ev) => this.onMessage(ev);
    ws.onclose = () => {
      this.clearHeartbeat();
      if (!this.running) return;
      this.setLifecycle('reconnecting');
      this.scheduleReconnect();
    };
    ws.onerror = () => {
      try {
        ws.close();
      } catch {
        /* el onclose se encarga del reintento */
      }
    };
  }

  private onMessage(ev: MessageEvent): void {
    let msg: AnomalyScoreMessage;
    try {
      msg = JSON.parse(ev.data as string) as AnomalyScoreMessage;
    } catch {
      return;
    }
    this.lastMessage = msg;
    this.lastMessageAt = Date.now();
    this.scoreBuf.push(msg.anomaly_score ?? 0);
    this.threshBuf.push(msg.adaptive_threshold ?? 0);
    this.rmsBuf.push(msg.rms ?? 0);
    this.rebuild();
  }

  private tick(): void {
    if (this.snapshot.status !== this.effectiveStatus()) this.rebuild();
  }

  private effectiveStatus(): StreamStatus {
    if (this.lifecycle === 'open') {
      const fresh =
        this.lastMessageAt !== null && Date.now() - this.lastMessageAt <= STALE_THRESHOLD_MS;
      return fresh ? 'live' : 'stale';
    }
    return this.lifecycle;
  }

  private build(): StreamSnapshot {
    return {
      status: this.effectiveStatus(),
      lastMessage: this.lastMessage,
      lastMessageAt: this.lastMessageAt,
      scoreSeries: this.scoreBuf.toArray(),
      thresholdSeries: this.threshBuf.toArray(),
      rmsSeries: this.rmsBuf.toArray(),
      reconnectNow: this.reconnectNow,
      clearHistory: this.clearHistory,
    };
  }

  private rebuild(): void {
    this.snapshot = this.build();
    this.listeners.forEach((cb) => cb());
  }

  private setLifecycle(lc: Lifecycle): void {
    if (this.lifecycle === lc) return;
    this.lifecycle = lc;
    this.rebuild();
  }

  private scheduleReconnect(): void {
    const delay = Math.min(1000 * 2 ** this.attempt, MAX_BACKOFF_MS) + Math.random() * 1000;
    this.attempt += 1;
    this.reconnectTimer = setTimeout(() => this.connect(), delay);
  }

  private clearReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private startHeartbeat(): void {
    this.clearHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) this.ws.send('ping');
    }, HEARTBEAT_MS);
  }

  private clearHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private closeSocket(): void {
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.onmessage = null;
      this.ws.onopen = null;
      this.ws.onerror = null;
      try {
        this.ws.close();
      } catch {
        /* noop */
      }
      this.ws = null;
    }
  }
}

const StreamContext = createContext<AnomalyStreamStore | null>(null);

export function AnomalyStreamProvider({ children }: { children: ReactNode }) {
  const [store] = useState(() => new AnomalyStreamStore());
  useEffect(() => {
    store.start();
    return () => store.destroy();
  }, [store]);
  return <StreamContext.Provider value={store}>{children}</StreamContext.Provider>;
}

export function useAnomalyStream(): StreamSnapshot {
  const store = useContext(StreamContext);
  if (!store) throw new Error('useAnomalyStream debe usarse dentro de <AnomalyStreamProvider>');
  return useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
}
