import { ANOMALY_THRESHOLD } from './constants';
import { fmtScore } from './format';

export type DetectorStatus = 'warmup' | 'anomaly' | 'normal';

/**
 * Texto/color para un tile del detector lento, distinguiendo apagado vs
 * calibrando vs con dato (evita el 0.000 ambiguo del warmup).
 */
export function slowTileDisplay(
  enabled: boolean,
  fitted: boolean,
  value: number,
): { text: string; cls: string } {
  if (!enabled) return { text: '— off', cls: 'text-dim' };
  if (!fitted) return { text: 'calibrando', cls: 'text-warning' };
  return { text: fmtScore(value), cls: 'text-ink' };
}

export function deriveStatus(isFitted: boolean, isAnomaly: boolean): DetectorStatus {
  if (!isFitted) return 'warmup';
  return isAnomaly ? 'anomaly' : 'normal';
}

export function scoreColorClass(score: number): string {
  if (score >= 0.65) return 'text-anomaly';
  if (score >= 0.35) return 'text-warning';
  return 'text-normal';
}

export function bigScoreColor(score: number, isFitted: boolean): string {
  if (!isFitted) return 'text-warning';
  return score >= ANOMALY_THRESHOLD ? 'text-anomaly' : 'text-normal';
}

export interface StatusMeta {
  label: string;
  /** Clases del chip (fondo translúcido + texto + borde). */
  chip: string;
  dot: string;
  /** Color de acento (texto) del estado. */
  accent: string;
  /** Sombra de glow del estado. */
  glow: string;
  /** Etiqueta larga para el banner del sistema. */
  system: string;
}

export const STATUS_META: Record<DetectorStatus, StatusMeta> = {
  warmup: {
    label: 'Calentando',
    chip: 'bg-warning/10 text-warning ring-1 ring-warning/30',
    dot: 'bg-warning',
    accent: 'text-warning',
    glow: 'shadow-glow-warning',
    system: 'CALIBRANDO',
  },
  anomaly: {
    label: 'Anomalía',
    chip: 'bg-anomaly/10 text-anomaly ring-1 ring-anomaly/40',
    dot: 'bg-anomaly',
    accent: 'text-anomaly',
    glow: 'shadow-glow-anomaly',
    system: 'ANOMALÍA DETECTADA',
  },
  normal: {
    label: 'Normal',
    chip: 'bg-normal/10 text-normal ring-1 ring-normal/30',
    dot: 'bg-normal',
    accent: 'text-normal',
    glow: 'shadow-glow-normal',
    system: 'SISTEMA NOMINAL',
  },
};
