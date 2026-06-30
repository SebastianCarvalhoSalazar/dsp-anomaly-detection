import { ANOMALY_THRESHOLD } from './constants';

export type DetectorStatus = 'warmup' | 'anomaly' | 'normal';

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

export const STATUS_META: Record<
  DetectorStatus,
  { label: string; chip: string; dot: string }
> = {
  warmup: {
    label: 'Calentando',
    chip: 'bg-amber-100 text-amber-800',
    dot: 'bg-warning',
  },
  anomaly: {
    label: 'ANOMALÍA',
    chip: 'bg-red-100 text-red-700',
    dot: 'bg-anomaly',
  },
  normal: {
    label: 'Normal',
    chip: 'bg-emerald-100 text-emerald-800',
    dot: 'bg-normal',
  },
};
