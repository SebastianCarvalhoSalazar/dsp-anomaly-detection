import type { FusionStrategy, Modality } from '@/api/types';

export const HISTORY_CAPACITY = 300;

export const FUSION_STRATEGIES: { value: FusionStrategy; label: string }[] = [
  { value: 'weighted', label: 'Weighted Average' },
  { value: 'max', label: 'Maximum' },
  { value: 'and', label: 'AND' },
  { value: 'or', label: 'OR' },
];

export const MODALITIES: { value: Modality; label: string; icon: string }[] = [
  { value: 'audio', label: 'Audio', icon: '🎤' },
  { value: 'image', label: 'Imagen', icon: '🖼️' },
];

export const ANOMALY_THRESHOLD = 0.5;

export const ACCEPTED_AUDIO = ['.wav', '.mp3', '.ogg'];
export const ACCEPTED_IMAGE = ['.jpg', '.jpeg', '.png', '.webp'];
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
