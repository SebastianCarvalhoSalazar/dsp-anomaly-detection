import type { Modality } from '@/api/types';

export const HISTORY_CAPACITY = 300;

export const MODALITIES: { value: Modality; label: string; icon: string }[] = [
  { value: 'audio', label: 'Audio', icon: '🎤' },
  { value: 'image', label: 'Imagen', icon: '🖼️' },
];

export const ANOMALY_THRESHOLD = 0.5;

export const ACCEPTED_AUDIO = ['.wav', '.mp3', '.ogg'];
export const ACCEPTED_IMAGE = ['.jpg', '.jpeg', '.png', '.webp'];
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
