// Tipos espejo de src/api/schemas.py. Ver docs/frontend-migration/03-contrato-api-frontend.md.

export type FusionStrategy = 'weighted' | 'max' | 'and' | 'or';
export type Modality = 'audio' | 'image';

export interface EventResponse {
  id: number;
  timestamp: string; // ISO 8601
  anomaly_score: number;
  event_dir: string;
  faiss_index_id: number | null;
  has_audio: boolean;
  has_frame: boolean;
  has_embedding: boolean;
  // Multimodal (v0.3) — null en eventos de versiones antiguas
  audio_score: number | null;
  video_score: number | null;
  combined_score: number | null;
  dominant_modality: string | null;
}

export interface SimilarEventResponse {
  event: EventResponse;
  cosine_similarity: number;
}

export interface OfflineAnalysisResponse {
  imfs: number[][];
  n_imfs: number;
  sample_rate: number;
  spectrogram: number[][];
}

export interface FusionConfig {
  strategy: FusionStrategy;
  audio_weight: number; // 0..1
  gates: boolean;
}

export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
  source_score: number;
}

export interface AnomalyScoreMessage {
  anomaly_score: number;
  is_anomaly: boolean;
  is_fitted: boolean;
  timestamp: string;
  window_index: number;
  bounding_boxes: BoundingBox[];
  motion_energy: number;
  rms: number;
  adaptive_threshold: number;
  score_mean: number;
  drift_auc: number;
  top_drift_features: string[];
  refit_count: number;
  refit_reason: string;
  audio_score: number;
  video_score: number;
  combined_score: number;
  fast_audio_score: number;
  slow_audio_score: number;
  fast_video_score: number;
  slow_video_score: number;
  slow_enabled: boolean;
  slow_audio_fitted: boolean;
  slow_video_fitted: boolean;
  top_audio_features: string[];
  top_video_features: string[];
  dominant_modality: string;
}
