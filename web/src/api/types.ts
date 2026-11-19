// Tipos espejo de src/api/schemas.py. Ver docs/frontend-migration/03-contrato-api-frontend.md.

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
  // Segmentación de eventos (ADR-0014) — null en eventos antiguos
  end_timestamp: string | null;
  duration_s: number | null;
}

export interface SimilarEventResponse {
  event: EventResponse;
  cosine_similarity: number;
}

export interface SpectrogramResponse {
  sample_rate: number;
  spectrogram: number[][]; // mel en dB (bandas × frames)
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
  top_audio_features: string[];
}
