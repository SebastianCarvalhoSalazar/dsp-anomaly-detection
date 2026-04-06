from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class EventResponse(BaseModel):
    id: int
    timestamp: datetime
    anomaly_score: float
    event_dir: str
    faiss_index_id: Optional[int]
    has_audio: bool
    has_frame: bool
    has_embedding: bool

    model_config = {"from_attributes": True}


class SimilarEventResponse(BaseModel):
    event: EventResponse
    cosine_similarity: float


class AnomalyScoreMessage(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    is_fitted: bool
    timestamp: str        # ISO format
    window_index: int
    bounding_boxes: list[dict]
    motion_energy: float = 0.0  # motion area ratio [0,1]
    rms: float = 0.0  # RMS amplitude of the audio window
    # Drift detection metrics
    adaptive_threshold: float = 0.0
    score_mean: float = 0.0
    feature_mean_drift: float = 0.0
    refit_count: int = 0


class OfflineAnalysisResponse(BaseModel):
    imfs: list[list[float]]          # list of IMF arrays (time series)
    n_imfs: int
    sample_rate: int
    spectrogram: list[list[float]]   # mel-spectrogram as nested list (freq x time)
