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
    # Multimodal fields (v0.3) — None for events written by older versions
    audio_score: Optional[float] = None
    video_score: Optional[float] = None
    combined_score: Optional[float] = None
    dominant_modality: Optional[str] = None

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
    drift_auc: float = 0.5
    top_drift_features: list[str] = []
    refit_count: int = 0
    refit_reason: str = "scheduled"
    # -- Multimodal scores (v0.3) ------------------------------------------
    # Defaults keep older dashboard clients working. In single-modality
    # operation audio_score == combined_score == anomaly_score.
    audio_score: float = 0.0
    video_score: float = 0.0
    combined_score: float = 0.0
    fast_audio_score: float = 0.0
    slow_audio_score: float = 0.0
    fast_video_score: float = 0.0
    slow_video_score: float = 0.0
    top_audio_features: list[str] = []
    top_video_features: list[str] = []
    dominant_modality: str = "audio"


class FusionConfigMessage(BaseModel):
    """Live fusion configuration driven from the dashboard and polled by the
    pipeline. ``strategy`` is one of weighted|max|and|or; ``gates`` toggles
    whether the fused score drives the event-gating decision."""

    strategy: str = "weighted"
    audio_weight: float = 0.5
    gates: bool = False


class OfflineAnalysisResponse(BaseModel):
    imfs: list[list[float]]          # list of IMF arrays (time series)
    n_imfs: int
    sample_rate: int
    spectrogram: list[list[float]]   # mel-spectrogram as nested list (freq x time)
