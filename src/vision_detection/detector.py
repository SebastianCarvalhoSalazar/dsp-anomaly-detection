"""Independent video anomaly detector.

Reuses the shared :class:`BaseAnomalyDetector` (IsolationForest + warmup +
buffer + refits + drift), so the video modality gets the same *correct*
machinery as audio without duplication. Two deliberate differences from the
audio detector:

  - **No PCA** (ADR-0006): the video feature vector is ~7-dimensional, so
    dimensionality reduction adds nothing and can only discard signal. The
    IsolationForest consumes the features directly.
  - A smaller default buffer, since motion statistics are lower-dimensional
    and the scene baseline shifts faster than the acoustic one.
"""

from __future__ import annotations

from typing import Optional

from src.detection import DetectorConfig
from src.detection.base import BaseAnomalyDetector

from .types import VIDEO_FEATURE_NAMES


def default_video_config() -> DetectorConfig:
    """A DetectorConfig tuned for low-dimensional motion features."""
    return DetectorConfig(
        buffer_size=300,
        refit_every=150,
        n_estimators=100,
        enable_pca=False,  # ADR-0006
        enable_drift_detection=True,
        enable_adaptive_threshold=True,
        state_path="data/video_detector_state.pkl",
    )


class VideoAnomalyDetector(BaseAnomalyDetector):
    """Online motion-based anomaly detector (IsolationForest, no PCA)."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        super().__init__(
            config=config or default_video_config(),
            feature_names=list(VIDEO_FEATURE_NAMES),
        )
