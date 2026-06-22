"""Feature representation for the independent video anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Order matters: this is the vector layout fed to the IsolationForest and the
# label list used for drift / explainability output.
VIDEO_FEATURE_NAMES: list[str] = [
    "motion_energy",
    "bbox_count",
    "largest_bbox_area_ratio",
    "total_foreground_area_ratio",
    "mean_bbox_area_ratio",
    "max_temporal_weight",
    "average_temporal_weight",
]


@dataclass
class VideoFeatureVector:
    """Motion-derived features describing a single frame's activity."""

    motion_energy: float
    bbox_count: float
    largest_bbox_area_ratio: float
    total_foreground_area_ratio: float
    mean_bbox_area_ratio: float
    max_temporal_weight: float
    average_temporal_weight: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.motion_energy,
                self.bbox_count,
                self.largest_bbox_area_ratio,
                self.total_foreground_area_ratio,
                self.mean_bbox_area_ratio,
                self.max_temporal_weight,
                self.average_temporal_weight,
            ],
            dtype=np.float32,
        )

    @classmethod
    def zeros(cls) -> "VideoFeatureVector":
        """A no-motion frame (empty scene)."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
