"""Turn motion-detection output into a fixed-length video feature vector.

Consumes the :class:`BoundingBox` list produced by ``MotionDetector.detect``
**before** the pipeline overwrites ``source_score`` with the cross-modal
ranking — at that point ``source_score`` still holds the IoU temporal weight
(1.0 = new object, 0.5 = persistent).
"""

from __future__ import annotations

from typing import Sequence

from src.vision.types import BoundingBox

from .types import VideoFeatureVector


class VideoFeatureExtractor:
    """Stateless extractor: (boxes, frame_shape) → :class:`VideoFeatureVector`."""

    def extract(
        self, boxes: Sequence[BoundingBox], frame_shape: tuple[int, ...]
    ) -> VideoFeatureVector:
        if not boxes or len(frame_shape) < 2:
            return VideoFeatureVector.zeros()

        frame_area = float(frame_shape[0] * frame_shape[1]) or 1.0
        ratios = [b.area / frame_area for b in boxes]
        weights = [b.source_score for b in boxes]
        # Ratios are clamped to [0,1]: overlapping boxes can otherwise sum to
        # more than the frame area (fix M7 carried into the feature space).
        total_ratio = min(sum(ratios), 1.0)

        return VideoFeatureVector(
            motion_energy=total_ratio,
            bbox_count=float(len(boxes)),
            largest_bbox_area_ratio=min(max(ratios), 1.0),
            total_foreground_area_ratio=total_ratio,
            mean_bbox_area_ratio=min(sum(ratios) / len(ratios), 1.0),
            max_temporal_weight=max(weights),
            average_temporal_weight=sum(weights) / len(weights),
        )
