from dataclasses import dataclass, field

import numpy as np


@dataclass
class BoundingBox:
    """Axis-aligned bounding box of a detected motion region.

    ``source_score`` is a normalised 0–1 value estimating how
    likely this box is to be the *source* of an audio anomaly.
    It is computed by the pipeline using temporal correlation
    (new boxes score higher) and area ratio.
    """

    x: int
    y: int
    w: int
    h: int
    area: int
    source_score: float = 0.0


@dataclass
class MotionResult:
    """Result of processing a single video frame for motion detection."""

    frame: np.ndarray           # Original BGR frame
    annotated_frame: np.ndarray # Copy of frame with bounding boxes drawn
    boxes: list[BoundingBox]
    has_motion: bool
    timestamp: float            # Unix timestamp at detection
