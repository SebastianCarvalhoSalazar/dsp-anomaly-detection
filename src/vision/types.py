from dataclasses import dataclass, field

import numpy as np


@dataclass
class BoundingBox:
    """Axis-aligned bounding box of a detected motion region.

    ``temporal_weight`` is 1.0 for a box that just appeared and 0.5 for one
    that persists from the previous frame (IoU tracking). ``motion_score`` =
    area_ratio × temporal_weight ranks the *main motion* of a frame: large,
    new movement first. It carries no acoustic information (ADR-0015).
    """

    x: int
    y: int
    w: int
    h: int
    area: int
    temporal_weight: float = 0.0
    motion_score: float = 0.0


@dataclass
class MotionResult:
    """Result of processing a single video frame for motion detection."""

    frame: np.ndarray           # Original BGR frame
    annotated_frame: np.ndarray # Copy of frame with bounding boxes drawn
    boxes: list[BoundingBox]
    has_motion: bool
    timestamp: float            # Unix timestamp at detection
