from dataclasses import dataclass, field

import numpy as np


@dataclass
class BoundingBox:
    """Axis-aligned bounding box of a detected motion region."""

    x: int
    y: int
    w: int
    h: int
    area: int


@dataclass
class MotionResult:
    """Result of processing a single video frame for motion detection."""

    frame: np.ndarray           # Original BGR frame
    annotated_frame: np.ndarray # Copy of frame with bounding boxes drawn
    boxes: list[BoundingBox]
    has_motion: bool
    timestamp: float            # Unix timestamp at detection
