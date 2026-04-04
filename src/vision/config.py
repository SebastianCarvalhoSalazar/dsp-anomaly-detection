from dataclasses import dataclass


@dataclass
class VisionConfig:
    """Configuration for video capture and motion detection."""

    capture_device: int = 0
    target_width: int = 640
    target_height: int = 480
    fps: float = 25.0
    # MOG2 background model history (number of frames)
    mog2_history: int = 200
    # Per-pixel variance threshold: lower = more sensitive to motion
    mog2_var_threshold: float = 16.0
    # Minimum contour area in pixels²; smaller regions discarded as noise
    min_contour_area: int = 500
    # Morphological kernel size for noise removal and region merging
    morph_kernel_size: int = 5
    dilation_iterations: int = 2
