from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureVector:
    """Feature vector extracted from a single audio window."""

    data: np.ndarray      # shape (feature_dim,), float32
    timestamp: float      # Unix timestamp at extraction (time.time())
    window_index: int     # Sequential index within the current session
    rms: float            # Root mean square amplitude (for dashboard display)
