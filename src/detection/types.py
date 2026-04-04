from dataclasses import dataclass, field

import numpy as np


@dataclass
class AnomalyResult:
    """Result of scoring a single feature vector."""

    raw_score: float          # IsolationForest.score_samples output (negative; lower = more anomalous)
    anomaly_score: float      # Normalized to [0, 1]; higher = more anomalous
    is_anomaly: bool          # True when anomaly_score exceeds the detector threshold
    is_fitted: bool           # False during warmup (buffer not yet full)
    timestamp: float          # Unix timestamp
    window_index: int
    feature_vector: np.ndarray = field(repr=False)  # shape (feature_dim,)
