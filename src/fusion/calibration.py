"""Score calibration so audio and video scores are comparable.

Raw anomaly scores from the two modalities live on different, drifting
scales, so combining them directly is meaningless. ``PercentileCalibrator``
maps a raw score to its **empirical percentile rank** in a rolling window of
recent history: the result is in [0, 1] and a value of 0.9 means "more
anomalous than ~90% of recent observations" — the *same* relative meaning for
either modality (ADR-0004).
"""

from __future__ import annotations

import bisect
import threading
from collections import deque


class PercentileCalibrator:
    """Maps raw scores to rolling-window percentile ranks in [0, 1].

    Thread-safe. During warmup (fewer than ``min_samples`` observations) the
    raw score is passed through clamped to [0, 1], so behaviour degrades
    gracefully before enough history exists.
    """

    def __init__(self, window: int = 500, min_samples: int = 50) -> None:
        self._hist: deque[float] = deque(maxlen=window)
        self._min_samples = min_samples
        self._lock = threading.Lock()

    def update(self, score: float) -> None:
        """Add a raw score to the rolling history."""
        with self._lock:
            self._hist.append(float(score))

    def calibrate(self, score: float) -> float:
        """Return the percentile rank of ``score`` within recent history."""
        with self._lock:
            n = len(self._hist)
            if n < self._min_samples:
                return float(min(max(score, 0.0), 1.0))
            ordered = sorted(self._hist)
        rank = bisect.bisect_right(ordered, float(score)) / n
        return float(min(max(rank, 0.0), 1.0))

    def calibrate_and_update(self, score: float) -> float:
        """Convenience: calibrate against current history, then record it."""
        out = self.calibrate(score)
        self.update(score)
        return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._hist)
