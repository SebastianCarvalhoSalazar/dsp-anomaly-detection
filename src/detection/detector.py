"""Audio anomaly detector.

Thin wrapper over :class:`BaseAnomalyDetector` (see ``base.py``) — all the
IsolationForest + PCA + drift machinery lives in the shared base so the
video detector can reuse it without duplicating the logic (and the bugs).

The audio detector accepts the feature-name layout from ``AudioProcessor``
(``feature_names``) so drift / explainability output uses the real names
instead of a hardcoded layout.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseAnomalyDetector, _WelfordNormalizer  # noqa: F401
from .config import DetectorConfig


class AnomalyDetector(BaseAnomalyDetector):
    """Online audio anomaly detector (IsolationForest + improvements).

    Enhancements over basic IF:
    - Z-score normalization (Welford's online) so all features contribute
      equally to tree splits.
    - EMA score smoothing with hysteresis to reduce false positives.
    - Adaptive percentile threshold (bottom N% of recent scores).
    - Persistent baseline: save/load state to skip warmup on restart.
    - C2ST drift detection between consecutive buffers.

    Thread safety: all mutable state is protected by a single Lock; the
    model fit runs outside the lock.
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__(config=config, feature_names=feature_names)
