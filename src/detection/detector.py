import time
import threading
from collections import deque

import numpy as np
from sklearn.ensemble import IsolationForest

from .config import DetectorConfig
from .types import AnomalyResult


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class AnomalyDetector:
    """Online anomaly detector using a buffered Isolation Forest.

    Because IsolationForest has no partial_fit, the model is refitted
    periodically on a fixed-size ring buffer. This approximates online
    adaptation to the recent signal distribution.

    Workflow:
        - Collect samples into a deque(maxlen=buffer_size).
        - First fit at buffer_size samples (default 200, ~25s of audio).
        - Refit every refit_every new samples thereafter.
        - During warmup returns is_fitted=False and anomaly_score=0.0.

    Thread safety: all mutable state is protected by a single Lock.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self._config = config or DetectorConfig()
        self._buffer: deque[np.ndarray] = deque(maxlen=self._config.buffer_size)
        self._model: IsolationForest | None = None
        self._is_fitted = False
        self._samples_since_refit = 0
        self._lock = threading.Lock()
        # Calibration bounds updated after each fit for score normalization
        self._score_min: float = -1.0
        self._score_max: float = 0.0

    def score(self, feature_vector: np.ndarray) -> AnomalyResult:
        """Compute anomaly score for a single feature vector.

        If the model is not yet fitted, returns a warmup result with
        anomaly_score=0.0 and is_anomaly=False.

        Parameters
        ----------
        feature_vector : np.ndarray
            Shape (feature_dim,), any float dtype.

        Returns
        -------
        AnomalyResult
        """
        fv = np.asarray(feature_vector, dtype=np.float32)
        ts = time.time()

        with self._lock:
            self._buffer.append(fv.copy())
            buf_len = len(self._buffer)

            if not self._is_fitted and buf_len >= self._config.buffer_size:
                self._fit()
                self._is_fitted = True
                self._samples_since_refit = 0
            elif self._is_fitted:
                self._samples_since_refit += 1
                if self._samples_since_refit >= self._config.refit_every:
                    self._fit()
                    self._samples_since_refit = 0

            if not self._is_fitted:
                return AnomalyResult(
                    raw_score=0.0,
                    anomaly_score=0.0,
                    is_anomaly=False,
                    is_fitted=False,
                    timestamp=ts,
                    window_index=buf_len - 1,
                    feature_vector=fv,
                )

            raw = float(self._model.score_samples([fv])[0])
            anomaly_score = self._normalize_score(raw)
            # IsolationForest.offset_ is the decision threshold (fitted)
            is_anomaly = raw < self._model.offset_

        return AnomalyResult(
            raw_score=raw,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            is_fitted=True,
            timestamp=ts,
            window_index=buf_len - 1,
            feature_vector=fv,
        )

    def get_status(self) -> dict:
        """Return current detector state."""
        with self._lock:
            return {
                "is_fitted": self._is_fitted,
                "buffer_fill": len(self._buffer),
                "buffer_size": self._config.buffer_size,
                "n_estimators": self._config.n_estimators,
                "samples_since_refit": self._samples_since_refit,
            }

    def reset(self) -> None:
        """Clear buffer and model, returning detector to unfitted state."""
        with self._lock:
            self._buffer.clear()
            self._model = None
            self._is_fitted = False
            self._samples_since_refit = 0

    def _fit(self) -> None:
        # Caller must hold self._lock
        X = np.array(list(self._buffer), dtype=np.float32)
        self._model = IsolationForest(
            n_estimators=self._config.n_estimators,
            contamination=self._config.contamination,
            max_samples=self._config.max_samples,
            random_state=self._config.random_state,
        )
        self._model.fit(X)
        # Calibrate normalization bounds from the training buffer scores
        scores = self._model.score_samples(X)
        self._score_min = float(scores.min())
        self._score_max = float(scores.max())

    def _normalize_score(self, raw_score: float) -> float:
        # Map IF score to [0, 1]: lower raw = more anomalous = higher output.
        # Use min-max calibration from the fit buffer, clamped to [0, 1].
        # anomaly_score = 1 - (raw - min) / (max - min)  (inverted: anomaly is high)
        denom = self._score_max - self._score_min
        if denom < 1e-8:
            return 0.0
        norm = (raw_score - self._score_min) / denom   # 0=most anomalous, 1=most normal
        return float(np.clip(1.0 - norm, 0.0, 1.0))
