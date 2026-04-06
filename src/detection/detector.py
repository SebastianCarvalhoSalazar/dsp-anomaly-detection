import os
import pickle
import time
import threading
from collections import deque
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from .config import DetectorConfig
from .types import AnomalyResult


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class _WelfordNormalizer:
    """Welford's online Z-score normalizer (per-feature).

    Maintains running mean and variance with O(1) memory and O(d) per
    update.  After a burn-in of ``min_samples`` calls, ``normalize``
    returns z-scored vectors; before that it returns the raw vector.
    """

    def __init__(self, dim: int, min_samples: int = 30) -> None:
        self._n = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._m2 = np.zeros(dim, dtype=np.float64)
        self._min_samples = min_samples

    def update(self, x: np.ndarray) -> None:
        self._n += 1
        delta = x.astype(np.float64) - self._mean
        self._mean += delta / self._n
        delta2 = x.astype(np.float64) - self._mean
        self._m2 += delta * delta2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self._n < self._min_samples:
            return x
        var = self._m2 / self._n
        std = np.sqrt(var + 1e-8)
        return ((x.astype(np.float64) - self._mean) / std).astype(
            np.float32
        )

    def get_state(self) -> dict:
        return {
            "n": self._n,
            "mean": self._mean.copy(),
            "m2": self._m2.copy(),
        }

    def load_state(self, state: dict) -> None:
        self._n = state["n"]
        self._mean = state["mean"]
        self._m2 = state["m2"]


class AnomalyDetector:
    """Online anomaly detector with IsolationForest + improvements.

    Enhancements over basic IF:
    - Z-score normalization (Welford's online) so all features
      contribute equally to tree splits.
    - EMA score smoothing with hysteresis to reduce false positives.
    - Adaptive percentile threshold (bottom N% of recent scores).
    - Persistent baseline: save/load state to skip warmup on restart.

    Thread safety: all mutable state is protected by a single Lock.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self._config = config or DetectorConfig()
        self._buffer: deque[np.ndarray] = deque(
            maxlen=self._config.buffer_size
        )
        self._model: IsolationForest | None = None
        self._is_fitted = False
        self._samples_since_refit = 0
        self._lock = threading.Lock()

        # Calibration bounds updated after each fit
        self._score_min: float = -1.0
        self._score_max: float = 0.0

        # Z-score normalizer (2.3) — dim set on first score() call
        self._normalizer: Optional[_WelfordNormalizer] = None

        # EMA smoothing (2.4)
        self._smoothed_score: float = 0.0
        self._consecutive_anomalies: int = 0

        # Adaptive threshold (3.5) — recent raw scores
        self._recent_scores: deque[float] = deque(
            maxlen=self._config.adaptive_score_window
        )
        # Mirror buffer of normalised [0,1] scores for
        # visual threshold (percentile 98 of norm scores).
        self._recent_norm_scores: deque[float] = deque(
            maxlen=self._config.adaptive_score_window
        )

        # PCA — fitted alongside the IF model
        self._pca: Optional[PCA] = None

        # Drift detection state
        self._refit_count: int = 0
        self._prev_feature_mean: Optional[np.ndarray] = None
        self._feature_mean_drift: float = 0.0
        self._adaptive_threshold: float = 0.0
        self._score_mean: float = 0.0

    # ------------------------------------------------------------------ #
    #  Core API                                                           #
    # ------------------------------------------------------------------ #

    def score(self, feature_vector: np.ndarray) -> AnomalyResult:
        """Compute anomaly score for a single feature vector.

        Returns a warmup result (score 0, is_anomaly False) until the
        buffer is full and the model is fitted.
        """
        fv = np.asarray(feature_vector, dtype=np.float32)
        ts = time.time()

        with self._lock:
            dim = fv.shape[0]

            # Lazy init normalizer on first call (dim is dynamic)
            if self._normalizer is None:
                self._normalizer = _WelfordNormalizer(dim)

            # Z-score (2.3)
            if self._config.enable_zscore:
                self._normalizer.update(fv)
                fv_norm = self._normalizer.normalize(fv)
            else:
                fv_norm = fv

            self._buffer.append(fv_norm.copy())
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

            # PCA transform for scoring (if enabled)
            fv_score = fv_norm
            if self._config.enable_pca and self._pca is not None:
                fv_score = self._pca.transform(
                    fv_norm.reshape(1, -1)
                )[0].astype(np.float32)

            raw = float(
                self._model.score_samples([fv_score])[0]
            )
            anomaly_score = self._normalize_score(raw)

            # Keep normalised scores for visual threshold
            self._recent_norm_scores.append(anomaly_score)

            # --- Adaptive threshold (3.5) ------------------
            self._recent_scores.append(raw)
            if (
                self._config.enable_adaptive_threshold
                and len(self._recent_scores)
                >= self._config.buffer_size
            ):
                adaptive_thresh = float(
                    np.percentile(
                        list(self._recent_scores),
                        self._config.adaptive_percentile,
                    )
                )
                is_raw_anomaly = raw < adaptive_thresh
                self._adaptive_threshold = adaptive_thresh
            else:
                is_raw_anomaly = raw < self._model.offset_
                self._adaptive_threshold = float(
                    self._model.offset_
                )

            # Drift: running mean of recent scores
            if self._recent_scores:
                self._score_mean = float(
                    np.mean(list(self._recent_scores))
                )

            # --- EMA smoothing + hysteresis (2.4) ----------
            alpha = self._config.ema_alpha
            self._smoothed_score = (
                alpha * anomaly_score
                + (1 - alpha) * self._smoothed_score
            )

            if is_raw_anomaly:
                self._consecutive_anomalies += 1
            else:
                self._consecutive_anomalies = 0

            is_anomaly = (
                self._consecutive_anomalies
                >= self._config.hysteresis_count
            )

        return AnomalyResult(
            raw_score=raw,
            anomaly_score=self._smoothed_score,
            is_anomaly=is_anomaly,
            is_fitted=True,
            timestamp=ts,
            window_index=buf_len - 1,
            feature_vector=fv,
        )

    # ------------------------------------------------------------------ #
    #  Status / reset                                                     #
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        with self._lock:
            return {
                "is_fitted": self._is_fitted,
                "buffer_fill": len(self._buffer),
                "buffer_size": self._config.buffer_size,
                "n_estimators": self._config.n_estimators,
                "samples_since_refit": self._samples_since_refit,
                "smoothed_score": self._smoothed_score,
                "consecutive_anomalies": self._consecutive_anomalies,
                "pca_enabled": self._config.enable_pca,
                "pca_components": (
                    self._pca.n_components_
                    if self._pca is not None
                    else 0
                ),
            }

    def get_drift_metrics(self) -> dict:
        """Return current drift detection metrics.

        Thread-safe snapshot of metrics that describe how
        the detector's operating environment is evolving.
        The adaptive_threshold is the 98th percentile of
        recent normalised scores, matching the chart scale.
        """
        with self._lock:
            # Visual threshold: p98 of normalised scores
            if (
                self._is_fitted
                and len(self._recent_norm_scores)
                >= self._config.buffer_size
            ):
                norm_thresh = float(np.percentile(
                    list(self._recent_norm_scores),
                    100.0
                    - self._config.adaptive_percentile,
                ))
            else:
                norm_thresh = 0.5  # sensible default

            return {
                "adaptive_threshold": round(
                    norm_thresh, 6,
                ),
                "score_mean": round(
                    self._score_mean, 6,
                ),
                "feature_mean_drift": round(
                    self._feature_mean_drift, 6,
                ),
                "refit_count": self._refit_count,
            }

    def reset(self) -> None:
        """Clear buffer and model, returning to unfitted state."""
        with self._lock:
            self._buffer.clear()
            self._model = None
            self._pca = None
            self._is_fitted = False
            self._samples_since_refit = 0
            self._smoothed_score = 0.0
            self._consecutive_anomalies = 0
            self._recent_scores.clear()
            self._recent_norm_scores.clear()
            self._normalizer = None
            self._refit_count = 0
            self._prev_feature_mean = None
            self._feature_mean_drift = 0.0
            self._adaptive_threshold = 0.0
            self._score_mean = 0.0

    # ------------------------------------------------------------------ #
    #  Persistent baseline (3.4)                                          #
    # ------------------------------------------------------------------ #

    def save_state(self, path: Optional[str] = None) -> None:
        """Serialize detector state to disk for warmup-free restart."""
        path = path or self._config.state_path
        with self._lock:
            state = {
                "model": self._model,
                "pca": self._pca,
                "is_fitted": self._is_fitted,
                "buffer": list(self._buffer),
                "score_min": self._score_min,
                "score_max": self._score_max,
                "smoothed_score": self._smoothed_score,
                "normalizer": (
                    self._normalizer.get_state()
                    if self._normalizer
                    else None
                ),
                "recent_scores": list(self._recent_scores),
                "refit_count": self._refit_count,
                "prev_feature_mean": self._prev_feature_mean,
            }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, path: Optional[str] = None) -> bool:
        """Load a previously saved state. Returns True on success."""
        path = path or self._config.state_path
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            with self._lock:
                self._model = state["model"]
                self._is_fitted = state["is_fitted"]
                self._buffer = deque(
                    state["buffer"],
                    maxlen=self._config.buffer_size,
                )
                self._score_min = state["score_min"]
                self._score_max = state["score_max"]
                self._smoothed_score = state.get(
                    "smoothed_score", 0.0
                )
                self._pca = state.get("pca")
                self._refit_count = state.get(
                    "refit_count", 0
                )
                self._prev_feature_mean = state.get(
                    "prev_feature_mean"
                )
                norm_state = state.get("normalizer")
                if norm_state is not None:
                    dim = len(norm_state["mean"])
                    self._normalizer = _WelfordNormalizer(dim)
                    self._normalizer.load_state(norm_state)
                for s in state.get("recent_scores", []):
                    self._recent_scores.append(s)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    #  Internal                                                           #
    # ------------------------------------------------------------------ #

    def _fit(self) -> None:
        # Caller must hold self._lock
        X = np.array(list(self._buffer), dtype=np.float32)

        # Drift: compute feature-mean shift vs previous fit
        current_mean = X.mean(axis=0)
        if (
            self._config.enable_drift_detection
            and self._prev_feature_mean is not None
        ):
            self._feature_mean_drift = float(
                np.linalg.norm(
                    current_mean - self._prev_feature_mean
                )
            )
        self._prev_feature_mean = current_mean

        # PCA: fit on the buffer, then transform
        if self._config.enable_pca:
            n_comp = min(
                self._config.pca_components, X.shape[1],
            )
            self._pca = PCA(n_components=n_comp)
            X_reduced = self._pca.fit_transform(X)
        else:
            X_reduced = X

        self._model = IsolationForest(
            n_estimators=self._config.n_estimators,
            contamination=self._config.contamination,
            max_samples=self._config.max_samples,
            random_state=self._config.random_state,
        )
        self._model.fit(X_reduced)
        scores = self._model.score_samples(X_reduced)
        self._score_min = float(scores.min())
        self._score_max = float(scores.max())
        self._refit_count += 1

    def _normalize_score(self, raw_score: float) -> float:
        denom = self._score_max - self._score_min
        if denom < 1e-8:
            return 0.0
        norm = (raw_score - self._score_min) / denom
        return float(np.clip(1.0 - norm, 0.0, 1.0))
