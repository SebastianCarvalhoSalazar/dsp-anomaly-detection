"""Reusable online anomaly detector core.

``BaseAnomalyDetector`` holds all the modality-agnostic machinery so it can
be shared by the audio detector and the video detector (and any future
modality) without copy-pasting the IsolationForest + PCA + drift logic:

  - Sliding buffer + warmup
  - Optional Welford online Z-score normalization
  - PCA + IsolationForest fit/refit
  - Score calibration to [0, 1]
  - EMA smoothing + hysteresis
  - Adaptive percentile threshold
  - C2ST drift detection (optional)
  - save/load state and status/metrics snapshots

Design notes (fixes from the v0.2.0 review):
  - **C1**: when fitted, the normalizer is not updated with windows flagged
    as raw anomalies (``freeze_normalizer_on_anomaly``).
  - **C2**: calibration widens the lower score bound by ``calibration_margin``
    so strong anomalies do not all saturate at 1.0.
  - **H1**: the heavy fit (PCA + IsolationForest + C2ST) runs *outside* the
    lock. We snapshot the buffer under the lock, compute the new model
    lock-free, then swap it in atomically under the lock. Scoring threads
    keep using the previous model until the swap completes.
  - **M3/M4**: full scoring state is persisted and the feature dimension is
    validated on load and on each score() call.
"""

from __future__ import annotations

import os
import pickle
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score

from .config import DetectorConfig
from .types import AnomalyResult


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
        return ((x.astype(np.float64) - self._mean) / std).astype(np.float32)

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


class BaseAnomalyDetector:
    """Modality-agnostic online anomaly detector.

    Subclasses (audio/video) only differ by the ``DetectorConfig`` they
    pass and, optionally, the human-readable ``feature_names`` used to
    label drift / explainability output.

    Thread safety: all mutable state is protected by a single Lock. The
    heavy model fit runs outside the lock (see module docstring, H1).
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        self._config = config or DetectorConfig()
        self._feature_names = feature_names
        self._buffer: deque[np.ndarray] = deque(maxlen=self._config.buffer_size)
        self._model: IsolationForest | None = None
        self._is_fitted = False
        self._fitting = False  # guard: a fit is being computed (H1)
        self._samples_since_refit = 0
        self._feature_dim: Optional[int] = None
        self._lock = threading.Lock()

        # Calibration bounds updated after each fit
        self._score_min: float = -1.0
        self._score_max: float = 0.0

        # Z-score normalizer — dim set on first score() call
        self._normalizer: Optional[_WelfordNormalizer] = None

        # EMA smoothing + hysteresis
        self._smoothed_score: float = 0.0
        self._consecutive_anomalies: int = 0

        # Adaptive threshold — recent raw scores
        self._recent_scores: deque[float] = deque(
            maxlen=self._config.adaptive_score_window
        )
        # Mirror buffer of normalised [0,1] scores for visual threshold.
        self._recent_norm_scores: deque[float] = deque(
            maxlen=self._config.adaptive_score_window
        )

        # PCA — fitted alongside the IF model
        self._pca: Optional[PCA] = None

        # Drift detection state (C2ST)
        self._refit_count: int = 0
        self._prev_buffer: Optional[np.ndarray] = None
        self._drift_auc: float = 0.5
        self._top_drift_features: list[str] = []
        self._adaptive_threshold: float = 0.0
        self._score_mean: float = 0.0

    # ------------------------------------------------------------------ #
    #  Core API                                                           #
    # ------------------------------------------------------------------ #

    def score(self, feature_vector: np.ndarray) -> AnomalyResult:
        """Compute the anomaly score for a single feature vector.

        Returns a warmup result (score 0, is_anomaly False) until the
        buffer is full and the model is fitted.
        """
        fv = np.asarray(feature_vector, dtype=np.float32)
        ts = time.time()

        # --- Phase A: under lock — normalize, buffer, decide fit -------- #
        with self._lock:
            dim = fv.shape[0]

            # M4: a feature_dim change invalidates the fitted model/PCA.
            if self._feature_dim is not None and dim != self._feature_dim:
                self._reset_locked()

            if self._normalizer is None:
                self._normalizer = _WelfordNormalizer(dim)
            self._feature_dim = dim

            normalizer_updated = False
            if self._config.enable_zscore:
                # During warmup we always update so stats can converge.
                if not self._is_fitted:
                    self._normalizer.update(fv)
                    normalizer_updated = True
                fv_norm = self._normalizer.normalize(fv)
            else:
                fv_norm = fv

            self._buffer.append(fv_norm.copy())
            buf_len = len(self._buffer)

            need_fit = False
            is_initial = False
            fit_X: Optional[np.ndarray] = None
            fit_prev: Optional[np.ndarray] = None

            if (
                not self._is_fitted
                and buf_len >= self._config.buffer_size
                and not self._fitting
            ):
                need_fit = True
                is_initial = True
                self._fitting = True
                fit_X = np.array(list(self._buffer), dtype=np.float32)
                fit_prev = self._prev_buffer
            elif self._is_fitted:
                self._samples_since_refit += 1
                if (
                    self._samples_since_refit >= self._config.refit_every
                    and not self._fitting
                ):
                    need_fit = True
                    self._fitting = True
                    self._samples_since_refit = 0
                    fit_X = np.array(list(self._buffer), dtype=np.float32)
                    fit_prev = self._prev_buffer

        # --- Phase B: heavy compute OUTSIDE the lock (H1) --------------- #
        if need_fit:
            fit_result = self._compute_fit(fit_X, fit_prev)
            with self._lock:
                self._apply_fit(fit_result)
                if is_initial:
                    self._is_fitted = True
                    self._samples_since_refit = 0
                self._fitting = False

        # --- Phase C: under lock — score with current model ------------- #
        with self._lock:
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

            fv_score = fv_norm
            if self._config.enable_pca and self._pca is not None:
                fv_score = self._pca.transform(fv_norm.reshape(1, -1))[0].astype(
                    np.float32
                )

            raw = float(self._model.score_samples([fv_score])[0])
            anomaly_score = self._normalize_score(raw)
            self._recent_norm_scores.append(anomaly_score)

            # --- Adaptive threshold -------------------------------------
            self._recent_scores.append(raw)
            if (
                self._config.enable_adaptive_threshold
                and len(self._recent_scores) >= self._config.adaptive_score_window
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
                self._adaptive_threshold = float(self._model.offset_)

            if self._recent_scores:
                self._score_mean = float(np.mean(list(self._recent_scores)))

            # C1: fold benign windows into the normalizer; skip anomalies.
            if self._config.enable_zscore and not normalizer_updated:
                if (
                    not self._config.freeze_normalizer_on_anomaly
                    or not is_raw_anomaly
                ):
                    self._normalizer.update(fv)

            # --- EMA smoothing + hysteresis -----------------------------
            alpha = self._config.ema_alpha
            self._smoothed_score = (
                alpha * anomaly_score + (1 - alpha) * self._smoothed_score
            )

            if is_raw_anomaly:
                self._consecutive_anomalies += 1
            else:
                self._consecutive_anomalies = 0

            is_anomaly = (
                self._consecutive_anomalies >= self._config.hysteresis_count
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
                    self._pca.n_components_ if self._pca is not None else 0
                ),
            }

    def get_drift_metrics(self) -> dict:
        """Return a thread-safe snapshot of drift detection metrics.

        ``adaptive_threshold`` is the (100 - percentile)th percentile of
        recent normalised scores, matching the dashboard chart scale.
        """
        with self._lock:
            if (
                self._is_fitted
                and len(self._recent_norm_scores)
                >= self._config.adaptive_score_window
            ):
                norm_thresh = float(
                    np.percentile(
                        list(self._recent_norm_scores),
                        100.0 - self._config.adaptive_percentile,
                    )
                )
            else:
                norm_thresh = 0.5  # sensible default

            return {
                "adaptive_threshold": round(norm_thresh, 6),
                "score_mean": round(self._score_mean, 6),
                "drift_auc": round(self._drift_auc, 4),
                "top_drift_features": self._top_drift_features.copy(),
                "refit_count": self._refit_count,
            }

    def reset(self) -> None:
        """Clear buffer and model, returning to unfitted state."""
        with self._lock:
            self._reset_locked()

    def _reset_locked(self) -> None:
        """Reset all state. Caller must hold ``self._lock``."""
        self._buffer.clear()
        self._model = None
        self._pca = None
        self._is_fitted = False
        self._fitting = False
        self._samples_since_refit = 0
        self._feature_dim = None
        self._smoothed_score = 0.0
        self._consecutive_anomalies = 0
        self._recent_scores.clear()
        self._recent_norm_scores.clear()
        self._normalizer = None
        self._refit_count = 0
        self._prev_buffer = None
        self._drift_auc = 0.5
        self._top_drift_features = []
        self._adaptive_threshold = 0.0
        self._score_mean = 0.0

    # ------------------------------------------------------------------ #
    #  Persistent baseline                                                #
    # ------------------------------------------------------------------ #

    def save_state(self, path: Optional[str] = None) -> None:
        """Serialize detector state to disk for warmup-free restart (M3)."""
        path = path or self._config.state_path
        with self._lock:
            state = {
                "model": self._model,
                "pca": self._pca,
                "is_fitted": self._is_fitted,
                "buffer": list(self._buffer),
                "feature_dim": self._feature_dim,
                "score_min": self._score_min,
                "score_max": self._score_max,
                "smoothed_score": self._smoothed_score,
                "consecutive_anomalies": self._consecutive_anomalies,
                "samples_since_refit": self._samples_since_refit,
                "normalizer": (
                    self._normalizer.get_state() if self._normalizer else None
                ),
                "recent_scores": list(self._recent_scores),
                "recent_norm_scores": list(self._recent_norm_scores),
                "refit_count": self._refit_count,
                "prev_buffer": self._prev_buffer,
                "drift_auc": self._drift_auc,
                "top_drift_features": list(self._top_drift_features),
                "score_mean": self._score_mean,
                "adaptive_threshold": self._adaptive_threshold,
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
                    state["buffer"], maxlen=self._config.buffer_size
                )
                self._feature_dim = state.get("feature_dim")
                self._score_min = state["score_min"]
                self._score_max = state["score_max"]
                self._smoothed_score = state.get("smoothed_score", 0.0)
                self._consecutive_anomalies = state.get(
                    "consecutive_anomalies", 0
                )
                self._samples_since_refit = state.get("samples_since_refit", 0)
                self._pca = state.get("pca")
                self._refit_count = state.get("refit_count", 0)
                self._prev_buffer = state.get("prev_buffer")
                self._drift_auc = state.get("drift_auc", 0.5)
                self._top_drift_features = list(
                    state.get("top_drift_features", [])
                )
                self._score_mean = state.get("score_mean", 0.0)
                self._adaptive_threshold = state.get("adaptive_threshold", 0.0)
                norm_state = state.get("normalizer")
                if norm_state is not None:
                    dim = len(norm_state["mean"])
                    # M4: keep feature_dim consistent with the normalizer.
                    if self._feature_dim is None:
                        self._feature_dim = dim
                    self._normalizer = _WelfordNormalizer(dim)
                    self._normalizer.load_state(norm_state)
                for s in state.get("recent_scores", []):
                    self._recent_scores.append(s)
                for s in state.get("recent_norm_scores", []):
                    self._recent_norm_scores.append(s)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    #  Internal                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_feature_names(dim: int) -> list[str]:
        """Build human-readable names for each audio feature.

        Layout (from AudioProcessor):
          [0 .. n_scat-1]  scattering coefficients
          [n_scat .. +8]   wavelet energy bands
          [+0, +1]         RMS, ZCR
          [+0..+3]         centroid, flatness, rolloff, bw
          [+0..+2]         delta_rms, delta_centroid, delta_scat
        """
        names: list[str] = []
        n_wavelet = 8  # level=7 → 8 bands
        n_temporal = 2
        n_spectral = 4
        n_delta = 3
        n_suffix = n_wavelet + n_temporal + n_spectral + n_delta
        n_scat = dim - n_suffix
        if n_scat < 0:  # safety
            return [f"f{i}" for i in range(dim)]

        for i in range(n_scat):
            names.append(f"scat_{i}")
        for k in range(n_wavelet):
            names.append(f"wavelet_band_{k}")
        names += ["rms", "zcr"]
        names += [
            "spectral_centroid",
            "spectral_flatness",
            "spectral_rolloff",
            "spectral_bandwidth",
        ]
        names += ["delta_rms", "delta_centroid", "delta_scat_energy"]
        names = names[:dim]
        while len(names) < dim:
            names.append(f"f{len(names)}")
        return names

    def _resolve_feature_names(self, dim: int) -> list[str]:
        """Use injected names (H6) when available; else fall back."""
        if self._feature_names is not None and len(self._feature_names) == dim:
            return list(self._feature_names)
        return self._build_feature_names(dim)

    def _run_c2st(
        self, X_prev: np.ndarray, X_curr: np.ndarray
    ) -> tuple[float, list[str]]:
        """Classifier Two-Sample Test (pure; no ``self`` mutation).

        Train a RandomForest classifier to separate the previous buffer
        (label 0) from the current one (label 1). Returns
        ``(auc, top_5_feature_names)``. AUC ≈ 0.5 → no drift.
        """
        n0, n1 = len(X_prev), len(X_curr)
        X = np.vstack([X_prev, X_curr])
        y = np.concatenate([np.zeros(n0), np.ones(n1)])
        clf = RandomForestClassifier(
            n_estimators=self._config.c2st_n_estimators,
            max_depth=4,
            random_state=self._config.random_state,
        )
        n_folds = min(3, n0, n1)
        if n_folds < 2:
            return 0.5, []
        aucs = cross_val_score(clf, X, y, cv=n_folds, scoring="roc_auc")
        auc = float(max(np.mean(aucs), 1 - np.mean(aucs)))

        clf.fit(X, y)
        importances = clf.feature_importances_
        dim = X.shape[1]
        feat_names = self._resolve_feature_names(dim)
        top_idx = np.argsort(importances)[::-1][:5]
        top_names = [feat_names[i] for i in top_idx if importances[i] > 0]
        return auc, top_names

    def _compute_fit(
        self, X: np.ndarray, prev_buffer: Optional[np.ndarray]
    ) -> dict:
        """Fit a fresh model on a buffer snapshot. Runs OUTSIDE the lock (H1).

        Pure with respect to ``self`` mutable state — returns the artefacts
        to be swapped in atomically by ``_apply_fit``.
        """
        drift_auc = self._drift_auc
        top_feats = self._top_drift_features
        if self._config.enable_drift_detection and prev_buffer is not None:
            drift_auc, top_feats = self._run_c2st(prev_buffer, X)

        if self._config.enable_pca:
            n_comp = min(self._config.pca_components, X.shape[1])
            pca = PCA(n_components=n_comp)
            X_reduced = pca.fit_transform(X)
        else:
            pca = None
            X_reduced = X

        model = IsolationForest(
            n_estimators=self._config.n_estimators,
            contamination=self._config.contamination,
            max_samples=self._config.max_samples,
            random_state=self._config.random_state,
        )
        model.fit(X_reduced)
        scores = model.score_samples(X_reduced)
        lo, hi = float(scores.min()), float(scores.max())
        # C2: widen the lower bound so worse-than-training anomalies do not
        # all clip to exactly 1.0.
        margin = self._config.calibration_margin * (hi - lo)
        return {
            "model": model,
            "pca": pca,
            "score_min": lo - margin,
            "score_max": hi,
            "drift_auc": drift_auc,
            "top_drift_features": top_feats,
            "prev_buffer": X,
        }

    def _apply_fit(self, fit: dict) -> None:
        """Swap in a freshly computed model. Caller must hold ``self._lock``."""
        self._model = fit["model"]
        self._pca = fit["pca"]
        self._score_min = fit["score_min"]
        self._score_max = fit["score_max"]
        self._drift_auc = fit["drift_auc"]
        self._top_drift_features = fit["top_drift_features"]
        self._prev_buffer = fit["prev_buffer"]
        self._refit_count += 1

    def _normalize_score(self, raw_score: float) -> float:
        denom = self._score_max - self._score_min
        if denom < 1e-8:
            return 0.0
        norm = (raw_score - self._score_min) / denom
        return float(np.clip(1.0 - norm, 0.0, 1.0))
