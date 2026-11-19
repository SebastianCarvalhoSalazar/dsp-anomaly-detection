"""Reusable online anomaly detector core.

``BaseAnomalyDetector`` holds all the modality-agnostic machinery so it can
be shared by the audio detector and the video detector (and any future
modality) without copy-pasting the IsolationForest + PCA + drift logic:

  - Sliding buffer of RAW feature vectors + warmup
  - Z-score scaler fitted on the training buffer and frozen with the model
  - PCA + IsolationForest fit/refit
  - Score calibration to [0, 1]
  - EMA smoothing + hysteresis
  - Adaptive percentile threshold
  - C2ST drift detection (optional)
  - save/load state and status/metrics snapshots

Design notes:
  - **F1 (plan-correcciones)**: the buffer stores raw vectors and every fit
    learns ``scaler → PCA → IsolationForest`` as one frozen chain, used
    unchanged for scoring until the next fit. The former online Welford
    normalizer let the first 30 warmup windows enter the buffer unscaled
    (PCA collapsed onto them) and mixed vectors scaled with different stats.
  - **F2 (plan-correcciones)**: the decision threshold is fixed at each fit
    from the training-buffer scores (not a percentile of live scores, which
    absorbed any anomaly longer than ~2 % of the window). Windows flagged as
    anomalous are quarantined out of the training buffer; if the anomaly
    persists for ``accept_after_windows`` they are accepted as the new normal
    and a refit is forced.
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

import logging
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
from .snapshots import SnapshotStore
from .types import AnomalyResult

logger = logging.getLogger(__name__)

# Minimum buffered windows before warmup explanations are meaningful.
_MIN_EXPLAIN_SAMPLES = 30


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
        snapshot_store: Optional[SnapshotStore] = None,
    ) -> None:
        self._config = config or DetectorConfig()
        self._feature_names = feature_names
        self._snapshot_store = snapshot_store
        self._buffer: deque[np.ndarray] = deque(maxlen=self._config.buffer_size)
        self._model: IsolationForest | None = None
        self._is_fitted = False
        self._fitting = False  # guard: a fit is being computed (H1)
        self._samples_since_refit = 0
        # Windows scored since construction / last reset (AnomalyResult.window_index)
        self._window_count = 0
        self._feature_dim: Optional[int] = None
        self._lock = threading.Lock()

        # Calibration bounds updated after each fit
        self._score_min: float = -1.0
        self._score_max: float = 0.0

        # Frozen z-score scaler + decision threshold (set at each fit, F1/F2)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._threshold: float = 0.0

        # Quarantine of anomalous windows (kept out of the training buffer)
        self._quarantine: deque[np.ndarray] = deque(
            maxlen=max(1, self._config.accept_after_windows)
        )
        self._quarantine_hold: int = 0
        self._force_refit: bool = False

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
        self._score_mean: float = 0.0
        # Why the last fit happened: "scheduled" | "drift" | "initial" (Req 7)
        self._refit_reason: str = "scheduled"
        self._pending_refit_reason: str = "scheduled"

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

        # --- Phase A: under lock — buffer (warmup) and decide fit ------- #
        with self._lock:
            dim = fv.shape[0]

            # M4: a feature_dim change invalidates the fitted model/PCA.
            if self._feature_dim is not None and dim != self._feature_dim:
                self._reset_locked()
            self._feature_dim = dim

            # During warmup every window is training data (raw, F1).
            appended = not self._is_fitted
            if appended:
                self._buffer.append(fv.copy())
            buf_len = len(self._buffer)
            # Running window index. (It used to be buf_len - 1, which froze at
            # buffer_size - 1 = 499 once the buffer was full.)
            window_index = self._window_count
            self._window_count += 1

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
                self._pending_refit_reason = "initial"
            elif self._is_fitted and not self._fitting:
                # Drift-aware: shorten the interval when drift is high (Req 7).
                interval = self._config.refit_every
                reason = "scheduled"
                if (
                    self._config.enable_drift_aware_refit
                    and self._drift_auc >= self._config.drift_refit_threshold
                ):
                    interval = max(
                        self._config.min_refit_interval,
                        int(interval * self._config.drift_refit_factor),
                    )
                    reason = "drift"
                if self._force_refit:
                    need_fit = True
                    reason = "accepted"
                elif self._samples_since_refit >= interval:
                    need_fit = True
                if need_fit:
                    self._pending_refit_reason = reason
                    self._samples_since_refit = 0
                    self._force_refit = False
            if need_fit:
                self._fitting = True
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
            # Persist a snapshot off the hot-path (Req 8).
            self._save_snapshot(fit_result, fit_X)

        # --- Phase C: under lock — score with the frozen chain ---------- #
        with self._lock:
            if not self._is_fitted:
                return AnomalyResult(
                    raw_score=0.0,
                    anomaly_score=0.0,
                    is_anomaly=False,
                    is_fitted=False,
                    timestamp=ts,
                    window_index=window_index,
                    feature_vector=fv,
                )

            raw = float(self._model.score_samples([self._transform(fv)])[0])
            anomaly_score = self._normalize_score(raw)
            self._recent_norm_scores.append(anomaly_score)
            self._recent_scores.append(raw)
            self._score_mean = float(np.mean(self._recent_scores))

            # F2: fixed threshold from the training buffer (set at fit).
            is_raw_anomaly = raw < self._threshold

            # F2: keep anomalous windows out of the training buffer. (The
            # window that completed the warmup is already buffered.)
            if not appended:
                self._route_to_buffer(fv, is_raw_anomaly)

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
            window_index=window_index,
            feature_vector=fv,
        )

    def _transform(self, fv: np.ndarray) -> np.ndarray:
        """Apply the frozen scaler (+ PCA). Caller must hold ``self._lock``."""
        x = ((fv.astype(np.float64) - self._mean) / self._std).astype(np.float32)
        if self._pca is not None:
            x = self._pca.transform(x.reshape(1, -1))[0].astype(np.float32)
        return x

    def _route_to_buffer(self, fv: np.ndarray, is_raw_anomaly: bool) -> None:
        """Append a post-fit window to the training buffer or quarantine it.

        Caller must hold ``self._lock``. Anomalous windows (plus a short
        ``hysteresis_count`` tail to bridge gaps) are quarantined so a long
        anomaly cannot teach the model that it is normal. If the quarantine
        streak reaches ``accept_after_windows`` the sound is accepted as the
        new normal: the quarantined windows join the buffer and a refit is
        forced. When the anomaly ends earlier, the quarantine is discarded.
        """
        cfg = self._config
        if not cfg.quarantine_anomalies:
            self._buffer.append(fv.copy())
            self._samples_since_refit += 1
            return
        if is_raw_anomaly:
            self._quarantine_hold = cfg.hysteresis_count
        elif self._quarantine_hold > 0:
            self._quarantine_hold -= 1
        else:
            self._quarantine.clear()
            self._buffer.append(fv.copy())
            self._samples_since_refit += 1
            return
        self._quarantine.append(fv.copy())
        if len(self._quarantine) >= cfg.accept_after_windows:
            self._buffer.extend(self._quarantine)
            self._samples_since_refit += len(self._quarantine)
            self._quarantine.clear()
            self._quarantine_hold = 0
            self._force_refit = True

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

        ``adaptive_threshold`` is the decision threshold mapped to the same
        [0, 1] scale as ``anomaly_score`` (F2: one threshold, one scale).
        """
        with self._lock:
            norm_thresh = (
                self._normalize_score(self._threshold)
                if self._is_fitted else 0.5
            )
            return {
                "adaptive_threshold": round(norm_thresh, 6),
                "score_mean": round(self._score_mean, 6),
                "drift_auc": round(self._drift_auc, 4),
                "top_drift_features": self._top_drift_features.copy(),
                "refit_count": self._refit_count,
                "refit_reason": self._refit_reason,
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
        self._mean = None
        self._std = None
        self._threshold = 0.0
        self._quarantine.clear()
        self._quarantine_hold = 0
        self._force_refit = False
        self._is_fitted = False
        self._fitting = False
        self._samples_since_refit = 0
        self._window_count = 0
        self._feature_dim = None
        self._smoothed_score = 0.0
        self._consecutive_anomalies = 0
        self._recent_scores.clear()
        self._recent_norm_scores.clear()
        self._refit_count = 0
        self._prev_buffer = None
        self._drift_auc = 0.5
        self._top_drift_features = []
        self._score_mean = 0.0
        self._refit_reason = "scheduled"
        self._pending_refit_reason = "scheduled"

    # ------------------------------------------------------------------ #
    #  Persistent baseline                                                #
    # ------------------------------------------------------------------ #

    # Bumped when the persisted layout changes incompatibly. v2: raw buffer
    # + frozen scaler/threshold (F1/F2); v1 states held a z-scored buffer
    # and a Welford normalizer and are discarded on load.
    _STATE_VERSION = 2

    def save_state(self, path: Optional[str] = None) -> None:
        """Serialize detector state to disk for warmup-free restart (M3)."""
        path = path or self._config.state_path
        with self._lock:
            state = {
                "version": self._STATE_VERSION,
                "model": self._model,
                "pca": self._pca,
                "mean": self._mean,
                "std": self._std,
                "threshold": self._threshold,
                "is_fitted": self._is_fitted,
                "buffer": list(self._buffer),
                "feature_dim": self._feature_dim,
                "score_min": self._score_min,
                "score_max": self._score_max,
                "smoothed_score": self._smoothed_score,
                "consecutive_anomalies": self._consecutive_anomalies,
                "samples_since_refit": self._samples_since_refit,
                "recent_scores": list(self._recent_scores),
                "recent_norm_scores": list(self._recent_norm_scores),
                "refit_count": self._refit_count,
                "prev_buffer": self._prev_buffer,
                "drift_auc": self._drift_auc,
                "top_drift_features": list(self._top_drift_features),
                "score_mean": self._score_mean,
                "refit_reason": self._refit_reason,
            }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, path: Optional[str] = None) -> bool:
        """Load a previously saved state. Returns True on success.

        States written by an incompatible version are ignored (the detector
        warms up again) rather than mixing scaled and raw buffers.
        """
        path = path or self._config.state_path
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            if state.get("version") != self._STATE_VERSION:
                logger.warning(
                    "Ignoring detector state %s: incompatible version %s "
                    "(expected %s); warming up from scratch.",
                    path, state.get("version"), self._STATE_VERSION,
                )
                return False
            with self._lock:
                self._model = state["model"]
                self._pca = state.get("pca")
                self._mean = state["mean"]
                self._std = state["std"]
                self._threshold = state["threshold"]
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
                self._refit_count = state.get("refit_count", 0)
                self._prev_buffer = state.get("prev_buffer")
                self._drift_auc = state.get("drift_auc", 0.5)
                self._top_drift_features = list(
                    state.get("top_drift_features", [])
                )
                self._score_mean = state.get("score_mean", 0.0)
                self._refit_reason = state.get("refit_reason", "scheduled")
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

        # F1: the scaler is learned from this buffer and frozen with the model.
        mean, std = self._fit_scaler(X)
        Xz = ((X.astype(np.float64) - mean) / std).astype(np.float32)

        if self._config.enable_pca:
            n_comp = min(self._config.pca_components, Xz.shape[1], Xz.shape[0])
            pca = PCA(n_components=n_comp)
            X_reduced = pca.fit_transform(Xz)
        else:
            pca = None
            X_reduced = Xz

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
        # F2: the decision threshold is fixed per fit. With the adaptive
        # option it is the given percentile of the *training* scores (the
        # reference "normal"), otherwise the IsolationForest offset_.
        if self._config.enable_adaptive_threshold:
            threshold = float(
                np.percentile(scores, self._config.adaptive_percentile)
            )
        else:
            threshold = float(model.offset_)
        return {
            "model": model,
            "pca": pca,
            "mean": mean,
            "std": std,
            "threshold": threshold,
            "score_min": lo - margin,
            "score_max": hi,
            "drift_auc": drift_auc,
            "top_drift_features": top_feats,
            "prev_buffer": X,
        }

    def _fit_scaler(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-feature mean/std of the training buffer (identity if z-score
        is disabled). Constant features get std=1 to avoid division by 0."""
        d = X.shape[1]
        if not self._config.enable_zscore:
            return np.zeros(d), np.ones(d)
        Xd = X.astype(np.float64)
        mean = Xd.mean(axis=0)
        std = Xd.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return mean, std

    def _apply_fit(self, fit: dict) -> None:
        """Swap in a freshly computed model. Caller must hold ``self._lock``."""
        self._model = fit["model"]
        self._pca = fit["pca"]
        self._mean = fit["mean"]
        self._std = fit["std"]
        self._threshold = fit["threshold"]
        self._score_min = fit["score_min"]
        self._score_max = fit["score_max"]
        self._drift_auc = fit["drift_auc"]
        self._top_drift_features = fit["top_drift_features"]
        self._prev_buffer = fit["prev_buffer"]
        self._refit_reason = self._pending_refit_reason
        self._refit_count += 1

    def _save_snapshot(self, fit: dict, fit_X: np.ndarray) -> None:
        """Persist a model snapshot (Req 8). Best-effort, off the hot-path."""
        if self._snapshot_store is None:
            return
        try:
            metadata = {
                "timestamp": time.time(),
                "refit_count": self._refit_count,
                "refit_reason": self._refit_reason,
                "drift_auc": round(self._drift_auc, 4),
                "n_samples": int(fit_X.shape[0]),
                "buffer_mean": float(np.mean(fit_X)),
                "buffer_std": float(np.std(fit_X)),
            }
            self._snapshot_store.save(
                model=fit["model"], pca=fit["pca"], metadata=metadata
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Snapshot save failed: %s", exc)

    def top_features(self, feature_vector: np.ndarray, k: int = 3) -> list[str]:
        """Top-k features by absolute z-score vs the recent baseline (Req 9).

        Explains *why* a window looks anomalous in terms of the named features,
        e.g. ``["spectral_centroid +4.2σ", "wavelet_band_4 +3.1σ"]``. Returns an
        empty list during warmup (insufficient baseline statistics).
        """
        fv = np.asarray(feature_vector, dtype=np.float64)
        with self._lock:
            if self._is_fitted and self._mean is not None:
                mean, std = self._mean, self._std
            elif len(self._buffer) >= _MIN_EXPLAIN_SAMPLES:
                # Warmup: explain against the buffer collected so far.
                mean, std = self._fit_scaler(
                    np.array(list(self._buffer), dtype=np.float32)
                )
            else:
                return []
            names = self._resolve_feature_names(fv.shape[0])
        if fv.shape[0] != mean.shape[0]:
            return []
        z = (fv - mean) / std
        order = np.argsort(np.abs(z))[::-1][:k]
        return [f"{names[i]} {z[i]:+.1f}σ" for i in order]

    def _normalize_score(self, raw_score: float) -> float:
        denom = self._score_max - self._score_min
        if denom < 1e-8:
            return 0.0
        norm = (raw_score - self._score_min) / denom
        return float(np.clip(1.0 - norm, 0.0, 1.0))
