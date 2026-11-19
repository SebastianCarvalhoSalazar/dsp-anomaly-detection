from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectorConfig:
    """Configuration for the anomaly detector."""

    n_estimators: int = 200
    # 'auto' → offset_ = -0.5 as in the original IsolationForest paper (only
    # used as decision threshold when enable_adaptive_threshold=False)
    contamination: float | str = "auto"
    max_samples: int | str = "auto"
    random_state: int = 42
    # Minimum samples collected before first fit. Live windows do not overlap
    # (4096 samples @ 16 kHz = 0.256 s), so 500 windows ≈ 128 s of audio.
    buffer_size: int = 500
    # Refit the model after this many new samples post-initial-fit (sliding window)
    refit_every: int = 200
    feature_dim: int = 0  # set dynamically from AudioProcessor.feature_dim

    # -- Z-score scaling ---------------------------------------------------------
    # Per-feature scaling fitted on the training buffer and frozen with the
    # model (F1). Needed because PCA is scale-sensitive; IsolationForest alone
    # is invariant to per-feature affine rescaling.
    enable_zscore: bool = True

    # -- Quarantine of anomalous windows (F2) ------------------------------------
    # Windows flagged as anomalous do not enter the training buffer, so a long
    # anomaly cannot become "normal" at the next refit.
    quarantine_anomalies: bool = True
    # ...unless it persists this long (≈60 s at 0.256 s/window): then it is
    # accepted as the new normal and a refit is forced.
    accept_after_windows: int = 234

    # -- Score smoothing / hysteresis (2.4) ------------------------------------
    ema_alpha: float = 0.3  # weight for current raw score in EMA
    # Number of consecutive anomalous windows to confirm an anomaly
    hysteresis_count: int = 3

    # -- Decision threshold (F2) -------------------------------------------------
    # True: threshold = ``adaptive_percentile`` of the TRAINING-buffer scores,
    # fixed at each fit. False: IsolationForest offset_.
    enable_adaptive_threshold: bool = True
    # Rolling window of recent scores kept for the dashboard (score_mean)
    adaptive_score_window: int = 500
    # Percentile of training scores below which a window is anomalous
    adaptive_percentile: float = 2.0

    # -- Persistent baseline (3.4) ---------------------------------------------
    state_path: str = "data/detector_state.pkl"

    # -- Score calibration -----------------------------------------------------
    # Fix C2: calibration bounds come from IsolationForest scores on the
    # training buffer, so unseen points more anomalous than anything in
    # training would clip to exactly 1.0 and become indistinguishable.
    # We widen the lower bound by ``calibration_margin * (max - min)`` so
    # the most anomalous *training* point maps below 1.0, leaving headroom
    # for worse-than-training anomalies. Only affects the [0,1] display
    # score; anomaly *decisions* use the raw score vs the threshold.
    calibration_margin: float = 0.5

    # -- PCA dimensionality reduction ---------------------------------------
    enable_pca: bool = True
    pca_components: int = 25  # 99% variance in ~9; 25 adds safety margin

    # -- Drift detection (C2ST) -----------------------------------------------
    enable_drift_detection: bool = True
    # Number of trees in the RandomForest classifier used for the
    # Classifier Two-Sample Test (C2ST).
    c2st_n_estimators: int = 50

    # -- Drift-aware refits (Req 7) -------------------------------------------
    # When drift_auc exceeds the threshold, shorten the refit interval so the
    # model adapts faster to a changing environment. Off by default to
    # preserve the fixed-interval behaviour; the pipeline enables it.
    enable_drift_aware_refit: bool = False
    drift_refit_threshold: float = 0.8
    # Effective interval under drift = refit_every * factor (clamped below).
    drift_refit_factor: float = 0.5
    min_refit_interval: int = 50
