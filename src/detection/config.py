from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectorConfig:
    """Configuration for the anomaly detector."""

    n_estimators: int = 200
    # 'auto' sets contamination to 0.1 (appropriate for unknown anomaly rate)
    contamination: float | str = "auto"
    max_samples: int | str = "auto"
    random_state: int = 42
    # Minimum samples collected before first fit; ~64s of audio at 16kHz/4096/1024
    buffer_size: int = 500
    # Refit the model after this many new samples post-initial-fit (sliding window)
    refit_every: int = 200
    feature_dim: int = 0  # set dynamically from AudioProcessor.feature_dim

    # -- Z-score normalization (2.3) -------------------------------------------
    enable_zscore: bool = True

    # -- Score smoothing / hysteresis (2.4) ------------------------------------
    ema_alpha: float = 0.3  # weight for current raw score in EMA
    # Number of consecutive anomalous windows to confirm an anomaly
    hysteresis_count: int = 3

    # -- Adaptive threshold (3.5) ----------------------------------------------
    enable_adaptive_threshold: bool = True
    # Recent-scores window for percentile computation
    adaptive_score_window: int = 500
    # Percentile below which raw scores are considered anomalous (bottom 2%)
    adaptive_percentile: float = 2.0

    # -- Persistent baseline (3.4) ---------------------------------------------
    state_path: str = "data/detector_state.pkl"

    # -- PCA dimensionality reduction ---------------------------------------
    enable_pca: bool = True
    pca_components: int = 25  # 99% variance in ~9; 25 adds safety margin

    # -- Drift detection (C2ST) -----------------------------------------------
    enable_drift_detection: bool = True
    # Number of trees in the HistGradientBoosting classifier
    # used for the Classifier Two-Sample Test (C2ST).
    c2st_n_estimators: int = 50
