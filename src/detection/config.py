from dataclasses import dataclass


@dataclass
class DetectorConfig:
    """Configuration for the anomaly detector."""

    n_estimators: int = 100
    # 'auto' sets contamination to 0.1 (appropriate for unknown anomaly rate)
    contamination: float | str = "auto"
    max_samples: int | str = "auto"
    random_state: int = 42
    # Minimum samples collected before first fit; ~25s of audio at 16kHz/2048/512
    buffer_size: int = 200
    # Refit the model after this many new samples post-initial-fit (sliding window)
    refit_every: int = 100
    feature_dim: int = 134
