from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DSPConfig:
    """Configuration for the audio DSP pipeline."""

    sample_rate: int = 16000
    # Powers-of-2 required by Kymatio's filter bank construction
    window_size: int = 4096
    # 75% overlap between windows
    hop_size: int = 1024
    # Daubechies-4: good time-frequency resolution for transient audio
    wavelet: str = "db4"
    # level=7 yields 8 coefficient arrays (1 approximation + 7 detail)
    wavelet_level: int = 7
    # J=8 < log2(4096)=12; covers down to ~62 Hz at 16 kHz sample rate
    scattering_J: int = 8
    # Filters per octave: Q=8 balances frequency resolution vs computation
    scattering_Q: int = 8

    # -- Spectral features (new) ------------------------------------------------
    spectral_rolloff_percentile: float = 0.85

    # -- Delta features (new) ---------------------------------------------------
    # Enable first-order deltas for RMS, spectral centroid, scattering energy
    enable_deltas: bool = True

    # feature_dim is computed dynamically at AudioProcessor init; this field
    # is overwritten after the scattering filter bank is built.
    feature_dim: int = field(default=0, init=False)
