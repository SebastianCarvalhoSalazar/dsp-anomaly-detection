from dataclasses import dataclass


@dataclass
class DSPConfig:
    """Configuration for the audio DSP pipeline."""

    sample_rate: int = 16000
    # Powers-of-2 required by Kymatio's filter bank construction
    window_size: int = 2048
    # 75% overlap between windows
    hop_size: int = 512
    # Daubechies-4: good time-frequency resolution for transient audio
    wavelet: str = "db4"
    # level=5 yields 6 coefficient arrays (1 approximation + 5 detail)
    wavelet_level: int = 5
    # J=6 < log2(2048)=11; controls maximum scale of scattering transform
    scattering_J: int = 6
    # Filters per octave: Q=8 balances frequency resolution vs computation
    scattering_Q: int = 8
    # 126 scattering + 6 wavelet energies + 1 RMS + 1 ZCR
    feature_dim: int = 134
