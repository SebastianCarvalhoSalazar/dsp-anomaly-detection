import time
from typing import Optional

import numpy as np
import pywt

from .config import DSPConfig
from .types import FeatureVector

# kymatio.numpy top-level import fails in scipy>=1.17 because
# kymatio.scattering3d.filter_bank uses scipy.special.sph_harm which was
# removed.  Import directly from the 1D submodule to avoid that chain.
from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D


class AudioProcessor:
    """Extracts feature vectors from raw audio windows.

    The feature vector concatenates (dimensions depend on config):
      - Kymatio Scattering1D coefficients (mean-pooled over time)
      - Per-level wavelet energy values (db4)
      - RMS amplitude
      - Zero Crossing Rate
      - Spectral centroid, flatness, rolloff, bandwidth   [NEW]
      - Delta RMS, delta centroid, delta scattering energy [NEW]

    ``feature_dim`` is computed dynamically after building the
    scattering filter bank and written back to ``config.feature_dim``.
    """

    def __init__(self, config: DSPConfig | None = None) -> None:
        self._config = config or DSPConfig()
        self._scattering = ScatteringNumPy1D(
            J=self._config.scattering_J,
            Q=self._config.scattering_Q,
            shape=(self._config.window_size,),
        )
        self._window_counter = 0

        # State for delta features (previous window)
        self._prev_rms: Optional[float] = None
        self._prev_centroid: Optional[float] = None
        self._prev_scat_energy: Optional[float] = None

        # --- Compute feature_dim dynamically -------------------------
        probe = np.zeros(self._config.window_size, dtype=np.float32)
        n_scat = self._extract_scattering(probe).shape[0]
        n_wavelet = self._config.wavelet_level + 1  # approx + details
        n_temporal = 2  # RMS + ZCR
        n_spectral = 4  # centroid, flatness, rolloff, bandwidth
        n_delta = 3 if self._config.enable_deltas else 0
        self._n_scattering = n_scat
        self._config.feature_dim = (
            n_scat + n_wavelet + n_temporal + n_spectral + n_delta
        )

    @property
    def feature_dim(self) -> int:
        return self._config.feature_dim

    @property
    def feature_names(self) -> list[str]:
        """Human-readable name for each feature, in vector order.

        Built from the *actual* config (scattering count, wavelet level,
        deltas on/off) so downstream drift / explainability labels stay
        correct even when the layout changes — unlike a hardcoded layout
        in the detector (fix H6).
        """
        names: list[str] = [f"scat_{i}" for i in range(self._n_scattering)]
        names += [
            f"wavelet_band_{k}"
            for k in range(self._config.wavelet_level + 1)
        ]
        names += ["rms", "zcr"]
        names += [
            "spectral_centroid",
            "spectral_flatness",
            "spectral_rolloff",
            "spectral_bandwidth",
        ]
        if self._config.enable_deltas:
            names += ["delta_rms", "delta_centroid", "delta_scat_energy"]
        return names

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def process_window(self, window: np.ndarray) -> np.ndarray:
        """Extract a feature vector from a single audio window.

        Parameters
        ----------
        window : np.ndarray
            Audio samples of shape (window_size,), any dtype
            (cast to float32 internally).

        Returns
        -------
        np.ndarray
            Feature vector of shape (feature_dim,), float32.
        """
        window = window.astype(np.float32)

        scat = self._extract_scattering(window)
        wavelet = self._extract_wavelet_energy(window)
        temporal = self._extract_temporal(window)
        spectral = self._extract_spectral(window)

        parts = [scat, wavelet, temporal, spectral]

        if self._config.enable_deltas:
            deltas = self._extract_deltas(
                rms=temporal[0],
                centroid=spectral[0],
                scat_energy=float(np.sum(scat ** 2)),
            )
            parts.append(deltas)

        return np.concatenate(parts).astype(np.float32)

    def process_window_with_meta(
        self, window: np.ndarray
    ) -> FeatureVector:
        """Extract features and return a FeatureVector with metadata."""
        data = self.process_window(window)
        rms = float(
            np.sqrt(np.mean(window.astype(np.float32) ** 2))
        )
        idx = self._window_counter
        self._window_counter += 1
        return FeatureVector(
            data=data, timestamp=time.time(),
            window_index=idx, rms=rms,
        )

    def segment_signal(self, signal: np.ndarray) -> list[np.ndarray]:
        """Slice a 1-D audio signal into overlapping windows.

        Trailing samples that do not fill a complete window are discarded.
        """
        win = self._config.window_size
        hop = self._config.hop_size
        windows: list[np.ndarray] = []
        start = 0
        while start + win <= len(signal):
            windows.append(signal[start: start + win])
            start += hop
        return windows

    # ------------------------------------------------------------------ #
    #  Feature extractors                                                 #
    # ------------------------------------------------------------------ #

    def _extract_scattering(self, window: np.ndarray) -> np.ndarray:
        out = self._scattering(window)  # (n_coeffs, T)
        return out.mean(axis=-1).astype(np.float32)

    def _extract_wavelet_energy(
        self, window: np.ndarray
    ) -> np.ndarray:
        coeffs = pywt.wavedec(
            window, self._config.wavelet,
            level=self._config.wavelet_level,
        )
        return np.array(
            [np.sum(c ** 2) for c in coeffs], dtype=np.float32
        )

    def _extract_temporal(self, window: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(window ** 2))
        zcr = np.mean(np.abs(np.diff(np.sign(window))) > 0)
        return np.array([rms, zcr], dtype=np.float32)

    # -- NEW: spectral features ----------------------------------------

    def _extract_spectral(self, window: np.ndarray) -> np.ndarray:
        """Compute spectral centroid, flatness, rolloff, bandwidth.

        Uses numpy FFT (faster than librosa for single windows).
        """
        n = len(window)
        spectrum = np.abs(np.fft.rfft(window))
        freqs = np.fft.rfftfreq(n, d=1.0 / self._config.sample_rate)

        # Avoid division by zero on silence
        mag_sum = spectrum.sum()
        if mag_sum < 1e-10:
            return np.zeros(4, dtype=np.float32)

        # Spectral centroid
        centroid = float(np.sum(freqs * spectrum) / mag_sum)

        # Spectral bandwidth (std dev weighted by magnitude)
        bandwidth = float(
            np.sqrt(
                np.sum(((freqs - centroid) ** 2) * spectrum)
                / mag_sum
            )
        )

        # Spectral rolloff (freq below which rolloff% energy)
        cumsum = np.cumsum(spectrum)
        rolloff_thresh = (
            self._config.spectral_rolloff_percentile * mag_sum
        )
        rolloff_idx = np.searchsorted(cumsum, rolloff_thresh)
        rolloff_idx = min(rolloff_idx, len(freqs) - 1)
        rolloff = float(freqs[rolloff_idx])

        # Spectral flatness (geometric / arithmetic mean)
        eps = 1e-10
        log_spec = np.log(spectrum + eps)
        geo_mean = np.exp(log_spec.mean())
        arith_mean = spectrum.mean()
        flatness = float(geo_mean / (arith_mean + eps))

        return np.array(
            [centroid, flatness, rolloff, bandwidth],
            dtype=np.float32,
        )

    # -- NEW: delta features -------------------------------------------

    def _extract_deltas(
        self,
        rms: float,
        centroid: float,
        scat_energy: float,
    ) -> np.ndarray:
        """First-order deltas: change from previous window."""
        if self._prev_rms is None:
            delta_rms = 0.0
            delta_centroid = 0.0
            delta_scat = 0.0
        else:
            delta_rms = rms - self._prev_rms
            delta_centroid = centroid - self._prev_centroid
            delta_scat = scat_energy - self._prev_scat_energy

        self._prev_rms = rms
        self._prev_centroid = centroid
        self._prev_scat_energy = scat_energy

        return np.array(
            [delta_rms, delta_centroid, delta_scat],
            dtype=np.float32,
        )
