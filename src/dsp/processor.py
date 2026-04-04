import time

import numpy as np
import pywt

from .config import DSPConfig
from .types import FeatureVector

# kymatio.numpy top-level import fails in scipy>=1.17 because
# kymatio.scattering3d.filter_bank uses scipy.special.sph_harm which was
# removed. Import directly from the 1D submodule to avoid that import chain.
from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D


class AudioProcessor:
    """Extracts 134-dimensional feature vectors from raw audio windows.

    The feature vector concatenates:
    - 126 Kymatio Scattering1D coefficients (mean-pooled over time)
    - 6 per-level wavelet energy values (db4, level=5)
    - 1 RMS amplitude
    - 1 Zero Crossing Rate
    """

    def __init__(self, config: DSPConfig | None = None) -> None:
        self._config = config or DSPConfig()
        # Instantiate filter bank once; reused across all process_window calls
        self._scattering = ScatteringNumPy1D(
            J=self._config.scattering_J,
            Q=self._config.scattering_Q,
            shape=(self._config.window_size,),
        )
        self._window_counter = 0

    def process_window(self, window: np.ndarray) -> np.ndarray:
        """Extract a 134-dim feature vector from a single audio window.

        Parameters
        ----------
        window : np.ndarray
            Audio samples of shape (window_size,), any dtype (cast to float32).

        Returns
        -------
        np.ndarray
            Feature vector of shape (feature_dim,), float32.
        """
        window = window.astype(np.float32)
        scattering_feats = self._extract_scattering(window)   # (126,)
        wavelet_feats = self._extract_wavelet_energy(window)  # (6,)
        temporal_feats = self._extract_temporal(window)       # (2,)
        return np.concatenate([scattering_feats, wavelet_feats, temporal_feats]).astype(np.float32)

    def process_window_with_meta(self, window: np.ndarray) -> FeatureVector:
        """Extract features and return a FeatureVector with metadata."""
        data = self.process_window(window)
        rms = float(np.sqrt(np.mean(window.astype(np.float32) ** 2)))
        idx = self._window_counter
        self._window_counter += 1
        return FeatureVector(data=data, timestamp=time.time(), window_index=idx, rms=rms)

    def segment_signal(self, signal: np.ndarray) -> list[np.ndarray]:
        """Slice a 1-D audio signal into overlapping windows.

        Parameters
        ----------
        signal : np.ndarray
            1-D audio signal of any length >= window_size.

        Returns
        -------
        list[np.ndarray]
            Windows of shape (window_size,). Trailing samples that do not
            fill a complete window are discarded.
        """
        win = self._config.window_size
        hop = self._config.hop_size
        windows = []
        start = 0
        while start + win <= len(signal):
            windows.append(signal[start : start + win])
            start += hop
        return windows

    def _extract_scattering(self, window: np.ndarray) -> np.ndarray:
        # ScatteringNumPy1D expects shape (batch, time) or (time,)
        # Output shape: (n_coeffs, time_steps); mean over time → (n_coeffs,)
        out = self._scattering(window)  # (n_coeffs, T)
        return out.mean(axis=-1).astype(np.float32)

    def _extract_wavelet_energy(self, window: np.ndarray) -> np.ndarray:
        # wavedec returns [cA_n, cD_n, ..., cD_1] — level+1 arrays for level=5 → 6
        coeffs = pywt.wavedec(window, self._config.wavelet, level=self._config.wavelet_level)
        energies = np.array([np.sum(c ** 2) for c in coeffs], dtype=np.float32)
        return energies

    def _extract_temporal(self, window: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(window ** 2))
        # ZCR: fraction of consecutive-sample sign changes
        zcr = np.mean(np.abs(np.diff(np.sign(window))) > 0)
        return np.array([rms, zcr], dtype=np.float32)
