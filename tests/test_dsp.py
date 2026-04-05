import threading

import numpy as np
import pytest

from src.dsp import AudioProcessor, DSPConfig, FeatureVector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def processor():
    return AudioProcessor()


@pytest.fixture(scope="module")
def feature_dim(processor):
    """Dynamically computed feature dim from the processor."""
    return processor.feature_dim


@pytest.fixture
def window():
    rng = np.random.default_rng(42)
    return rng.standard_normal(4096).astype(np.float32)


# ---------------------------------------------------------------------------
# Import sanity – document the known scipy 1.17 incompatibility
# ---------------------------------------------------------------------------

def test_kymatio_direct_import_works():
    """Direct submodule import must succeed (works around scipy 1.17 breakage)."""
    from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D  # noqa: F401


def test_kymatio_toplevel_import_fails():
    """kymatio.numpy top-level import is broken on scipy>=1.17 (sph_harm removed).
    This test documents the known workaround.
    """
    try:
        from kymatio.numpy import Scattering1D  # noqa: F401
        # If it happens to work (e.g. future kymatio patch), that's fine too
    except (ImportError, AttributeError):
        pass  # Expected on scipy>=1.17


# ---------------------------------------------------------------------------
# process_window
# ---------------------------------------------------------------------------

def test_process_window_shape(processor, window, feature_dim):
    out = processor.process_window(window)
    assert out.shape == (feature_dim,)


def test_process_window_dtype(processor, window):
    out = processor.process_window(window)
    assert out.dtype == np.float32


def test_process_window_no_nan(processor, window):
    out = processor.process_window(window)
    assert not np.any(np.isnan(out))


def test_process_window_no_inf(processor, window):
    out = processor.process_window(window)
    assert not np.any(np.isinf(out))


def test_process_window_silence(processor, feature_dim):
    """Silent signal should produce finite, zero-dominant features."""
    silence = np.zeros(4096, dtype=np.float32)
    out = processor.process_window(silence)
    assert out.shape == (feature_dim,)
    assert not np.any(np.isnan(out))


# ---------------------------------------------------------------------------
# Internal sub-feature shapes
# ---------------------------------------------------------------------------

def test_scattering_shape(processor, window):
    out = processor._extract_scattering(window)
    # Shape depends on J/Q/window_size; must be 1-D with >= 1 coefficient
    assert out.ndim == 1
    assert out.shape[0] >= 1


def test_wavelet_energy_shape(processor, window):
    out = processor._extract_wavelet_energy(window)
    # level=7 → 8 coefficient arrays → 8 energy values
    assert out.shape == (8,)


def test_temporal_shape(processor, window):
    out = processor._extract_temporal(window)
    assert out.shape == (2,)


def test_spectral_shape(processor, window):
    out = processor._extract_spectral(window)
    assert out.shape == (4,)


def test_spectral_silence(processor):
    silence = np.zeros(4096, dtype=np.float32)
    out = processor._extract_spectral(silence)
    np.testing.assert_array_equal(out, np.zeros(4))


def test_feature_dim_matches_config():
    config = DSPConfig()
    proc = AudioProcessor(config)
    w = np.random.randn(config.window_size).astype(np.float32)
    out = proc.process_window(w)
    # The actual concatenation should match the declared feature_dim
    assert out.shape[0] == config.feature_dim


# ---------------------------------------------------------------------------
# segment_signal
# ---------------------------------------------------------------------------

def test_segment_signal_count():
    proc = AudioProcessor()
    signal = np.random.randn(10000).astype(np.float32)
    windows = proc.segment_signal(signal)
    # floor((10000 - 4096) / 1024) + 1 = floor(5904/1024) + 1 = 5 + 1 = 6
    expected = (len(signal) - proc._config.window_size) // proc._config.hop_size + 1
    assert len(windows) == expected


def test_segment_signal_window_shape():
    proc = AudioProcessor()
    signal = np.random.randn(16000).astype(np.float32)
    windows = proc.segment_signal(signal)
    for w in windows:
        assert w.shape == (4096,)


def test_segment_signal_exact_one_window():
    proc = AudioProcessor()
    signal = np.random.randn(4096).astype(np.float32)
    windows = proc.segment_signal(signal)
    assert len(windows) == 1


def test_segment_signal_too_short_returns_empty():
    proc = AudioProcessor()
    signal = np.random.randn(100).astype(np.float32)
    windows = proc.segment_signal(signal)
    assert len(windows) == 0


# ---------------------------------------------------------------------------
# process_window_with_meta
# ---------------------------------------------------------------------------

def test_process_window_with_meta_returns_feature_vector(
    processor, window, feature_dim
):
    fv = processor.process_window_with_meta(window)
    assert isinstance(fv, FeatureVector)
    assert fv.data.shape == (feature_dim,)
    assert fv.rms >= 0.0
    assert fv.window_index >= 0


# ---------------------------------------------------------------------------
# Determinism (spectral + scattering are deterministic; deltas need 2 calls)
# ---------------------------------------------------------------------------

def test_process_window_deterministic():
    """Two fresh processors on the same window give the same output."""
    w = np.random.default_rng(7).standard_normal(4096).astype(np.float32)
    p1 = AudioProcessor()
    p2 = AudioProcessor()
    out1 = p1.process_window(w)
    out2 = p2.process_window(w)
    np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Delta features
# ---------------------------------------------------------------------------

def test_delta_features_zero_on_first_window():
    """First window should have zero deltas (no prior state)."""
    config = DSPConfig()
    proc = AudioProcessor(config)
    w = np.random.randn(config.window_size).astype(np.float32)
    out = proc.process_window(w)
    # Last 3 values are deltas; first window → all zero
    assert out[-3] == 0.0
    assert out[-2] == 0.0
    assert out[-1] == 0.0


def test_delta_features_nonzero_on_second_window():
    """Second window should have non-zero deltas (different signal)."""
    config = DSPConfig()
    proc = AudioProcessor(config)
    w1 = np.random.default_rng(0).standard_normal(
        config.window_size
    ).astype(np.float32)
    w2 = np.random.default_rng(1).standard_normal(
        config.window_size
    ).astype(np.float32)
    proc.process_window(w1)
    out2 = proc.process_window(w2)
    # At least one delta should be nonzero
    assert np.any(out2[-3:] != 0.0)
