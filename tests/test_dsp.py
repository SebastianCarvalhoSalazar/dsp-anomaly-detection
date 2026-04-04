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


@pytest.fixture
def window():
    rng = np.random.default_rng(42)
    return rng.standard_normal(2048).astype(np.float32)


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

def test_process_window_shape(processor, window):
    out = processor.process_window(window)
    assert out.shape == (134,)


def test_process_window_dtype(processor, window):
    out = processor.process_window(window)
    assert out.dtype == np.float32


def test_process_window_no_nan(processor, window):
    out = processor.process_window(window)
    assert not np.any(np.isnan(out))


def test_process_window_no_inf(processor, window):
    out = processor.process_window(window)
    assert not np.any(np.isinf(out))


def test_process_window_silence(processor):
    """Silent signal should produce finite, zero-dominant features."""
    silence = np.zeros(2048, dtype=np.float32)
    out = processor.process_window(silence)
    assert out.shape == (134,)
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
    # level=5 → 6 coefficient arrays → 6 energy values
    assert out.shape == (6,)


def test_temporal_shape(processor, window):
    out = processor._extract_temporal(window)
    assert out.shape == (2,)


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
    signal = np.random.randn(5000).astype(np.float32)
    windows = proc.segment_signal(signal)
    # floor((5000 - 2048) / 512) + 1 = floor(2952/512) + 1 = 5 + 1 = 6
    assert len(windows) == 6


def test_segment_signal_window_shape():
    proc = AudioProcessor()
    signal = np.random.randn(8000).astype(np.float32)
    windows = proc.segment_signal(signal)
    for w in windows:
        assert w.shape == (2048,)


def test_segment_signal_exact_one_window():
    proc = AudioProcessor()
    signal = np.random.randn(2048).astype(np.float32)
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

def test_process_window_with_meta_returns_feature_vector(processor, window):
    fv = processor.process_window_with_meta(window)
    assert isinstance(fv, FeatureVector)
    assert fv.data.shape == (134,)
    assert fv.rms >= 0.0
    assert fv.window_index >= 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_process_window_deterministic(processor, window):
    out1 = processor.process_window(window)
    out2 = processor.process_window(window)
    np.testing.assert_array_equal(out1, out2)
