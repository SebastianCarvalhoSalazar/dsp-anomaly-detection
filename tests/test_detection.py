import threading
import time

import numpy as np
import pytest

from src.detection import AnomalyDetector, AnomalyResult, DetectorConfig

# Feature dim from the new AudioProcessor (dynamic); use a reasonable
# default for synthetic test vectors.  The detector doesn't care about
# the specific dim — it adapts on first score() call.
_DIM = 150


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    """Fresh detector with hysteresis=1 for simpler unit tests."""
    return AnomalyDetector(
        DetectorConfig(
            buffer_size=200,
            refit_every=100,
            n_estimators=50,
            hysteresis_count=1,
            enable_adaptive_threshold=False,
        )
    )


@pytest.fixture
def small_config():
    """Smaller buffer so tests run faster."""
    return DetectorConfig(
        buffer_size=50, refit_every=25, n_estimators=20,
        hysteresis_count=1, enable_adaptive_threshold=False,
    )


def make_normal_vectors(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [
        rng.normal(0.0, 0.5, size=_DIM).astype(np.float32)
        for _ in range(n)
    ]


def make_outlier_vectors(n: int, seed: int = 99):
    rng = np.random.default_rng(seed)
    return [
        rng.normal(0.0, 8.0, size=_DIM).astype(np.float32)
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Warmup phase
# ---------------------------------------------------------------------------

def test_warmup_returns_not_fitted(detector):
    normal_vecs = make_normal_vectors(199)
    for fv in normal_vecs:
        result = detector.score(fv)
        assert not result.is_fitted
        assert result.anomaly_score == 0.0


def test_fitted_after_buffer_full(detector):
    vecs = make_normal_vectors(200)
    for fv in vecs:
        result = detector.score(fv)
    assert result.is_fitted


def test_first_200th_triggers_fit():
    det = AnomalyDetector(DetectorConfig(
        buffer_size=200, n_estimators=10,
        hysteresis_count=1,
        enable_adaptive_threshold=False,
    ))
    vecs = make_normal_vectors(200)
    for fv in vecs:
        result = det.score(fv)
    assert result.is_fitted


# ---------------------------------------------------------------------------
# Score range
# ---------------------------------------------------------------------------

def test_anomaly_score_in_range():
    det = AnomalyDetector(DetectorConfig(
        buffer_size=50, refit_every=25, n_estimators=10,
        hysteresis_count=1, enable_adaptive_threshold=False,
    ))
    vecs = make_normal_vectors(60)
    for fv in vecs:
        result = det.score(fv)
        if result.is_fitted:
            assert 0.0 <= result.anomaly_score <= 1.0


# ---------------------------------------------------------------------------
# Outliers score higher than normals
# ---------------------------------------------------------------------------

def test_outliers_score_higher_than_normals():
    config = DetectorConfig(
        buffer_size=50, refit_every=200, n_estimators=50,
        random_state=0, hysteresis_count=1,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(config)

    # Fit on normal data
    normal_train = make_normal_vectors(50, seed=1)
    for fv in normal_train:
        det.score(fv)

    assert det.get_status()["is_fitted"]

    # Score fresh normals and outliers
    normal_test = make_normal_vectors(10, seed=2)
    outlier_test = make_outlier_vectors(10, seed=3)

    normal_scores = [
        det.score(fv).anomaly_score for fv in normal_test
    ]
    outlier_scores = [
        det.score(fv).anomaly_score for fv in outlier_test
    ]

    assert np.mean(outlier_scores) > np.mean(normal_scores)


# ---------------------------------------------------------------------------
# AnomalyResult fields
# ---------------------------------------------------------------------------

def test_result_has_feature_vector():
    config = DetectorConfig(
        buffer_size=50, n_estimators=10,
        hysteresis_count=1,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(config)
    vecs = make_normal_vectors(51)
    for fv in vecs:
        result = det.score(fv)
    assert result.feature_vector.shape == (_DIM,)


def test_result_timestamp_is_recent():
    config = DetectorConfig(
        buffer_size=50, n_estimators=10,
        hysteresis_count=1,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(config)
    before = time.time()
    vecs = make_normal_vectors(51)
    for fv in vecs:
        result = det.score(fv)
    after = time.time()
    assert before <= result.timestamp <= after


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_clears_fitted_state():
    config = DetectorConfig(
        buffer_size=50, n_estimators=10,
        hysteresis_count=1,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(config)
    for fv in make_normal_vectors(50):
        det.score(fv)
    assert det.get_status()["is_fitted"]

    det.reset()

    status = det.get_status()
    assert not status["is_fitted"]
    assert status["buffer_fill"] == 0


def test_reset_then_refit():
    config = DetectorConfig(
        buffer_size=50, n_estimators=10,
        hysteresis_count=1,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(config)
    for fv in make_normal_vectors(50):
        det.score(fv)
    det.reset()
    for fv in make_normal_vectors(50):
        result = det.score(fv)
    assert result.is_fitted


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_thread_safety():
    config = DetectorConfig(
        buffer_size=200, n_estimators=10,
        hysteresis_count=1,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(config)
    errors = []

    def worker():
        try:
            rng = np.random.default_rng()
            for _ in range(30):
                fv = rng.standard_normal(_DIM).astype(np.float32)
                det.score(fv)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread errors: {errors}"
    assert det.get_status()["buffer_fill"] <= config.buffer_size


# ---------------------------------------------------------------------------
# get_status()
# ---------------------------------------------------------------------------

def test_get_status_keys(detector):
    status = detector.get_status()
    assert "is_fitted" in status
    assert "buffer_fill" in status
    assert "buffer_size" in status
    assert "n_estimators" in status
    assert "smoothed_score" in status
    assert "consecutive_anomalies" in status


# ---------------------------------------------------------------------------
# Persistent baseline (3.4)
# ---------------------------------------------------------------------------

def test_save_and_load_state(tmp_path):
    path = str(tmp_path / "state.pkl")
    config = DetectorConfig(
        buffer_size=50, n_estimators=10,
        hysteresis_count=1,
        enable_adaptive_threshold=False,
        state_path=path,
    )
    det1 = AnomalyDetector(config)
    for fv in make_normal_vectors(50):
        det1.score(fv)
    assert det1.get_status()["is_fitted"]

    det1.save_state(path)

    det2 = AnomalyDetector(config)
    assert not det2.get_status()["is_fitted"]
    assert det2.load_state(path)
    assert det2.get_status()["is_fitted"]

    # Scoring should work immediately without warmup
    fv = make_normal_vectors(1)[0]
    result = det2.score(fv)
    assert result.is_fitted


def test_load_state_nonexistent(tmp_path):
    config = DetectorConfig(
        state_path=str(tmp_path / "missing.pkl")
    )
    det = AnomalyDetector(config)
    assert not det.load_state()


# ---------------------------------------------------------------------------
# EMA smoothing (2.4)
# ---------------------------------------------------------------------------

def test_ema_smoothing_dampens_single_spike():
    """A single anomalous window should not trigger is_anomaly
    when hysteresis_count > 1."""
    config = DetectorConfig(
        buffer_size=50, refit_every=200, n_estimators=50,
        random_state=0,
        hysteresis_count=3,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(config)

    for fv in make_normal_vectors(50, seed=1):
        det.score(fv)

    # One outlier followed by normals
    outlier = make_outlier_vectors(1, seed=5)[0]
    r = det.score(outlier)
    # Single spike — hysteresis prevents triggering
    assert not r.is_anomaly


# ---------------------------------------------------------------------------
# Z-score normalization (2.3)
# ---------------------------------------------------------------------------

def test_zscore_does_not_crash():
    """Z-score normalizer should work from the first sample."""
    config = DetectorConfig(
        buffer_size=50, n_estimators=10,
        enable_zscore=True,
        hysteresis_count=1,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(config)
    vecs = make_normal_vectors(60)
    for fv in vecs:
        result = det.score(fv)
    assert result.is_fitted


# -------------------------------------------------------------------
# PCA dimensionality reduction
# -------------------------------------------------------------------

def test_pca_reduces_dimensionality():
    """With PCA enabled the IF receives fewer features."""
    cfg = DetectorConfig(
        buffer_size=50,
        refit_every=25,
        n_estimators=20,
        hysteresis_count=1,
        enable_adaptive_threshold=False,
        enable_pca=True,
        pca_components=10,
    )
    det = AnomalyDetector(cfg)
    for fv in make_normal_vectors(60):
        det.score(fv)

    assert det._is_fitted
    assert det._pca is not None
    assert det._pca.n_components == 10


def test_pca_disabled_by_default_override():
    """With PCA disabled, no PCA object is created."""
    cfg = DetectorConfig(
        buffer_size=50,
        refit_every=25,
        n_estimators=20,
        hysteresis_count=1,
        enable_adaptive_threshold=False,
        enable_pca=False,
    )
    det = AnomalyDetector(cfg)
    for fv in make_normal_vectors(60):
        det.score(fv)

    assert det._is_fitted
    assert det._pca is None


def test_pca_scoring_gives_valid_results():
    """PCA-enabled detector still produces valid 0-1 scores."""
    cfg = DetectorConfig(
        buffer_size=50,
        refit_every=25,
        n_estimators=20,
        hysteresis_count=1,
        enable_pca=True,
        pca_components=10,
    )
    det = AnomalyDetector(cfg)
    for fv in make_normal_vectors(55):
        r = det.score(fv)

    assert r.is_fitted
    assert 0.0 <= r.anomaly_score <= 1.0


def test_pca_status_fields():
    """get_status() exposes PCA info after fitting."""
    cfg = DetectorConfig(
        buffer_size=50,
        refit_every=25,
        n_estimators=20,
        enable_pca=True,
        pca_components=15,
        hysteresis_count=1,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(cfg)

    # Before fitting: pca_components == 0 (no PCA yet)
    status = det.get_status()
    assert status["pca_enabled"] is True
    assert status["pca_components"] == 0

    # After fitting: components reflect config
    for fv in make_normal_vectors(55):
        det.score(fv)
    status = det.get_status()
    assert status["pca_enabled"] is True
    assert status["pca_components"] == 15


# -------------------------------------------------------------------
# Drift detection
# -------------------------------------------------------------------

def test_drift_metrics_keys():
    """get_drift_metrics returns expected keys."""
    cfg = DetectorConfig(
        buffer_size=50,
        refit_every=25,
        n_estimators=20,
        hysteresis_count=1,
        enable_drift_detection=True,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(cfg)
    m = det.get_drift_metrics()
    expected = {
        "adaptive_threshold",
        "score_mean",
        "drift_auc",
        "top_drift_features",
        "refit_count",
        "refit_reason",
    }
    assert set(m.keys()) == expected


def test_refit_count_increments():
    """refit_count grows each time the model refits."""
    cfg = DetectorConfig(
        buffer_size=30,
        refit_every=30,
        n_estimators=10,
        hysteresis_count=1,
        enable_drift_detection=True,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(cfg)
    for fv in make_normal_vectors(35):
        det.score(fv)

    assert det.get_drift_metrics()["refit_count"] >= 1


def test_feature_drift_after_shift():
    """C2ST drift AUC should be high after distribution shift."""
    cfg = DetectorConfig(
        buffer_size=30,
        refit_every=30,
        n_estimators=10,
        hysteresis_count=1,
        enable_drift_detection=True,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(cfg)

    # Phase 1: fit on normal
    for fv in make_normal_vectors(35, seed=0):
        det.score(fv)

    auc_1 = det.get_drift_metrics()["drift_auc"]
    # First refit has no previous buffer → default 0.5
    assert auc_1 == pytest.approx(0.5, abs=0.01)

    # Phase 2: feed shifted distribution to trigger refit
    rng = np.random.default_rng(42)
    shifted = [
        rng.normal(5.0, 0.5, size=_DIM).astype(np.float32)
        for _ in range(35)
    ]
    for fv in shifted:
        det.score(fv)

    m = det.get_drift_metrics()
    # Easy to separate → AUC should be well above 0.5
    assert m["drift_auc"] > 0.7
    # Should report which features changed
    assert len(m["top_drift_features"]) > 0


def test_pca_save_load_state(tmp_path):
    """PCA state survives save/load cycle."""
    cfg = DetectorConfig(
        buffer_size=50,
        refit_every=25,
        n_estimators=20,
        hysteresis_count=1,
        enable_pca=True,
        pca_components=10,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(cfg)
    for fv in make_normal_vectors(55):
        det.score(fv)

    assert det._pca is not None
    save_file = tmp_path / "state.pkl"
    det.save_state(str(save_file))

    det2 = AnomalyDetector(cfg)
    det2.load_state(str(save_file))
    assert det2._pca is not None
    assert det2._refit_count == det._refit_count


def test_reset_clears_drift_state():
    """reset() should zero out all drift tracking."""
    cfg = DetectorConfig(
        buffer_size=30,
        refit_every=30,
        n_estimators=10,
        hysteresis_count=1,
        enable_pca=True,
        pca_components=5,
        enable_drift_detection=True,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(cfg)
    for fv in make_normal_vectors(35):
        det.score(fv)

    det.reset()
    m = det.get_drift_metrics()
    assert m["refit_count"] == 0
    assert m["drift_auc"] == 0.5
    assert m["top_drift_features"] == []
    assert det._pca is None


def test_c2st_no_drift_low_auc():
    """Same distribution in both buffers → AUC near 0.5.

    With 200 samples and 150 dims the RF cross-val AUC
    should stay moderate when both buffers come from the
    same N(0, 0.5) distribution.
    """
    cfg = DetectorConfig(
        buffer_size=200,
        refit_every=200,
        n_estimators=10,
        hysteresis_count=1,
        enable_drift_detection=True,
        enable_adaptive_threshold=False,
    )
    det = AnomalyDetector(cfg)
    # Two consecutive fits on same distribution
    for fv in make_normal_vectors(420, seed=0):
        det.score(fv)

    m = det.get_drift_metrics()
    # Cannot separate → AUC should stay well below 0.85
    assert m["drift_auc"] < 0.85


def test_feature_names_builder():
    """_build_feature_names returns correct length."""
    names = AnomalyDetector._build_feature_names(251)
    assert len(names) == 251
    assert names[0] == "scat_0"
    assert "rms" in names
    assert "spectral_centroid" in names
    assert "delta_rms" in names
    assert names[-1] == "delta_scat_energy"
