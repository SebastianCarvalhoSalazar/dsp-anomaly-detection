import threading
import time

import numpy as np
import pytest

from src.detection import AnomalyDetector, AnomalyResult, DetectorConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    """Fresh detector for each test."""
    return AnomalyDetector()


@pytest.fixture
def small_config():
    """Smaller buffer so tests run faster."""
    return DetectorConfig(buffer_size=50, refit_every=25, n_estimators=20)


def make_normal_vectors(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, 0.5, size=134).astype(np.float32) for _ in range(n)]


def make_outlier_vectors(n: int, seed: int = 99) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, 8.0, size=134).astype(np.float32) for _ in range(n)]


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
    det = AnomalyDetector(DetectorConfig(buffer_size=200, n_estimators=10))
    vecs = make_normal_vectors(200)
    for i, fv in enumerate(vecs):
        result = det.score(fv)
    assert result.is_fitted


# ---------------------------------------------------------------------------
# Score range
# ---------------------------------------------------------------------------

def test_anomaly_score_in_range():
    det = AnomalyDetector(DetectorConfig(buffer_size=50, refit_every=25, n_estimators=10))
    vecs = make_normal_vectors(60)
    for fv in vecs:
        result = det.score(fv)
        if result.is_fitted:
            assert 0.0 <= result.anomaly_score <= 1.0


# ---------------------------------------------------------------------------
# Outliers score higher than normals
# ---------------------------------------------------------------------------

def test_outliers_score_higher_than_normals():
    config = DetectorConfig(buffer_size=50, refit_every=200, n_estimators=50, random_state=0)
    det = AnomalyDetector(config)

    # Fit on normal data
    normal_train = make_normal_vectors(50, seed=1)
    for fv in normal_train:
        det.score(fv)

    assert det.get_status()["is_fitted"]

    # Score a fresh batch of normals and outliers
    normal_test = make_normal_vectors(10, seed=2)
    outlier_test = make_outlier_vectors(10, seed=3)

    normal_scores = [det.score(fv).anomaly_score for fv in normal_test]
    outlier_scores = [det.score(fv).anomaly_score for fv in outlier_test]

    assert np.mean(outlier_scores) > np.mean(normal_scores)


# ---------------------------------------------------------------------------
# AnomalyResult fields
# ---------------------------------------------------------------------------

def test_result_has_feature_vector():
    config = DetectorConfig(buffer_size=50, n_estimators=10)
    det = AnomalyDetector(config)
    vecs = make_normal_vectors(51)
    for fv in vecs:
        result = det.score(fv)
    assert result.feature_vector.shape == (134,)


def test_result_timestamp_is_recent():
    config = DetectorConfig(buffer_size=50, n_estimators=10)
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
    config = DetectorConfig(buffer_size=50, n_estimators=10)
    det = AnomalyDetector(config)
    for fv in make_normal_vectors(50):
        det.score(fv)
    assert det.get_status()["is_fitted"]

    det.reset()

    status = det.get_status()
    assert not status["is_fitted"]
    assert status["buffer_fill"] == 0


def test_reset_then_refit():
    config = DetectorConfig(buffer_size=50, n_estimators=10)
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
    config = DetectorConfig(buffer_size=200, n_estimators=10)
    det = AnomalyDetector(config)
    errors = []

    def worker():
        try:
            rng = np.random.default_rng()
            for _ in range(30):
                fv = rng.standard_normal(134).astype(np.float32)
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
