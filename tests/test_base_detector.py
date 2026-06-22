"""Regression tests for the Fase 0 BaseAnomalyDetector fixes.

Covers the review findings the refactor was meant to close:
  C1 — normalizer not polluted by anomalies once fitted
  C2 — calibration leaves headroom (no blanket saturation at 1.0)
  H1 — heavy fit is computed outside the lock (purity of _compute_fit)
  H6 — injected feature names used for drift output
  M3/M4 — full state persisted; feature_dim change resets cleanly
"""

import numpy as np
import pytest

from src.detection import AnomalyDetector, DetectorConfig

_DIM = 60


def _normals(n, dim=_DIM, seed=0, scale=0.5):
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, scale, size=dim).astype(np.float32) for _ in range(n)]


def _outliers(n, dim=_DIM, seed=99, scale=10.0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, scale, size=dim).astype(np.float32) for _ in range(n)]


def _cfg(**kw):
    base = dict(
        buffer_size=50,
        refit_every=200,
        n_estimators=30,
        hysteresis_count=1,
        enable_adaptive_threshold=False,
        enable_pca=False,
        random_state=0,
    )
    base.update(kw)
    return DetectorConfig(**base)


# --------------------------------------------------------------------------- #
# C1 — normalizer freeze on anomaly
# --------------------------------------------------------------------------- #

def test_c1_normalizer_skips_anomalous_windows_when_frozen():
    det = AnomalyDetector(_cfg(freeze_normalizer_on_anomaly=True))
    for fv in _normals(50, seed=1):
        det.score(fv)
    n_before = det._normalizer._n
    for fv in _outliers(30, seed=2):
        det.score(fv)
    # Some clearly-anomalous windows must have been excluded from the
    # running statistics → fewer than 30 updates folded in.
    assert det._normalizer._n - n_before < 30


def test_c1_legacy_path_updates_every_window():
    det = AnomalyDetector(_cfg(freeze_normalizer_on_anomaly=False))
    for fv in _normals(50, seed=1):
        det.score(fv)
    n_before = det._normalizer._n
    for fv in _outliers(30, seed=2):
        det.score(fv)
    assert det._normalizer._n - n_before == 30


# --------------------------------------------------------------------------- #
# C2 — calibration headroom
# --------------------------------------------------------------------------- #

def test_c2_margin_leaves_headroom_for_training_min():
    """The most anomalous *training* point must map below 1.0 so that
    worse-than-training anomalies remain distinguishable."""
    det = AnomalyDetector(_cfg(calibration_margin=0.5, enable_zscore=False))
    for fv in _normals(50, seed=1):
        det.score(fv)
    # The raw training minimum corresponds to score_min + margin span.
    raw_train_min = det._score_min + det._config.calibration_margin * (
        det._score_max - det._score_min
    ) / (1 + det._config.calibration_margin)
    assert det._normalize_score(raw_train_min) < 1.0


def test_c2_zero_margin_restores_saturation():
    det = AnomalyDetector(_cfg(calibration_margin=0.0, enable_zscore=False))
    for fv in _normals(50, seed=1):
        det.score(fv)
    # With no margin the training minimum maps to exactly 1.0 (legacy).
    assert det._normalize_score(det._score_min) == pytest.approx(1.0)


def test_c2_scores_stay_in_unit_range():
    det = AnomalyDetector(_cfg(calibration_margin=0.5))
    for fv in _normals(55, seed=1):
        r = det.score(fv)
    for fv in _outliers(10, seed=7):
        r = det.score(fv)
        assert 0.0 <= r.anomaly_score <= 1.0


# --------------------------------------------------------------------------- #
# H1 — fit computed outside the lock (purity)
# --------------------------------------------------------------------------- #

def test_h1_compute_fit_does_not_mutate_state():
    det = AnomalyDetector(_cfg())
    for fv in _normals(50, seed=1):
        det.score(fv)
    refit_before = det._refit_count
    X = np.array(list(det._buffer), dtype=np.float32)
    result = det._compute_fit(X, None)
    # _compute_fit is pure: refit_count only changes in _apply_fit.
    assert det._refit_count == refit_before
    assert "model" in result and "score_min" in result


# --------------------------------------------------------------------------- #
# H6 — injected feature names
# --------------------------------------------------------------------------- #

def test_h6_injected_feature_names_used_in_drift():
    names = [f"feat_{i}" for i in range(_DIM)]
    det = AnomalyDetector(
        _cfg(buffer_size=40, refit_every=40, enable_drift_detection=True),
        feature_names=names,
    )
    # Two fits with a distribution shift between them → drift features set.
    for fv in _normals(40, seed=0):
        det.score(fv)
    for fv in _outliers(40, seed=1):
        det.score(fv)
    top = det.get_drift_metrics()["top_drift_features"]
    assert top  # non-empty
    assert all(t.startswith("feat_") for t in top)


# --------------------------------------------------------------------------- #
# M3 / M4 — state persistence and dim-change reset
# --------------------------------------------------------------------------- #

def test_m3_state_roundtrip_preserves_drift_and_scores(tmp_path):
    path = str(tmp_path / "state.pkl")
    det = AnomalyDetector(
        _cfg(buffer_size=40, refit_every=40, enable_drift_detection=True)
    )
    for fv in _normals(40, seed=0):
        det.score(fv)
    for fv in _outliers(40, seed=1):
        det.score(fv)
    det.save_state(path)

    det2 = AnomalyDetector(_cfg(buffer_size=40, refit_every=40))
    assert det2.load_state(path)
    assert det2._refit_count == det._refit_count
    assert det2._drift_auc == det._drift_auc
    assert list(det2._recent_norm_scores) == list(det._recent_norm_scores)
    assert det2._samples_since_refit == det._samples_since_refit


def test_m4_feature_dim_change_resets_cleanly():
    det = AnomalyDetector(_cfg(buffer_size=40))
    for fv in _normals(40, dim=_DIM, seed=0):
        det.score(fv)
    assert det.get_status()["is_fitted"]
    # A vector of a different dimension must not crash; it resets and rewarms.
    r = det.score(np.zeros(_DIM + 5, dtype=np.float32))
    assert not r.is_fitted
    assert det._feature_dim == _DIM + 5
