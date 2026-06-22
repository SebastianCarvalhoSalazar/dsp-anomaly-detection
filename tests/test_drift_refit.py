"""Tests for drift-aware refits, snapshots and explainability (Fase 3)."""

import numpy as np

from src.detection import AnomalyDetector, DetectorConfig, SnapshotStore

_DIM = 40


def _normals(n, seed=0, scale=0.5, dim=_DIM):
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, scale, size=dim).astype(np.float32) for _ in range(n)]


def _cfg(**kw):
    base = dict(
        buffer_size=40, refit_every=40, n_estimators=15,
        hysteresis_count=1, enable_adaptive_threshold=False,
    )
    base.update(kw)
    return DetectorConfig(**base)


# --------------------------------------------------------------------------- #
# Drift-aware refit (Req 7)
# --------------------------------------------------------------------------- #

def test_refit_reason_scheduled_by_default():
    det = AnomalyDetector(_cfg(enable_drift_aware_refit=False))
    for fv in _normals(85):
        det.score(fv)
    assert det.get_drift_metrics()["refit_reason"] in ("scheduled", "initial")


def _ramp(n, seed=1, dim=_DIM, step=0.2):
    """Progressively drifting stream so consecutive buffers always differ,
    keeping the C2ST drift signal alive across refits."""
    rng = np.random.default_rng(seed)
    return [
        (rng.normal(0.0, 0.5, size=dim) + i * step).astype(np.float32)
        for i in range(n)
    ]


def test_drift_aware_refits_more_often_than_fixed():
    """Under sustained drift, the drift-aware detector refits more often than
    an identical detector with the feature disabled, fed the same data."""
    common = dict(
        drift_refit_threshold=0.6, drift_refit_factor=0.25,
        min_refit_interval=5, enable_drift_detection=True,
    )
    drift_aware = AnomalyDetector(_cfg(enable_drift_aware_refit=True, **common))
    fixed = AnomalyDetector(_cfg(enable_drift_aware_refit=False, **common))

    data = _ramp(200, seed=1)
    for fv in data:
        drift_aware.score(fv)
    for fv in data:
        fixed.score(fv)

    aware_refits = drift_aware.get_drift_metrics()["refit_count"]
    fixed_refits = fixed.get_drift_metrics()["refit_count"]
    assert aware_refits > fixed_refits
    assert drift_aware.get_drift_metrics()["refit_reason"] == "drift"


# --------------------------------------------------------------------------- #
# Snapshots (Req 8)
# --------------------------------------------------------------------------- #

def test_snapshots_saved_on_refit(tmp_path):
    store = SnapshotStore(str(tmp_path / "snaps"), max_snapshots=5)
    det = AnomalyDetector(_cfg(), snapshot_store=store)
    for fv in _normals(120):
        det.score(fv)
    snaps = store.list_snapshots()
    assert len(snaps) >= 1
    meta = store.load_metadata(snaps[-1])
    assert {"drift_auc", "refit_reason", "buffer_mean", "n_samples"} <= set(meta)


def test_snapshot_retention_prunes_oldest(tmp_path):
    store = SnapshotStore(str(tmp_path / "snaps"), max_snapshots=2)
    det = AnomalyDetector(_cfg(buffer_size=30, refit_every=15), snapshot_store=store)
    for fv in _normals(150):
        det.score(fv)
    assert len(store.list_snapshots()) <= 2


# --------------------------------------------------------------------------- #
# Explainability (Req 9)
# --------------------------------------------------------------------------- #

def test_top_features_empty_during_warmup():
    det = AnomalyDetector(_cfg())
    assert det.top_features(np.zeros(_DIM, dtype=np.float32)) == []


def test_top_features_ranks_by_zscore():
    names = [f"feat_{i}" for i in range(_DIM)]
    det = AnomalyDetector(_cfg(), feature_names=names)
    for fv in _normals(40, seed=0):
        det.score(fv)
    # A vector with one strongly deviating feature.
    probe = np.zeros(_DIM, dtype=np.float32)
    probe[3] = 20.0
    top = det.top_features(probe, k=3)
    assert top
    assert top[0].startswith("feat_3")
    assert "σ" in top[0]
