"""Behavioural regression tests for the detection core (plan-correcciones F1/F2).

They reproduce, with the real ``AudioProcessor`` and synthetic audio, the
failures found in the review:
  F1 — warmup windows entered the training buffer unscaled (PCA collapsed)
  F2 — a sustained anomaly was absorbed by the live-percentile threshold
"""

import numpy as np
import pytest
from scipy.signal import lfilter

from src.detection import AnomalyDetector, DetectorConfig
from src.dsp import AudioProcessor

SR = 16000
WIN = 4096


def _background(n_windows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = n_windows * WIN
    b = lfilter([0.05], [1.0, -0.95], rng.normal(0.0, 1.0, n))
    b = b / np.std(b) * 0.01
    t = np.arange(n) / SR
    wander = 1.0 + 0.2 * np.sin(2 * np.pi * t / 37.0)
    return ((b + 0.003 * np.sin(2 * np.pi * 60 * t)) * wander).astype(np.float32)


def _features(signal: np.ndarray) -> list[np.ndarray]:
    proc = AudioProcessor()
    return [
        proc.process_window(signal[k * WIN:(k + 1) * WIN])
        for k in range(len(signal) // WIN)
    ]


@pytest.fixture(scope="module")
def feats():
    """Feature vectors for: stationary background, a 1 kHz tone over the
    background, and a different steady noise (a 'new normal')."""
    bg = _background(800, seed=0)
    tone_bg = _background(60, seed=1)
    t = np.arange(len(tone_bg)) / SR
    tone = tone_bg + (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    rng = np.random.default_rng(2)
    new_normal = _background(160, seed=3) + rng.normal(
        0.0, 0.05, 160 * WIN
    ).astype(np.float32)
    return {
        "background": _features(bg),
        "tone": _features(tone),
        "new_normal": _features(new_normal),
    }


def _detector(**kw) -> AnomalyDetector:
    base = dict(
        buffer_size=300, refit_every=100, n_estimators=100,
        enable_drift_detection=False, random_state=0,
    )
    base.update(kw)
    return AnomalyDetector(DetectorConfig(**base))


def test_f1_training_buffer_is_homogeneous(feats):
    det = _detector()
    for fv in feats["background"][:300]:
        det.score(fv)
    assert det.get_status()["is_fitted"]

    # Before the fix PC1 explained 100 % of the variance (it only separated
    # the 30 unscaled warmup rows). Some dominance is genuine here: the
    # background's slow loudness wander moves all energy features together.
    assert det._pca.explained_variance_ratio_[0] < 0.9

    B = np.array(list(det._buffer), dtype=np.float64)
    # Raw vectors are stored as-is (no mix of scaled and unscaled rows).
    assert np.allclose(B[0], feats["background"][0])
    Z = (B - det._mean) / det._std
    norms = np.linalg.norm(Z, axis=1)
    # Before the fix the first 30 rows had norms ~1800x the rest.
    assert norms[:30].mean() < 2.0 * norms[30:].mean()


def test_f2_sustained_anomaly_stays_detected(feats):
    det = _detector()
    # 500 post-fit windows: the old live-percentile threshold needed a full
    # 500-score window before taking over (that is when it absorbed tones).
    for fv in feats["background"]:
        det.score(fv)
    flags = [det.score(fv).is_anomaly for fv in feats["tone"]]
    hyst = det._config.hysteresis_count
    coverage = np.mean(flags[hyst - 1:])
    # Before the fix: 9 of 60 windows (the threshold absorbed the tone).
    assert coverage >= 0.9, f"coverage={coverage:.2f}"


def test_f2_persistent_change_is_accepted_as_new_normal(feats):
    det = _detector(accept_after_windows=40)
    for fv in feats["background"][:400]:
        det.score(fv)
    refits_before = det.get_drift_metrics()["refit_count"]
    flags = [det.score(fv).is_anomaly for fv in feats["new_normal"]]
    # It is anomalous at first...
    assert np.mean(flags[:20]) > 0.5
    # ...but once accepted (forced refit) it stops alarming.
    assert det.get_drift_metrics()["refit_count"] > refits_before
    assert np.mean(flags[-40:]) <= 0.1


def test_f2_stationary_background_has_few_false_alarms(feats):
    det = _detector()
    flags = np.array(
        [det.score(fv).is_anomaly for fv in feats["background"]], dtype=int
    )
    bursts = int((np.diff(flags[300:]) == 1).sum() + flags[300])
    assert bursts <= 2, f"{bursts} false-alarm bursts in 500 windows"
