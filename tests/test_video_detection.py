"""Tests for the independent video anomaly detector (Fase 2)."""

import numpy as np

from src.vision.types import BoundingBox
from src.vision_detection import (
    VideoAnomalyDetector,
    VideoFeatureExtractor,
    VideoFeatureVector,
    default_video_config,
)


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #

def test_extract_empty_scene_is_zeros():
    fx = VideoFeatureExtractor()
    fv = fx.extract([], (480, 640, 3))
    assert fv.to_array().tolist() == [0.0] * 7


def test_extract_computes_ratios_and_weights():
    fx = VideoFeatureExtractor()
    # Two boxes; source_score holds the IoU temporal weight at this stage.
    boxes = [
        BoundingBox(x=0, y=0, w=64, h=48, area=64 * 48, source_score=1.0),
        BoundingBox(x=100, y=100, w=32, h=24, area=32 * 24, source_score=0.5),
    ]
    fv = fx.extract(boxes, (480, 640, 3))
    frame_area = 480 * 640
    assert fv.bbox_count == 2.0
    assert fv.largest_bbox_area_ratio == (64 * 48) / frame_area
    assert fv.max_temporal_weight == 1.0
    assert fv.average_temporal_weight == 0.75
    assert 0.0 <= fv.motion_energy <= 1.0


def test_extract_clamps_overlapping_area_to_one():
    fx = VideoFeatureExtractor()
    # Two boxes that each cover the whole tiny frame → sum ratio > 1 → clamp.
    big = BoundingBox(x=0, y=0, w=10, h=10, area=100, source_score=1.0)
    fv = fx.extract([big, big], (10, 10, 3))
    assert fv.total_foreground_area_ratio == 1.0


# --------------------------------------------------------------------------- #
# Detector — warmup / fit / score (no PCA)
# --------------------------------------------------------------------------- #

def _motion_vectors(n, seed=0, scale=0.05):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        base = np.abs(rng.normal(0.1, scale, size=7)).astype(np.float32)
        out.append(np.clip(base, 0.0, 1.0))
    return out


def test_video_detector_has_no_pca():
    assert default_video_config().enable_pca is False
    det = VideoAnomalyDetector()
    for fv in _motion_vectors(det._config.buffer_size + 5):
        r = det.score(fv)
    assert det.get_status()["is_fitted"]
    assert det._pca is None


def test_video_detector_scores_in_range():
    det = VideoAnomalyDetector(default_video_config())
    r = None
    for fv in _motion_vectors(det._config.buffer_size + 10):
        r = det.score(fv)
    assert r.is_fitted
    assert 0.0 <= r.anomaly_score <= 1.0


def test_video_detector_warmup_not_fitted():
    det = VideoAnomalyDetector()
    r = det.score(VideoFeatureVector.zeros().to_array())
    assert not r.is_fitted
    assert r.anomaly_score == 0.0
