"""Tests for vision module using synthetic frames — no camera required."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.vision import BoundingBox, FrameCapture, MotionDetector, VisionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _black_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _frame_with_patch(h: int = 480, w: int = 640, brightness: int = 220) -> np.ndarray:
    """Frame with a bright 150×150 patch in the upper-left corner."""
    frame = _black_frame(h, w)
    frame[50:200, 50:200] = brightness
    return frame


def _warmup(detector: MotionDetector, n: int = 15) -> None:
    """Feed black frames to prime the MOG2 background model."""
    for _ in range(n):
        detector.detect(_black_frame())


# ---------------------------------------------------------------------------
# MotionDetector — detection
# ---------------------------------------------------------------------------

def test_motion_detector_detects_bright_patch():
    config = VisionConfig(min_contour_area=100)
    det = MotionDetector(config)
    _warmup(det)
    boxes = det.detect(_frame_with_patch(brightness=200))
    assert len(boxes) >= 1


def test_motion_detector_no_motion_on_constant_background():
    det = MotionDetector()
    # After warmup with black frames, another black frame should produce no motion
    _warmup(det, n=30)
    boxes = det.detect(_black_frame())
    assert boxes == []


def test_motion_detector_bboxes_meet_min_area():
    config = VisionConfig(min_contour_area=500)
    det = MotionDetector(config)
    _warmup(det)
    boxes = det.detect(_frame_with_patch(brightness=220))
    for box in boxes:
        assert box.area >= config.min_contour_area


def test_motion_detector_sorted_by_area_descending():
    config = VisionConfig(min_contour_area=100)
    det = MotionDetector(config)
    _warmup(det)
    # Two distinct bright patches
    frame = _black_frame()
    frame[50:200, 50:200] = 220    # 150x150 = large
    frame[300:350, 300:350] = 220  # 50x50 = smaller; may merge with MOG2 dilation
    boxes = det.detect(frame)
    if len(boxes) >= 2:
        assert boxes[0].area >= boxes[1].area


def test_motion_detector_returns_bounding_box_type():
    config = VisionConfig(min_contour_area=100)
    det = MotionDetector(config)
    _warmup(det)
    boxes = det.detect(_frame_with_patch())
    for box in boxes:
        assert isinstance(box, BoundingBox)
        assert box.w > 0 and box.h > 0


# ---------------------------------------------------------------------------
# MotionDetector — IoU and temporal weights
# ---------------------------------------------------------------------------

def test_iou_identical_boxes():
    a = BoundingBox(x=0, y=0, w=100, h=100, area=10000)
    b = BoundingBox(x=0, y=0, w=100, h=100, area=10000)
    assert abs(MotionDetector._iou(a, b) - 1.0) < 1e-6


def test_iou_no_overlap():
    a = BoundingBox(x=0, y=0, w=50, h=50, area=2500)
    b = BoundingBox(x=200, y=200, w=50, h=50, area=2500)
    assert MotionDetector._iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = BoundingBox(x=0, y=0, w=100, h=100, area=10000)
    b = BoundingBox(x=50, y=50, w=100, h=100, area=10000)
    iou = MotionDetector._iou(a, b)
    assert 0.0 < iou < 1.0


def test_temporal_weight_new_box():
    """First frame: all boxes are new → weight=1.0."""
    det = MotionDetector()
    boxes = [
        BoundingBox(
            x=10, y=10, w=50, h=50, area=2500,
        ),
    ]
    det.assign_temporal_weights(boxes)
    assert boxes[0].source_score == 1.0


def test_temporal_weight_persistent_box():
    """Same box twice → second time weight=0.5."""
    det = MotionDetector()
    # Simulate prev_boxes with same location
    det._prev_boxes = [
        BoundingBox(
            x=10, y=10, w=50, h=50, area=2500,
        ),
    ]
    boxes = [
        BoundingBox(
            x=10, y=10, w=50, h=50, area=2500,
        ),
    ]
    det.assign_temporal_weights(boxes)
    assert boxes[0].source_score == 0.5


def test_detect_assigns_source_score():
    """detect() should set source_score on returned boxes."""
    config = VisionConfig(min_contour_area=100)
    det = MotionDetector(config)
    _warmup(det)
    boxes = det.detect(_frame_with_patch())
    if boxes:
        # First detection → new boxes → weight 1.0
        assert boxes[0].source_score == 1.0


def test_source_score_default_zero():
    """BoundingBox created without source_score defaults to 0."""
    b = BoundingBox(x=0, y=0, w=10, h=10, area=100)
    assert b.source_score == 0.0


# ---------------------------------------------------------------------------
# MotionDetector — merge nearby boxes
# ---------------------------------------------------------------------------

def test_merge_overlapping_boxes():
    """Two overlapping boxes should merge into one."""
    a = BoundingBox(x=0, y=0, w=100, h=100, area=10000)
    b = BoundingBox(x=80, y=80, w=100, h=100, area=10000)
    merged = MotionDetector._merge_nearby_boxes([a, b], gap=0)
    assert len(merged) == 1
    m = merged[0]
    assert m.x == 0 and m.y == 0
    assert m.w == 180 and m.h == 180


def test_merge_nearby_within_gap():
    """Two boxes separated by less than gap should merge."""
    a = BoundingBox(x=0, y=0, w=50, h=50, area=2500)
    b = BoundingBox(x=70, y=0, w=50, h=50, area=2500)
    # gap between edges = 20px, gap threshold = 30
    merged = MotionDetector._merge_nearby_boxes(
        [a, b], gap=30,
    )
    assert len(merged) == 1
    m = merged[0]
    assert m.x == 0 and m.w == 120


def test_merge_far_apart_boxes_untouched():
    """Two distant boxes should NOT merge."""
    a = BoundingBox(x=0, y=0, w=50, h=50, area=2500)
    b = BoundingBox(x=300, y=300, w=50, h=50, area=2500)
    merged = MotionDetector._merge_nearby_boxes(
        [a, b], gap=30,
    )
    assert len(merged) == 2


def test_merge_three_into_one():
    """Three boxes forming a chain should all merge."""
    a = BoundingBox(x=0, y=0, w=50, h=50, area=2500)
    b = BoundingBox(x=60, y=0, w=50, h=50, area=2500)
    c = BoundingBox(x=120, y=0, w=50, h=50, area=2500)
    merged = MotionDetector._merge_nearby_boxes(
        [a, b, c], gap=15,
    )
    assert len(merged) == 1
    m = merged[0]
    assert m.x == 0 and m.w == 170


def test_merge_empty_list():
    merged = MotionDetector._merge_nearby_boxes([], gap=30)
    assert merged == []


def test_merge_single_box():
    b = BoundingBox(x=10, y=10, w=50, h=50, area=2500)
    merged = MotionDetector._merge_nearby_boxes(
        [b], gap=30,
    )
    assert len(merged) == 1
    assert merged[0].x == 10


def test_edge_distance_overlapping():
    a = BoundingBox(x=0, y=0, w=100, h=100, area=10000)
    b = BoundingBox(x=50, y=50, w=100, h=100, area=10000)
    assert MotionDetector._edge_distance(a, b) == 0.0


def test_edge_distance_separated():
    a = BoundingBox(x=0, y=0, w=50, h=50, area=2500)
    b = BoundingBox(x=70, y=0, w=50, h=50, area=2500)
    # gap in x = 20, gap in y = 0
    assert MotionDetector._edge_distance(a, b) == 20.0


def test_merge_disabled_with_negative_gap():
    """gap < 0 should disable merging entirely."""
    a = BoundingBox(x=0, y=0, w=100, h=100, area=10000)
    b = BoundingBox(x=50, y=50, w=100, h=100, area=10000)
    merged = MotionDetector._merge_nearby_boxes(
        [a, b], gap=-1,
    )
    assert len(merged) == 2


def test_merge_blocked_by_max_area():
    """Overlapping boxes should NOT merge if result exceeds max_area."""
    a = BoundingBox(x=0, y=0, w=200, h=200, area=40000)
    b = BoundingBox(x=100, y=100, w=200, h=200, area=40000)
    # Union would be 300×300 = 90000 > max_area=50000
    merged = MotionDetector._merge_nearby_boxes(
        [a, b], gap=0, max_area=50000,
    )
    assert len(merged) == 2


def test_merge_allowed_under_max_area():
    """Small overlapping boxes should merge when under max_area."""
    a = BoundingBox(x=0, y=0, w=50, h=50, area=2500)
    b = BoundingBox(x=40, y=0, w=50, h=50, area=2500)
    # Union = 90×50 = 4500 < max_area=10000
    merged = MotionDetector._merge_nearby_boxes(
        [a, b], gap=0, max_area=10000,
    )
    assert len(merged) == 1


def test_merge_chain_blocked_by_max_area():
    """Chain of 3 overlapping boxes: first pair merges,
    but merging with the third exceeds max_area."""
    a = BoundingBox(x=0, y=0, w=80, h=80, area=6400)
    b = BoundingBox(x=70, y=0, w=80, h=80, area=6400)
    c = BoundingBox(x=140, y=0, w=80, h=80, area=6400)
    # a+b union = 150×80 = 12000 (ok if max_area=15000)
    # (a+b)+c union = 220×80 = 17600 (exceeds 15000)
    merged = MotionDetector._merge_nearby_boxes(
        [a, b, c], gap=0, max_area=15000,
    )
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# MotionDetector — draw_boxes
# ---------------------------------------------------------------------------

def test_draw_boxes_returns_copy():
    det = MotionDetector()
    frame = _black_frame()
    boxes = [BoundingBox(x=10, y=10, w=50, h=50, area=2500)]
    annotated = det.draw_boxes(frame, boxes)
    assert annotated is not frame


def test_draw_boxes_same_shape():
    det = MotionDetector()
    frame = _black_frame()
    boxes = [BoundingBox(x=10, y=10, w=50, h=50, area=2500)]
    annotated = det.draw_boxes(frame, boxes)
    assert annotated.shape == frame.shape


def test_draw_boxes_modifies_pixels():
    det = MotionDetector()
    frame = _black_frame()
    boxes = [BoundingBox(x=10, y=10, w=100, h=100, area=10000)]
    annotated = det.draw_boxes(frame, boxes)
    # The green rectangle line must alter at least one pixel
    assert not np.array_equal(annotated, frame)


def test_draw_boxes_no_boxes_unchanged():
    det = MotionDetector()
    frame = _black_frame()
    annotated = det.draw_boxes(frame, [])
    np.testing.assert_array_equal(annotated, frame)


# ---------------------------------------------------------------------------
# MotionDetector — reset
# ---------------------------------------------------------------------------

def test_reset_reinitializes_background():
    config = VisionConfig(min_contour_area=100)
    det = MotionDetector(config)
    _warmup(det, n=20)
    det._prev_boxes = [
        BoundingBox(
            x=0, y=0, w=10, h=10, area=100,
        ),
    ]
    det.reset()
    assert det._prev_boxes == []
    # Verify detector remains usable
    boxes = det.detect(_black_frame())
    assert isinstance(boxes, list)


# ---------------------------------------------------------------------------
# MotionDetector — detect_with_result
# ---------------------------------------------------------------------------

def test_detect_with_result_returns_motion_result():
    from src.vision.types import MotionResult
    config = VisionConfig(min_contour_area=100)
    det = MotionDetector(config)
    _warmup(det)
    result = det.detect_with_result(_frame_with_patch())
    assert isinstance(result, MotionResult)
    assert result.annotated_frame.shape == result.frame.shape
    assert isinstance(result.has_motion, bool)
    assert result.timestamp > 0


# ---------------------------------------------------------------------------
# FrameCapture — mocked cv2.VideoCapture
# ---------------------------------------------------------------------------

def test_frame_capture_raises_if_device_unavailable():
    """open() must raise RuntimeError when the device cannot be opened."""
    with patch("src.vision.capture.cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        cap = FrameCapture()
        with pytest.raises(RuntimeError, match="Could not open"):
            cap.open()


def test_frame_capture_read_returns_frame():
    with patch("src.vision.capture.cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)
        mock_cap_cls.return_value = mock_cap

        cap = FrameCapture()
        cap.open()
        frame = cap.read()
        assert frame is not None
        assert frame.shape == (480, 640, 3)


def test_frame_capture_read_returns_none_when_closed():
    cap = FrameCapture()
    assert cap.read() is None


def test_frame_capture_context_manager():
    with patch("src.vision.capture.cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cap_cls.return_value = mock_cap

        with FrameCapture() as cap:
            assert cap.read() is None
        mock_cap.release.assert_called_once()
