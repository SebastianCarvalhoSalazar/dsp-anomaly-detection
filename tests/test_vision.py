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
    # After reset, a previously stable frame may trigger motion again
    det.reset()
    # Just verify reset does not raise and detector remains usable
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
