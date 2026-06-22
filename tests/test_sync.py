"""Tests for audio-video temporal alignment (Fase 1)."""

import numpy as np

from src.sync import AudioWindow, CapturedFrame, FrameRingBuffer


def _frame(val: int) -> np.ndarray:
    return np.full((4, 4, 3), val, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# AudioWindow
# --------------------------------------------------------------------------- #

def test_audio_window_timestamp_is_midpoint():
    w = AudioWindow(start_timestamp=10.0, end_timestamp=12.0)
    assert w.timestamp == 11.0


# --------------------------------------------------------------------------- #
# FrameRingBuffer — nearest
# --------------------------------------------------------------------------- #

def test_nearest_picks_closest_in_time():
    buf = FrameRingBuffer(maxlen=10)
    buf.push(_frame(1), timestamp=100.0)
    buf.push(_frame(2), timestamp=101.0)
    buf.push(_frame(3), timestamp=102.0)

    near = buf.nearest(101.4)
    assert near is not None
    assert int(near.frame[0, 0, 0]) == 2  # closest to 101.0


def test_nearest_after_all_returns_last():
    buf = FrameRingBuffer(maxlen=10)
    buf.push(_frame(1), timestamp=100.0)
    buf.push(_frame(2), timestamp=101.0)
    near = buf.nearest(200.0)
    assert int(near.frame[0, 0, 0]) == 2


def test_nearest_before_all_returns_first():
    buf = FrameRingBuffer(maxlen=10)
    buf.push(_frame(7), timestamp=100.0)
    buf.push(_frame(8), timestamp=101.0)
    near = buf.nearest(0.0)
    assert int(near.frame[0, 0, 0]) == 7


def test_nearest_empty_returns_none():
    buf = FrameRingBuffer()
    assert buf.nearest(123.0) is None


def test_nearest_respects_max_delta():
    buf = FrameRingBuffer()
    buf.push(_frame(1), timestamp=100.0)
    # Closest frame is 5s away; with a 1s tolerance there is no aligned frame.
    assert buf.nearest(105.0, max_delta=1.0) is None
    assert buf.nearest(100.5, max_delta=1.0) is not None


# --------------------------------------------------------------------------- #
# FrameRingBuffer — capacity / latest / clear
# --------------------------------------------------------------------------- #

def test_ring_buffer_evicts_oldest():
    buf = FrameRingBuffer(maxlen=2)
    buf.push(_frame(1), timestamp=1.0)
    buf.push(_frame(2), timestamp=2.0)
    buf.push(_frame(3), timestamp=3.0)
    assert len(buf) == 2
    # The oldest (ts=1.0) was evicted.
    assert buf.nearest(1.0).timestamp == 2.0


def test_latest_and_clear():
    buf = FrameRingBuffer()
    assert buf.latest() is None
    buf.push(_frame(9), timestamp=5.0)
    assert isinstance(buf.latest(), CapturedFrame)
    assert buf.latest().timestamp == 5.0
    buf.clear()
    assert len(buf) == 0
    assert buf.latest() is None
