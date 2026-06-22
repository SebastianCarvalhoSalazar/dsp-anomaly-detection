"""Temporal alignment between the audio and video streams.

The pipeline used to correlate each audio window with *the last frame
captured*, which is only loosely related in time. This module provides
explicit timestamps and a thread-safe ring buffer of recent frames so an
audio window can be matched to the frame **nearest in time** — the basis
for trustworthy cross-modal correlation, embeddings and stored events.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AudioWindow:
    """A processed audio window with explicit start/end timestamps (Unix)."""

    start_timestamp: float
    end_timestamp: float

    @property
    def timestamp(self) -> float:
        """Representative instant of the window (its midpoint)."""
        return (self.start_timestamp + self.end_timestamp) / 2.0


@dataclass
class CapturedFrame:
    """A camera frame tagged with the Unix timestamp at capture time."""

    timestamp: float
    frame: np.ndarray


class FrameRingBuffer:
    """Thread-safe circular buffer of recent :class:`CapturedFrame`.

    The camera thread ``push``es frames; the processing loop queries the
    frame ``nearest`` to an audio window's timestamp. All access is guarded
    by an internal lock, so no external locking is required.
    """

    def __init__(self, maxlen: int = 64) -> None:
        self._buf: deque[CapturedFrame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray, timestamp: float) -> None:
        """Append a frame. The caller is responsible for passing a frame it
        will not mutate (the pipeline pushes a ``.copy()``)."""
        with self._lock:
            self._buf.append(CapturedFrame(timestamp=timestamp, frame=frame))

    def nearest(
        self, timestamp: float, max_delta: Optional[float] = None
    ) -> Optional[CapturedFrame]:
        """Return the buffered frame closest in time to ``timestamp``.

        Returns ``None`` if the buffer is empty, or — when ``max_delta`` is
        given — if the closest frame is farther than ``max_delta`` seconds
        away (i.e. there is no temporally-aligned frame).
        """
        with self._lock:
            if not self._buf:
                return None
            best = min(self._buf, key=lambda cf: abs(cf.timestamp - timestamp))
        if max_delta is not None and abs(best.timestamp - timestamp) > max_delta:
            return None
        return best

    def latest(self) -> Optional[CapturedFrame]:
        """Return the most recently pushed frame, or None if empty."""
        with self._lock:
            return self._buf[-1] if self._buf else None

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)
