"""Turn a per-window anomaly decision into discrete events (plan F5).

The detector decides window by window (every 0.256 s). Saving one event per
anomalous window produced bursts of near-identical events for a single sound.
``EventSegmenter`` groups consecutive anomalous windows into **one** event
with a start, an end and a peak:

  IDLE ──(anomaly)──► ACTIVE ──(no anomaly for release_s)──► IDLE, emit event
                        │
                        └──(duration ≥ max_duration_s)──► IDLE, emit event

Gaps shorter than ``release_s`` do not split an event, so two bursts close
together are one event. Synchronous and pure (no I/O): persistence happens
elsewhere (see ``EventWriter``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class SegmentedEvent:
    start_ts: float          # wall-clock onset (capture time)
    end_ts: float            # wall-clock end of the last anomalous window
    peak_score: float        # highest anomaly score inside the event
    peak: Any                # caller payload captured at the peak window
    audio: np.ndarray = field(repr=False)  # pre-roll + event + release tail

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)


class EventSegmenter:
    """Group per-window anomaly flags into start/end events.

    Parameters
    ----------
    window_s : float
        Duration of one window (0.256 s live).
    release_s : float
        Quiet time (no anomalous window) that closes an event. Gaps shorter
        than this are merged into the same event.
    max_duration_s : float
        Hard cap on an event's length; a longer anomaly is split.
    onset_lookback_s : float
        The detector confirms an anomaly after ``hysteresis_count`` windows;
        the onset is placed this much earlier than the confirming window.
    """

    def __init__(
        self,
        window_s: float,
        release_s: float = 2.0,
        max_duration_s: float = 60.0,
        onset_lookback_s: float = 0.0,
    ) -> None:
        self.window_s = window_s
        self.release_s = release_s
        self.max_duration_s = max_duration_s
        self.onset_lookback_s = onset_lookback_s
        self._active = False
        self._start = 0.0
        self._last_anomaly = 0.0
        self._peak_score = float("-inf")
        self._peak: Any = None
        self._chunks: list[np.ndarray] = []

    @property
    def active(self) -> bool:
        return self._active

    def update(
        self,
        ts: float,
        is_anomaly: bool,
        score: float,
        window: np.ndarray,
        payload: Any = None,
        pre_roll: Optional[Callable[[], np.ndarray]] = None,
    ) -> Optional[SegmentedEvent]:
        """Feed one window; returns a finished event or ``None``.

        ``ts`` is the window's capture midpoint. ``pre_roll`` is called once,
        when an event starts, and must return the audio preceding (and
        including) the current window.
        """
        if not self._active:
            if not is_anomaly:
                return None
            self._active = True
            self._start = ts - self.onset_lookback_s
            self._last_anomaly = ts
            self._peak_score = score
            self._peak = payload
            first = pre_roll() if pre_roll is not None else window
            self._chunks = [np.asarray(first, dtype=np.float32)]
            return None

        self._chunks.append(np.asarray(window, dtype=np.float32))
        if is_anomaly:
            self._last_anomaly = ts
            if score > self._peak_score:
                self._peak_score = score
                self._peak = payload
        elif ts - self._last_anomaly >= self.release_s:
            return self._close()
        if ts - self._start >= self.max_duration_s:
            return self._close()
        return None

    def flush(self) -> Optional[SegmentedEvent]:
        """Close and return the active event, if any (e.g. on shutdown)."""
        return self._close() if self._active else None

    def _close(self) -> SegmentedEvent:
        event = SegmentedEvent(
            start_ts=self._start,
            end_ts=self._last_anomaly + self.window_s / 2.0,
            peak_score=self._peak_score,
            peak=self._peak,
            audio=np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32),
        )
        self._active = False
        self._chunks = []
        self._peak = None
        self._peak_score = float("-inf")
        return event
