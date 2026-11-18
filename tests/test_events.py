"""Tests for event segmentation and the background event writer (plan F5)."""

import threading
import time

import numpy as np
import pytest

from src.events import EventSegmenter, EventWriter

W = 0.256


def _run(flags, seg=None, scores=None):
    seg = seg or EventSegmenter(window_s=W, release_s=2.0, max_duration_s=60.0)
    events = []
    for k, f in enumerate(flags):
        s = scores[k] if scores is not None else (0.9 if f else 0.1)
        ev = seg.update(k * W, bool(f), s, np.full(4, k, np.float32), payload=k)
        if ev is not None:
            events.append(ev)
    tail = seg.flush()
    if tail is not None:
        events.append(tail)
    return events


def test_one_burst_is_one_event():
    flags = [0] * 10 + [1] * 20 + [0] * 20
    events = _run(flags)
    assert len(events) == 1
    ev = events[0]
    assert ev.start_ts == pytest.approx(10 * W)
    assert ev.end_ts == pytest.approx(29 * W + W / 2)


def test_bursts_closer_than_release_merge():
    # 1 s gap (4 windows) < release_s=2 s → one event
    flags = [0] * 5 + [1] * 6 + [0] * 4 + [1] * 6 + [0] * 20
    assert len(_run(flags)) == 1


def test_bursts_farther_than_release_split():
    flags = [0] * 5 + [1] * 6 + [0] * 12 + [1] * 6 + [0] * 20
    assert len(_run(flags)) == 2


def test_long_anomaly_is_capped():
    seg = EventSegmenter(window_s=W, release_s=2.0, max_duration_s=10.0)
    flags = [1] * 100  # 25.6 s
    events = _run(flags, seg=seg)
    assert len(events) == 3
    # the window that reaches the cap is included (≤ 2 windows of overshoot)
    assert all(e.duration_s <= 10.0 + 2 * W for e in events)


def test_peak_payload_and_onset_lookback():
    seg = EventSegmenter(window_s=W, release_s=2.0, onset_lookback_s=2 * W)
    flags = [0] * 4 + [1] * 5 + [0] * 12
    scores = [0.1] * 4 + [0.5, 0.7, 0.95, 0.6, 0.55] + [0.1] * 12
    (ev,) = _run(flags, seg=seg, scores=scores)
    assert ev.peak == 6 and ev.peak_score == 0.95
    assert ev.start_ts == pytest.approx(4 * W - 2 * W)


def test_pre_roll_is_used_once_at_start():
    seg = EventSegmenter(window_s=W, release_s=0.5)
    pre = np.arange(10, dtype=np.float32)
    calls = []

    def pre_roll():
        calls.append(1)
        return pre

    events = []
    for k, f in enumerate([1, 1, 0, 0, 0]):
        ev = seg.update(k * W, bool(f), 0.9, np.full(3, 100 + k, np.float32),
                        pre_roll=pre_roll)
        if ev is not None:
            events.append(ev)
    assert len(calls) == 1
    (ev,) = events
    assert np.array_equal(ev.audio[:10], pre)
    # pre-roll (includes the first window) + windows 1..3 (closes at k=3)
    assert len(ev.audio) == 10 + 3 * 3


def test_writer_runs_jobs_off_thread_in_order():
    w = EventWriter()
    out, threads = [], []
    for i in range(5):
        w.submit(lambda i=i: (out.append(i), threads.append(threading.get_ident())))
    w.flush()
    assert out == list(range(5))
    assert threading.get_ident() not in threads


def test_writer_submit_never_blocks_and_counts_drops():
    w = EventWriter(maxsize=2)
    gate = threading.Event()
    w.submit(gate.wait)          # occupies the worker
    time.sleep(0.05)
    t0 = time.perf_counter()
    results = [w.submit(lambda: None) for _ in range(5)]
    assert time.perf_counter() - t0 < 0.05
    assert results.count(False) == 3 and w.dropped == 3
    gate.set()
    w.flush()


def test_writer_survives_failing_job():
    w = EventWriter()
    out = []
    w.submit(lambda: 1 / 0)
    w.submit(lambda: out.append("ok"))
    w.flush()
    assert out == ["ok"] and w.failed == 1
