"""Tests for Pipeline._check_reset() without requiring microphone or camera."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline import Pipeline


@pytest.fixture
def pipeline(tmp_path):
    """Create a Pipeline instance with all stores pointed at tmp_path."""
    import os
    os.environ["EVENTS_DIR"] = str(tmp_path / "eventos")
    os.environ["DB_PATH"]    = str(tmp_path / "events.db")
    os.environ["FAISS_PATH"] = str(tmp_path / "faiss.index")
    p = Pipeline()
    p.db.init()
    p.faiss_store.init()
    return p


# ---------------------------------------------------------------------------
# _check_reset
# ---------------------------------------------------------------------------

def test_check_reset_resets_detector_when_pending(pipeline):
    """If API returns pending=True, detector and buffer are reset."""
    # First fill the buffer a bit
    pipeline._pre_audio_buffer.extend([0.1] * 100)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"pending": True}

    with patch("src.pipeline.httpx.get", return_value=mock_resp):
        pipeline._check_reset()

    assert len(pipeline._pre_audio_buffer) == 0
    # After reset, detector should report not fitted
    assert pipeline.detector.get_status()["is_fitted"] is False


def test_check_reset_does_nothing_when_not_pending(pipeline):
    """If API returns pending=False, nothing changes."""
    import numpy as np
    pipeline._pre_audio_buffer.extend([0.5] * 50)
    original_len = len(pipeline._pre_audio_buffer)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"pending": False}

    with patch("src.pipeline.httpx.get", return_value=mock_resp):
        pipeline._check_reset()

    assert len(pipeline._pre_audio_buffer) == original_len


def test_check_reset_is_silent_when_api_unavailable(pipeline):
    """Network errors must not propagate — best-effort only."""
    import httpx as _httpx
    with patch("src.pipeline.httpx.get", side_effect=_httpx.ConnectError("refused")):
        pipeline._check_reset()  # must not raise


def test_check_reset_logs_debug_on_failure(pipeline, caplog):
    """A connection failure should emit a DEBUG log, not crash."""
    import logging
    import httpx as _httpx
    with caplog.at_level(logging.DEBUG, logger="src.pipeline"):
        with patch("src.pipeline.httpx.get", side_effect=_httpx.ConnectError("refused")):
            pipeline._check_reset()
    assert any("Reset poll failed" in r.message for r in caplog.records)


def test_check_reset_handles_bad_json(pipeline):
    """Malformed API response must not propagate."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("bad json")

    with patch("src.pipeline.httpx.get", return_value=mock_resp):
        pipeline._check_reset()  # must not raise


def test_check_reset_handles_non_200_status(pipeline):
    """Non-200 HTTP responses must not trigger a reset."""
    pipeline._pre_audio_buffer.extend([0.1] * 10)

    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.json.return_value = {}

    with patch("src.pipeline.httpx.get", return_value=mock_resp):
        pipeline._check_reset()

    # Buffer untouched
    assert len(pipeline._pre_audio_buffer) == 10


# ---------------------------------------------------------------------------
# Live fusion config (dashboard-driven)
# ---------------------------------------------------------------------------

def test_apply_fusion_config_rebuilds_strategy_and_gates(pipeline):
    from src.fusion import Maximum
    pipeline._apply_fusion_config(
        {"strategy": "max", "audio_weight": 0.2, "gates": True}
    )
    assert isinstance(pipeline.fusion_strategy, Maximum)
    assert pipeline._fusion_gates is True


def test_apply_fusion_config_weighted_uses_weight(pipeline):
    from src.fusion import WeightedAverage
    pipeline._apply_fusion_config(
        {"strategy": "weighted", "audio_weight": 0.8, "gates": False}
    )
    assert isinstance(pipeline.fusion_strategy, WeightedAverage)
    assert pipeline.fusion_strategy.audio_weight == 0.8
    assert pipeline._fusion_gates is False


def test_apply_fusion_config_ignores_unknown_strategy(pipeline):
    before = pipeline.fusion_strategy
    pipeline._apply_fusion_config({"strategy": "bogus", "audio_weight": 0.5})
    assert pipeline.fusion_strategy is before  # unchanged


def test_check_fusion_config_applies_remote_config(pipeline):
    from src.fusion import OrStrategy
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "strategy": "or", "audio_weight": 0.5, "gates": True,
    }
    with patch("src.pipeline.httpx.get", return_value=mock_resp):
        pipeline._check_fusion_config()
    assert isinstance(pipeline.fusion_strategy, OrStrategy)
    assert pipeline._fusion_gates is True


def test_check_fusion_config_silent_when_api_unavailable(pipeline):
    import httpx as _httpx
    with patch("src.pipeline.httpx.get", side_effect=_httpx.ConnectError("x")):
        pipeline._check_fusion_config()  # must not raise


# ---------------------------------------------------------------------------
# process_window — regressions (plan-correcciones)
# ---------------------------------------------------------------------------

def _noise_windows(n, seed=0, scale=0.01):
    import numpy as np
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, scale, 4096).astype(np.float32) for _ in range(n)]


@pytest.fixture
def offline_pipeline(tmp_path, monkeypatch):
    """Pipeline driven through process_window(): no mic, camera, API or models.

    Runs from tmp_path so relative detector-state paths (data/...) never touch
    the repo.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVENTS_DIR", str(tmp_path / "eventos"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "events.db"))
    monkeypatch.setenv("FAISS_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setenv("SNAPSHOTS_DIR", str(tmp_path / "snapshots"))
    p = Pipeline()
    p.encoder = MagicMock()
    p.encoder.encode.return_value = __import__("numpy").ones(1536, "float32")
    p._notify_api = MagicMock()
    p.db.init()
    p.faiss_store.init()
    return p


def test_f3_no_fused_events_during_warmup(offline_pipeline):
    """F3: with the fused decision gating events, the warmup must not fire.

    Before the fix both calibrators filled with warmup zeros, ties mapped to
    rank 1.0 and combined_score=1.0 fired an event on every window."""
    p = offline_pipeline
    p._fusion_gates = True
    p._handle_anomaly = MagicMock()
    for k, w in enumerate(_noise_windows(120)):
        out = p.process_window(w, 1000.0 + k * 0.256)
        assert out["mm"]["combined_score"] == 0.0
    assert not p.detector.get_status()["is_fitted"]
    p._handle_anomaly.assert_not_called()


def test_f4_capture_midpoint_uses_adc_time():
    from types import SimpleNamespace
    from src.pipeline import capture_midpoint
    ti = SimpleNamespace(currentTime=50.30, inputBufferAdcTime=50.00)
    # Block started 0.30 s before the callback → midpoint = now - 0.30 + 0.128
    assert capture_midpoint(1000.0, ti, 0.256) == pytest.approx(999.828)
    # Host APIs that report zeros: assume the block just ended.
    zeros = SimpleNamespace(currentTime=0.0, inputBufferAdcTime=0.0)
    assert capture_midpoint(1000.0, zeros, 0.256) == pytest.approx(999.872)


def test_f4_loop_uses_capture_time_not_dequeue_time(offline_pipeline):
    """F4: a window that waited in the queue keeps its capture timestamp."""
    import threading
    import time as _time
    import numpy as np
    p = offline_pipeline
    seen = []

    def _record(window, ts):
        seen.append(ts)
        p._running = False
        return {}

    p.process_window = _record
    p._audio_queue.put((np.zeros(4096, np.float32), 123.456))
    p._running = True
    t = threading.Thread(target=p._process_loop)
    t.start()
    t.join(timeout=5)
    assert seen == [123.456]
    assert abs(seen[0] - _time.time()) > 1000  # not the dequeue time


def test_f4_frame_matched_by_capture_time_or_none(offline_pipeline):
    """F4: the frame nearest to the capture time is used; if none is close
    enough the window has no frame (instead of a wrong one)."""
    import numpy as np
    p = offline_pipeline
    used = []
    p.vision.detect = lambda frame: used.append(int(frame[0, 0, 0])) or []
    # Frames every 40 ms over 2 s, each tagged with its index.
    for i in range(50):
        f = np.full((4, 4, 3), i % 256, np.uint8)
        p._frame_buffer.push(f, 1000.0 + i * 0.04)
    w = np.zeros(4096, np.float32)
    p.process_window(w, 1000.0 + 0.5)       # → frame 12 (t=1000.48/1000.52)
    assert used and abs(used[-1] - 12.5) <= 0.5
    n = len(used)
    p.process_window(w, 1000.0 + 5.0)       # 3 s after the last frame
    assert len(used) == n                   # no frame → detect not called


def _room(n_windows, seed=0):
    import numpy as np
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    b = lfilter([0.05], [1.0, -0.95], rng.normal(0, 1, n_windows * 4096))
    return (b / b.std() * 0.01).astype(np.float32)


def test_f5_sustained_tone_is_one_event_and_loop_never_blocks(offline_pipeline):
    """F5: a 10 s tone produces exactly ONE stored event (before: one per
    anomalous window), and a slow encoder (2 s) never stalls the audio loop."""
    import time as _time
    import numpy as np
    from src.detection import AnomalyDetector, DetectorConfig

    p = offline_pipeline
    p.detector = AnomalyDetector(
        DetectorConfig(buffer_size=200, refit_every=100,
                       enable_drift_detection=False),
        feature_names=p.dsp.feature_names,
    )
    p.encoder.encode.side_effect = (
        lambda *a, **k: (_time.sleep(2.0), np.ones(1536, "float32"))[1]
    )
    sig = _room(320, seed=4)
    t = np.arange(40 * 4096) / 16000
    sig[260 * 4096:300 * 4096] += (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(
        np.float32
    )
    times = []
    for k in range(320):
        a = _time.perf_counter()
        p.process_window(sig[k * 4096:(k + 1) * 4096], 5000.0 + (k + 0.5) * 0.256)
        times.append(_time.perf_counter() - a)
    p.flush_events()

    rows = p.db.list_events(limit=100)
    assert len(rows) == 1
    assert 8.0 <= rows[0].duration_s <= 12.0      # tone lasts 10.24 s
    assert rows[0].end_timestamp is not None
    # The 2 s encode ran on the writer thread, never inside process_window.
    assert max(times[200:]) < 0.2, f"max window time {max(times[200:]):.3f}s"


def test_f6_non_deciding_components_are_off_by_default(offline_pipeline):
    """Fase 6 / ADR-0014: drift, snapshots and the video detector are opt-in."""
    import numpy as np
    p = offline_pipeline
    assert p.detector._config.enable_drift_detection is False
    assert p.detector._config.enable_drift_aware_refit is False
    assert p.detector._snapshot_store is None
    p.video_detector.score = MagicMock()
    p._frame_buffer.push(np.zeros((8, 8, 3), np.uint8), 7000.0)
    out = p.process_window(np.zeros(4096, np.float32), 7000.0)
    p.video_detector.score.assert_not_called()
    assert out["mm"]["video_score"] == 0.0


def test_f6_flags_reenable_components(tmp_path, monkeypatch):
    for k in ("ENABLE_DRIFT", "ENABLE_SNAPSHOTS", "ENABLE_VIDEO_DETECTOR"):
        monkeypatch.setenv(k, "true")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SNAPSHOTS_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "events.db"))
    monkeypatch.setenv("FAISS_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setenv("EVENTS_DIR", str(tmp_path / "eventos"))
    p = Pipeline()
    assert p.detector._config.enable_drift_detection is True
    assert p.detector._snapshot_store is not None
    assert p._enable_video is True
