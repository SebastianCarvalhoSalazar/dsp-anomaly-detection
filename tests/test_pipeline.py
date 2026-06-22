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
