"""Tests for the dashboard APIClient — all HTTP calls are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.dashboard.api_client import APIClient


FAKE_EVENTS = [
    {
        "id": 1,
        "timestamp": "2024-01-01T12:00:00+00:00",
        "anomaly_score": 0.85,
        "event_dir": "eventos/2024-01-01T12-00-00",
        "faiss_index_id": 0,
        "has_audio": True,
        "has_frame": True,
        "has_embedding": True,
    },
    {
        "id": 2,
        "timestamp": "2024-01-01T13:00:00+00:00",
        "anomaly_score": 0.60,
        "event_dir": "eventos/2024-01-01T13-00-00",
        "faiss_index_id": 1,
        "has_audio": True,
        "has_frame": False,
        "has_embedding": True,
    },
]


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# list_events
# ---------------------------------------------------------------------------

def test_list_events_calls_correct_url():
    with patch("src.dashboard.api_client.httpx.get") as mock_get:
        mock_get.return_value = _mock_response(FAKE_EVENTS)
        client = APIClient(base_url="http://testserver")
        result = client.list_events(limit=10, min_score=0.5)

    mock_get.assert_called_once()
    call_kwargs = mock_get.call_args
    assert "http://testserver/events/" in call_kwargs[0][0]
    assert call_kwargs[1]["params"]["limit"] == 10
    assert call_kwargs[1]["params"]["min_score"] == 0.5


def test_list_events_returns_parsed_list():
    with patch("src.dashboard.api_client.httpx.get") as mock_get:
        mock_get.return_value = _mock_response(FAKE_EVENTS)
        client = APIClient()
        result = client.list_events()

    assert len(result) == 2
    assert result[0]["id"] == 1
    assert result[1]["anomaly_score"] == 0.60


def test_list_events_empty():
    with patch("src.dashboard.api_client.httpx.get") as mock_get:
        mock_get.return_value = _mock_response([])
        client = APIClient()
        result = client.list_events()
    assert result == []


# ---------------------------------------------------------------------------
# get_event
# ---------------------------------------------------------------------------

def test_get_event_calls_correct_url():
    with patch("src.dashboard.api_client.httpx.get") as mock_get:
        mock_get.return_value = _mock_response(FAKE_EVENTS[0])
        client = APIClient(base_url="http://testserver")
        result = client.get_event(1)

    assert "http://testserver/events/1" in mock_get.call_args[0][0]
    assert result["id"] == 1


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def test_get_audio_url():
    client = APIClient(base_url="http://testserver")
    assert client.get_audio_url(5) == "http://testserver/events/5/audio"


def test_get_frame_url():
    client = APIClient(base_url="http://testserver")
    assert client.get_frame_url(5) == "http://testserver/events/5/frame"


# ---------------------------------------------------------------------------
# search_similar
# ---------------------------------------------------------------------------

def test_search_similar_sends_multipart():
    fake_results = [
        {"event": FAKE_EVENTS[0], "cosine_similarity": 0.95}
    ]
    with patch("src.dashboard.api_client.httpx.post") as mock_post:
        mock_post.return_value = _mock_response(fake_results)
        client = APIClient(base_url="http://testserver")
        result = client.search_similar(
            file_bytes=b"fake_audio_bytes",
            filename="test.wav",
            modality="audio",
            k=3,
        )

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "http://testserver/search/similar" in call_kwargs[0][0]
    assert call_kwargs[1]["params"]["modality"] == "audio"
    assert call_kwargs[1]["params"]["k"] == 3
    assert "file" in call_kwargs[1]["files"]
    assert result[0]["cosine_similarity"] == 0.95


def test_search_similar_returns_empty_on_no_results():
    with patch("src.dashboard.api_client.httpx.post") as mock_post:
        mock_post.return_value = _mock_response([])
        client = APIClient()
        result = client.search_similar(b"", "f.wav")
    assert result == []


# ---------------------------------------------------------------------------
# get_offline_analysis
# ---------------------------------------------------------------------------

def test_get_offline_analysis_calls_correct_url():
    fake_analysis = {
        "imfs": [[0.1, 0.2, 0.3]],
        "n_imfs": 1,
        "sample_rate": 16000,
        "spectrogram": [[0.0, 1.0]],
    }
    with patch("src.dashboard.api_client.httpx.get") as mock_get:
        mock_get.return_value = _mock_response(fake_analysis)
        client = APIClient(base_url="http://testserver")
        result = client.get_offline_analysis(3)

    assert "http://testserver/events/3/offline_analysis" in mock_get.call_args[0][0]
    assert result["n_imfs"] == 1
    assert result["sample_rate"] == 16000


# ---------------------------------------------------------------------------
# delete_event
# ---------------------------------------------------------------------------

def test_delete_event_calls_correct_url():
    with patch("src.dashboard.api_client.httpx.delete") as mock_del:
        mock_del.return_value = _mock_response(None, status_code=204)
        client = APIClient(base_url="http://testserver")
        client.delete_event(7)

    assert "http://testserver/events/7" in mock_del.call_args[0][0]


def test_delete_event_raises_on_error():
    import httpx as _httpx
    with patch("src.dashboard.api_client.httpx.delete") as mock_del:
        err_resp = MagicMock()
        err_resp.raise_for_status.side_effect = _httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )
        mock_del.return_value = err_resp
        client = APIClient(base_url="http://testserver")
        with pytest.raises(_httpx.HTTPStatusError):
            client.delete_event(999)


# ---------------------------------------------------------------------------
# clear_events
# ---------------------------------------------------------------------------

def test_clear_events_calls_correct_url():
    with patch("src.dashboard.api_client.httpx.delete") as mock_del:
        mock_del.return_value = _mock_response(None, status_code=204)
        client = APIClient(base_url="http://testserver")
        client.clear_events()

    assert "http://testserver/events/" in mock_del.call_args[0][0]


# ---------------------------------------------------------------------------
# reset_detector
# ---------------------------------------------------------------------------

def test_reset_detector_calls_correct_url():
    with patch("src.dashboard.api_client.httpx.post") as mock_post:
        mock_post.return_value = _mock_response({"ok": True})
        client = APIClient(base_url="http://testserver")
        client.reset_detector()

    assert "http://testserver/internal/reset-detector" in mock_post.call_args[0][0]
