"""Tests for the FastAPI application using TestClient and injected mock stores."""
from __future__ import annotations

import json
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.storage.config import StorageConfig
from src.storage.db import Database
from src.storage.event_store import EventStore
from src.storage.faiss_store import FAISSStore
from src.storage.models import AnomalyEvent


# ---------------------------------------------------------------------------
# Fixtures — minimal real stores backed by tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_stores(tmp_path):
    config = StorageConfig(
        events_dir=str(tmp_path / "eventos"),
        db_path=str(tmp_path / "events.db"),
        faiss_path=str(tmp_path / "faiss.index"),
    )
    db = Database(config)
    db.init()
    faiss_store = FAISSStore(config)
    faiss_store.init()
    event_store = EventStore(config)
    return db, faiss_store, event_store


@pytest.fixture
def client(tmp_stores):
    db, faiss_store, event_store = tmp_stores
    app = create_app(db=db, faiss_store=faiss_store, event_store=event_store)
    with TestClient(app) as c:
        yield c


_event_counter = 0


def _insert_event(db: Database, event_store: EventStore, tmp_path: Path, score: float = 0.7) -> int:
    """Helper: persist a dummy event to DB and filesystem."""
    global _event_counter
    _event_counter += 1
    from datetime import timedelta
    ts = datetime(2024, 6, 1, 12, 0, _event_counter, tzinfo=timezone.utc)
    audio = np.zeros(1024, dtype=np.float32)
    event_dir = event_store.save_event(ts, audio, 16000, frame=None, anomaly_score=score)
    orm_event = AnomalyEvent(
        timestamp=ts,
        anomaly_score=score,
        event_dir=str(event_dir),
    )
    return db.save_event(orm_event)


# ---------------------------------------------------------------------------
# GET /events/
# ---------------------------------------------------------------------------

def test_list_events_empty(client):
    resp = client.get("/events/")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_events_returns_saved(client, tmp_stores, tmp_path):
    db, faiss_store, event_store = tmp_stores
    # Re-create client with the same stores to share state
    app = create_app(db=db, faiss_store=faiss_store, event_store=event_store)
    with TestClient(app) as c:
        _insert_event(db, event_store, tmp_path)
        resp = c.get("/events/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert "id" in data[0]
        assert "anomaly_score" in data[0]


def test_list_events_min_score_filter(client, tmp_stores, tmp_path):
    db, faiss_store, event_store = tmp_stores
    app = create_app(db=db, faiss_store=faiss_store, event_store=event_store)
    with TestClient(app) as c:
        _insert_event(db, event_store, tmp_path, score=0.3)
        _insert_event(db, event_store, tmp_path, score=0.9)
        resp = c.get("/events/?min_score=0.5")
        data = resp.json()
        assert all(e["anomaly_score"] >= 0.5 for e in data)


# ---------------------------------------------------------------------------
# GET /events/{id}
# ---------------------------------------------------------------------------

def test_get_event_not_found(client):
    resp = client.get("/events/9999")
    assert resp.status_code == 404


def test_get_event_found(client, tmp_stores, tmp_path):
    db, faiss_store, event_store = tmp_stores
    app = create_app(db=db, faiss_store=faiss_store, event_store=event_store)
    with TestClient(app) as c:
        event_id = _insert_event(db, event_store, tmp_path)
        resp = c.get(f"/events/{event_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == event_id


# ---------------------------------------------------------------------------
# GET /events/{id}/audio
# ---------------------------------------------------------------------------

def test_get_audio_returns_wav(client, tmp_stores, tmp_path):
    db, faiss_store, event_store = tmp_stores
    app = create_app(db=db, faiss_store=faiss_store, event_store=event_store)
    with TestClient(app) as c:
        event_id = _insert_event(db, event_store, tmp_path)
        resp = c.get(f"/events/{event_id}/audio")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"


def test_get_audio_not_found(client):
    resp = client.get("/events/9999/audio")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /events/{id}/frame — no frame case
# ---------------------------------------------------------------------------

def test_get_frame_not_found_when_no_frame(client, tmp_stores, tmp_path):
    db, faiss_store, event_store = tmp_stores
    app = create_app(db=db, faiss_store=faiss_store, event_store=event_store)
    with TestClient(app) as c:
        event_id = _insert_event(db, event_store, tmp_path)
        resp = c.get(f"/events/{event_id}/frame")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# WebSocket /ws/stream
# ---------------------------------------------------------------------------

def test_websocket_connects(client):
    with client.websocket_connect("/ws/stream") as ws:
        pass  # connect and cleanly disconnect


def test_websocket_broadcast(client, tmp_stores, tmp_path):
    """broadcast() should push a message to all connected WS clients."""
    from src.api.routers.websocket import broadcast
    import asyncio

    db, faiss_store, event_store = tmp_stores
    app = create_app(db=db, faiss_store=faiss_store, event_store=event_store)
    received = []

    with TestClient(app) as c:
        with c.websocket_connect("/ws/stream") as ws:
            # Push via /internal/score endpoint
            payload = {
                "anomaly_score": 0.9,
                "is_anomaly": True,
                "is_fitted": True,
                "timestamp": "2024-01-01T00:00:00",
                "window_index": 1,
                "bounding_boxes": [],
            }
            resp = c.post("/internal/score", json=payload)
            assert resp.status_code == 200
            msg = ws.receive_text()
            data = json.loads(msg)
            assert abs(data["anomaly_score"] - 0.9) < 1e-6
            assert data["is_anomaly"] is True


# ---------------------------------------------------------------------------
# POST /search/similar — empty index returns empty list
# ---------------------------------------------------------------------------

def test_search_similar_empty_index(client, tmp_path):
    audio = np.zeros(16000, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, 16000)
        with open(f.name, "rb") as wav_file:
            resp = client.post(
                "/search/similar?modality=audio&k=5",
                files={"file": ("audio.wav", wav_file, "audio/wav")},
            )
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# DELETE /events/{id}
# ---------------------------------------------------------------------------

def test_delete_event_returns_204(tmp_stores, tmp_path):
    db, faiss_store, event_store = tmp_stores
    app = create_app(db=db, faiss_store=faiss_store, event_store=event_store)
    with TestClient(app) as c:
        event_id = _insert_event(db, event_store, tmp_path)
        resp = c.delete(f"/events/{event_id}")
        assert resp.status_code == 204


def test_delete_event_removes_from_db(tmp_stores, tmp_path):
    db, faiss_store, event_store = tmp_stores
    app = create_app(db=db, faiss_store=faiss_store, event_store=event_store)
    with TestClient(app) as c:
        event_id = _insert_event(db, event_store, tmp_path)
        c.delete(f"/events/{event_id}")
        resp = c.get(f"/events/{event_id}")
        assert resp.status_code == 404


def test_delete_event_removes_filesystem(tmp_stores, tmp_path):
    db, faiss_store, event_store = tmp_stores
    app = create_app(db=db, faiss_store=faiss_store, event_store=event_store)
    with TestClient(app) as c:
        event_id = _insert_event(db, event_store, tmp_path)
        event = db.get_event(event_id)
        event_dir = Path(event.event_dir)
        assert event_dir.exists()
        c.delete(f"/events/{event_id}")
        assert not event_dir.exists()


def test_delete_event_not_found_returns_404(client):
    resp = client.delete("/events/99999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /events/  (clear all)
# ---------------------------------------------------------------------------

def test_clear_events_returns_204(tmp_stores, tmp_path):
    db, faiss_store, event_store = tmp_stores
    app = create_app(db=db, faiss_store=faiss_store, event_store=event_store)
    with TestClient(app) as c:
        _insert_event(db, event_store, tmp_path)
        _insert_event(db, event_store, tmp_path)
        resp = c.delete("/events/")
        assert resp.status_code == 204


def test_clear_events_empties_db(tmp_stores, tmp_path):
    db, faiss_store, event_store = tmp_stores
    app = create_app(db=db, faiss_store=faiss_store, event_store=event_store)
    with TestClient(app) as c:
        _insert_event(db, event_store, tmp_path)
        _insert_event(db, event_store, tmp_path)
        c.delete("/events/")
        resp = c.get("/events/")
        assert resp.json() == []


def test_clear_events_resets_faiss(tmp_stores, tmp_path):
    db, faiss_store, event_store = tmp_stores
    app = create_app(db=db, faiss_store=faiss_store, event_store=event_store)
    with TestClient(app) as c:
        _insert_event(db, event_store, tmp_path)
        c.delete("/events/")
    assert faiss_store.get_total() == 0


# ---------------------------------------------------------------------------
# POST /internal/reset-detector  +  GET /internal/reset-pending
# ---------------------------------------------------------------------------

def test_reset_detector_sets_pending_flag(client):
    resp = client.post("/internal/reset-detector")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_reset_pending_returns_true_then_clears(client):
    client.post("/internal/reset-detector")
    resp1 = client.get("/internal/reset-pending")
    assert resp1.json()["pending"] is True
    # Second call — flag was cleared
    resp2 = client.get("/internal/reset-pending")
    assert resp2.json()["pending"] is False


def test_reset_pending_is_false_by_default(client):
    resp = client.get("/internal/reset-pending")
    assert resp.json()["pending"] is False
