"""Tests for storage backends using temp directories and in-memory SQLite."""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.storage.config import StorageConfig
from src.storage.db import Database
from src.storage.event_store import EventStore
from src.storage.faiss_store import FAISSStore
from src.storage.models import AnomalyEvent


# ---------------------------------------------------------------------------
# EventStore
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_event_store(tmp_path):
    config = StorageConfig(events_dir=str(tmp_path / "eventos"))
    return EventStore(config)


def _make_audio(n: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal(n).astype(np.float32)


def test_event_store_creates_audio_file(tmp_event_store, tmp_path):
    ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    audio = _make_audio()
    event_dir = tmp_event_store.save_event(ts, audio, 16000, None, anomaly_score=0.8)
    assert (event_dir / "audio.wav").exists()


def test_event_store_creates_metadata(tmp_event_store):
    ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
    audio = _make_audio()
    event_dir = tmp_event_store.save_event(ts, audio, 16000, None, anomaly_score=0.75)
    meta = tmp_event_store.load_metadata(event_dir)
    assert abs(meta["anomaly_score"] - 0.75) < 1e-6
    assert meta["sample_rate"] == 16000


def test_event_store_audio_roundtrip(tmp_event_store):
    ts = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
    audio = _make_audio(2048)
    event_dir = tmp_event_store.save_event(ts, audio, 16000, None, anomaly_score=0.5)
    loaded, sr = tmp_event_store.load_audio(event_dir)
    assert sr == 16000
    assert loaded.shape == audio.shape
    np.testing.assert_allclose(loaded, audio, atol=1e-4)


def test_event_store_no_frame_when_none(tmp_event_store):
    ts = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    event_dir = tmp_event_store.save_event(ts, _make_audio(), 16000, frame=None, anomaly_score=0.3)
    assert not (event_dir / "frame.jpg").exists()
    assert tmp_event_store.load_frame(event_dir) is None


def test_event_store_frame_roundtrip(tmp_event_store):
    ts = datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc)
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    event_dir = tmp_event_store.save_event(ts, _make_audio(), 16000, frame=frame, anomaly_score=0.9)
    assert (event_dir / "frame.jpg").exists()
    loaded = tmp_event_store.load_frame(event_dir)
    assert loaded is not None
    assert loaded.shape == (64, 64, 3)


def test_event_store_embedding_roundtrip(tmp_event_store):
    ts = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
    event_dir = tmp_event_store.save_event(ts, _make_audio(), 16000, None, anomaly_score=0.6)
    emb = np.random.randn(1536).astype(np.float32)
    tmp_event_store.save_embedding(event_dir, emb)
    loaded = tmp_event_store.load_embedding(event_dir)
    assert loaded is not None
    np.testing.assert_allclose(loaded, emb, rtol=1e-5)


def test_event_store_no_embedding_returns_none(tmp_event_store):
    ts = datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
    event_dir = tmp_event_store.save_event(ts, _make_audio(), 16000, None, anomaly_score=0.4)
    assert tmp_event_store.load_embedding(event_dir) is None


def test_event_store_extra_metadata(tmp_event_store):
    ts = datetime(2024, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
    event_dir = tmp_event_store.save_event(
        ts, _make_audio(), 16000, None, anomaly_score=0.55,
        extra_metadata={"window_index": 42}
    )
    meta = tmp_event_store.load_metadata(event_dir)
    assert meta["window_index"] == 42


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    config = StorageConfig(db_path=str(tmp_path / "test.db"))
    database = Database(config)
    database.init()
    return database


def _make_orm_event(ts=None, score=0.7, event_dir="eventos/test") -> AnomalyEvent:
    return AnomalyEvent(
        timestamp=ts or datetime(2024, 1, 1, tzinfo=timezone.utc),
        anomaly_score=score,
        event_dir=event_dir,
    )


def test_db_save_and_get(db):
    event = _make_orm_event()
    event_id = db.save_event(event)
    assert isinstance(event_id, int)
    loaded = db.get_event(event_id)
    assert loaded is not None
    assert abs(loaded.anomaly_score - 0.7) < 1e-6


def test_db_get_nonexistent_returns_none(db):
    assert db.get_event(9999) is None


def test_db_list_events(db):
    for i in range(5):
        db.save_event(_make_orm_event(score=0.1 * (i + 1), event_dir=f"eventos/ev{i}"))
    events = db.list_events(limit=10)
    assert len(events) == 5


def test_db_list_events_min_score_filter(db):
    db.save_event(_make_orm_event(score=0.3, event_dir="eventos/low"))
    db.save_event(_make_orm_event(score=0.8, event_dir="eventos/high"))
    events = db.list_events(min_score=0.5)
    assert all(e.anomaly_score >= 0.5 for e in events)
    assert len(events) == 1


def test_db_update_faiss_id(db):
    event = _make_orm_event(event_dir="eventos/faiss_test")
    event_id = db.save_event(event)
    db.update_faiss_id(event_id, 42)
    loaded = db.get_event(event_id)
    assert loaded.faiss_index_id == 42


def test_db_get_event_by_faiss_id(db):
    """get_event_by_faiss_id returns the correct event."""
    event = _make_orm_event(event_dir="eventos/faiss_lookup")
    event_id = db.save_event(event)
    db.update_faiss_id(event_id, 77)
    found = db.get_event_by_faiss_id(77)
    assert found is not None
    assert found.id == event_id
    assert found.faiss_index_id == 77


def test_db_get_event_by_faiss_id_not_found(db):
    """Returns None when no event has the given faiss_index_id."""
    assert db.get_event_by_faiss_id(9999) is None


def test_db_get_event_by_faiss_id_none_id(db):
    """Events without a faiss_index_id are not returned."""
    db.save_event(_make_orm_event(event_dir="eventos/no_fid"))
    assert db.get_event_by_faiss_id(0) is None


def test_db_list_events_ordered_desc(db):
    from datetime import timedelta
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(3):
        db.save_event(_make_orm_event(
            ts=base + timedelta(hours=i),
            event_dir=f"eventos/ordered{i}"
        ))
    events = db.list_events()
    timestamps = [e.timestamp for e in events]
    assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# FAISSStore
# ---------------------------------------------------------------------------

@pytest.fixture
def faiss_store(tmp_path):
    config = StorageConfig(
        faiss_path=str(tmp_path / "test.index"),
        embedding_dim=1536,
    )
    store = FAISSStore(config)
    store.init()
    return store


def _make_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1536).astype(np.float32)
    return v / np.linalg.norm(v)


def test_faiss_add_echoes_explicit_id(faiss_store):
    for explicit_id in (10, 20, 30):
        returned = faiss_store.add(_make_embedding(explicit_id), faiss_id=explicit_id)
        assert returned == explicit_id


def test_faiss_search_returns_explicit_ids(faiss_store):
    """search() returns the caller-supplied IDs (the SQLite PKs), not positions."""
    faiss_store.add(_make_embedding(0), faiss_id=100)
    faiss_store.add(_make_embedding(1), faiss_id=200)
    emb = _make_embedding(5)
    faiss_store.add(emb, faiss_id=300)
    D, I = faiss_store.search(emb, k=1)
    assert int(I[0, 0]) == 300  # the matching vector's explicit id


def test_faiss_get_total(faiss_store):
    for i in range(3):
        faiss_store.add(_make_embedding(i), faiss_id=i)
    assert faiss_store.get_total() == 3


def test_faiss_search_shape(faiss_store):
    for i in range(5):
        faiss_store.add(_make_embedding(i), faiss_id=i)
    D, I = faiss_store.search(_make_embedding(0), k=3)
    assert D.shape == (1, 3)
    assert I.shape == (1, 3)


def test_faiss_identical_vector_cosine_similarity_is_one(faiss_store):
    emb = _make_embedding(7)
    faiss_store.add(emb, faiss_id=7)
    D, I = faiss_store.search(emb, k=1)
    assert abs(D[0, 0] - 1.0) < 1e-5


def test_faiss_search_empty_returns_empty(faiss_store):
    D, I = faiss_store.search(_make_embedding(0), k=5)
    assert D.shape[1] == 0


def test_faiss_remove_by_id(faiss_store):
    """Removing by ID drops the vector (no orphan) and is reflected in search."""
    faiss_store.add(_make_embedding(0), faiss_id=1)
    faiss_store.add(_make_embedding(1), faiss_id=2)
    assert faiss_store.remove(1) == 1
    assert faiss_store.get_total() == 1
    # The remaining vector keeps its explicit id.
    _, I = faiss_store.search(_make_embedding(1), k=1)
    assert int(I[0, 0]) == 2


def test_faiss_dim_mismatch_raises(faiss_store):
    import numpy as _np
    with pytest.raises(ValueError):
        faiss_store.add(_np.zeros(10, dtype=_np.float32), faiss_id=1)


def test_faiss_persists_and_reloads(tmp_path):
    config = StorageConfig(
        faiss_path=str(tmp_path / "persist.index"),
        embedding_dim=1536,
    )
    store1 = FAISSStore(config)
    store1.init()
    for i in range(5):
        store1.add(_make_embedding(i), faiss_id=i)

    # Reload from disk preserves vectors and their explicit ids.
    store2 = FAISSStore(config)
    store2.init()
    assert store2.get_total() == 5
    _, I = store2.search(_make_embedding(3), k=1)
    assert int(I[0, 0]) == 3


# ---------------------------------------------------------------------------
# Database — delete / clear
# ---------------------------------------------------------------------------

def test_db_delete_event_removes_row(db):
    event_id = db.save_event(_make_orm_event(event_dir="eventos/del_me"))
    assert db.get_event(event_id) is not None
    result = db.delete_event(event_id)
    assert result is True
    assert db.get_event(event_id) is None


def test_db_delete_nonexistent_returns_false(db):
    assert db.delete_event(99999) is False


def test_db_clear_events_removes_all(db):
    for i in range(4):
        db.save_event(_make_orm_event(event_dir=f"eventos/clear{i}"))
    count = db.clear_events()
    assert count == 4
    assert db.list_events() == []


def test_db_clear_events_empty_returns_zero(db):
    assert db.clear_events() == 0


# ---------------------------------------------------------------------------
# EventStore — delete_event
# ---------------------------------------------------------------------------

def test_event_store_delete_removes_directory(tmp_event_store, tmp_path):
    ts = datetime(2024, 2, 1, tzinfo=timezone.utc)
    event_dir = tmp_event_store.save_event(ts, _make_audio(), 16000, None, anomaly_score=0.5)
    assert event_dir.exists()
    tmp_event_store.delete_event(event_dir)
    assert not event_dir.exists()


def test_event_store_delete_nonexistent_dir_is_noop(tmp_event_store, tmp_path):
    # A non-existent path *inside* the events dir must not raise
    inner = Path(tmp_path) / "eventos" / "nonexistent"
    tmp_event_store.delete_event(inner)  # must not raise


def test_event_store_validate_rejects_traversal(tmp_path):
    config = StorageConfig(events_dir=str(tmp_path / "eventos"))
    store = EventStore(config)
    outside = tmp_path / ".." / "escape"
    with pytest.raises(ValueError, match="outside the configured events directory"):
        store.load_audio(outside)


def test_event_store_validate_rejects_absolute_outside(tmp_path):
    config = StorageConfig(events_dir=str(tmp_path / "eventos"))
    store = EventStore(config)
    with pytest.raises(ValueError, match="outside the configured events directory"):
        store.load_embedding(Path("/tmp/evil"))


def test_safe_event_dir_accepts_inside_rejects_traversal(tmp_path):
    """H2: safe_event_dir returns an absolute path for valid dirs and rejects
    anything resolving outside the events directory."""
    root = tmp_path / "eventos"
    config = StorageConfig(events_dir=str(root))
    store = EventStore(config)

    good = store.safe_event_dir(str(root / "2024-01-01T00-00-00"))
    assert good.is_absolute()

    with pytest.raises(ValueError):
        store.safe_event_dir(str(root / ".." / "etc" / "passwd"))


# ---------------------------------------------------------------------------
# FAISSStore — clear
# ---------------------------------------------------------------------------

def test_faiss_clear_resets_to_zero(faiss_store):
    for i in range(5):
        faiss_store.add(_make_embedding(i), faiss_id=i)
    assert faiss_store.get_total() == 5
    faiss_store.clear()
    assert faiss_store.get_total() == 0


def test_faiss_clear_persists_empty_index(tmp_path):
    config = StorageConfig(
        faiss_path=str(tmp_path / "clear.index"),
        embedding_dim=1536,
    )
    store = FAISSStore(config)
    store.init()
    store.add(_make_embedding(0), faiss_id=0)
    store.clear()

    # Reloading should show 0 vectors
    store2 = FAISSStore(config)
    store2.init()
    assert store2.get_total() == 0


def test_faiss_usable_after_clear(faiss_store):
    faiss_store.add(_make_embedding(0), faiss_id=1)
    faiss_store.clear()
    faiss_store.add(_make_embedding(1), faiss_id=2)
    assert faiss_store.get_total() == 1


def test_faiss_reload_picks_up_vectors_from_another_instance(tmp_path):
    """Simulates pipeline (writer) and API (reader) as separate processes."""
    config = StorageConfig(
        faiss_path=str(tmp_path / "shared.index"),
        embedding_dim=1536,
    )
    # Pipeline instance: adds vectors and persists to disk
    writer = FAISSStore(config)
    writer.init()
    for i in range(3):
        writer.add(_make_embedding(i), faiss_id=i)

    # API instance: loaded at startup when index was empty
    reader = FAISSStore(config)
    reader.init()  # reloads from disk — should already see the 3 vectors

    # Simulate pipeline adding more vectors AFTER the API started
    writer.add(_make_embedding(99), faiss_id=99)

    # Before reload, reader still sees 3
    assert reader.get_total() == 3

    # After reload, reader sees all 4
    reader.reload()
    assert reader.get_total() == 4


def test_faiss_reload_when_no_file_creates_empty_index(tmp_path):
    config = StorageConfig(
        faiss_path=str(tmp_path / "missing.index"),
        embedding_dim=1536,
    )
    store = FAISSStore(config)
    store.init()
    store.reload()  # file doesn't exist yet — must not raise
    assert store.get_total() == 0
