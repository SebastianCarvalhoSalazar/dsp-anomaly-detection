"""Backward-compatibility test for the event schema migration (Fase 1).

A database created with the *old* (v0.2.0) column set must keep working:
``Database.init()`` adds the new multimodal columns idempotently and old
rows remain readable with the new fields as NULL.
"""

from datetime import datetime, timezone

import sqlalchemy as sa

from src.storage import Database, StorageConfig
from src.storage.db import _ADDED_COLUMNS

# The original v0.2.0 columns, before any multimodal fields existed.
_OLD_DDL = """
CREATE TABLE anomaly_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    anomaly_score FLOAT NOT NULL,
    event_dir VARCHAR(512) NOT NULL UNIQUE,
    faiss_index_id INTEGER,
    audio_path VARCHAR(512),
    frame_path VARCHAR(512),
    embedding_path VARCHAR(512),
    source_region_json TEXT,
    extra_json TEXT
)
"""


def _make_old_db(path: str) -> None:
    engine = sa.create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        conn.execute(sa.text(_OLD_DDL))
        conn.execute(
            sa.text(
                "INSERT INTO anomaly_events "
                "(timestamp, anomaly_score, event_dir) "
                "VALUES (:ts, :s, :d)"
            ),
            {"ts": datetime.now(timezone.utc), "s": 0.9, "d": "eventos/old-1"},
        )
    engine.dispose()


def test_migration_adds_columns_and_preserves_old_rows(tmp_path):
    db_path = str(tmp_path / "events.db")
    _make_old_db(db_path)

    db = Database(StorageConfig(db_path=db_path))
    db.init()  # should ALTER TABLE ADD COLUMN for each new field

    # All new columns now exist on the table.
    insp = sa.inspect(sa.create_engine(f"sqlite:///{db_path}"))
    cols = {c["name"] for c in insp.get_columns("anomaly_events")}
    assert set(_ADDED_COLUMNS).issubset(cols)

    # The pre-existing row is still readable; new fields are NULL.
    events = db.list_events(limit=10)
    assert len(events) == 1
    assert events[0].event_dir == "eventos/old-1"
    assert events[0].audio_score is None
    assert events[0].dominant_modality is None


def test_migration_is_idempotent(tmp_path):
    db_path = str(tmp_path / "events.db")
    db = Database(StorageConfig(db_path=db_path))
    db.init()
    # Running init() again on an already-migrated DB must not raise.
    db.init()
    events = db.list_events(limit=10)
    assert events == []
