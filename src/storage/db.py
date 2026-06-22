from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session

from .config import StorageConfig
from .models import AnomalyEvent, Base

# Columns added after v0.2.0. ``create_all`` handles fresh databases; this
# map drives an idempotent ALTER for databases created by older versions.
_ADDED_COLUMNS: dict[str, str] = {
    "audio_score": "FLOAT",
    "video_score": "FLOAT",
    "combined_score": "FLOAT",
    "fast_audio_score": "FLOAT",
    "slow_audio_score": "FLOAT",
    "fast_video_score": "FLOAT",
    "slow_video_score": "FLOAT",
    "top_audio_features": "TEXT",
    "top_video_features": "TEXT",
    "dominant_modality": "VARCHAR(32)",
}


class Database:
    """SQLite metadata store using synchronous SQLAlchemy.

    All methods are synchronous. FastAPI routes call them via
    ``anyio.to_thread.run_sync`` to avoid blocking the event loop.
    """

    def __init__(self, config: StorageConfig | None = None) -> None:
        self._config = config or StorageConfig()
        db_url = f"sqlite:///{self._config.db_path}"
        os.makedirs(os.path.dirname(self._config.db_path) or ".", exist_ok=True)
        self._engine = create_engine(db_url, connect_args={"check_same_thread": False})
        self._Session = sessionmaker(bind=self._engine)

    def init(self) -> None:
        """Create all tables if they do not exist, then run migrations."""
        Base.metadata.create_all(self._engine)
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add columns introduced after v0.2.0 to a pre-existing table.

        Idempotent: only columns missing from the live table are added, and
        all are nullable, so events written by older versions remain valid.
        """
        insp = inspect(self._engine)
        if not insp.has_table(AnomalyEvent.__tablename__):
            return
        existing = {c["name"] for c in insp.get_columns(AnomalyEvent.__tablename__)}
        missing = {n: t for n, t in _ADDED_COLUMNS.items() if n not in existing}
        if not missing:
            return
        with self._engine.begin() as conn:
            for name, sqltype in missing.items():
                conn.execute(
                    text(
                        f"ALTER TABLE {AnomalyEvent.__tablename__} "
                        f"ADD COLUMN {name} {sqltype}"
                    )
                )

    def save_event(self, event: AnomalyEvent) -> int:
        """Persist a new anomaly event. Returns the assigned database id."""
        with self._Session() as session:
            session.add(event)
            session.commit()
            session.refresh(event)
            return event.id

    def update_faiss_id(self, event_id: int, faiss_index_id: int) -> None:
        """Set faiss_index_id after embedding is stored in FAISS."""
        with self._Session() as session:
            event = session.get(AnomalyEvent, event_id)
            if event is not None:
                event.faiss_index_id = faiss_index_id
                session.commit()

    def list_events(
        self,
        limit: int = 50,
        offset: int = 0,
        min_score: float = 0.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[AnomalyEvent]:
        """Query events with optional filters, ordered by timestamp DESC."""
        with self._Session() as session:
            q = session.query(AnomalyEvent).filter(
                AnomalyEvent.anomaly_score >= min_score
            )
            if start_time is not None:
                q = q.filter(AnomalyEvent.timestamp >= start_time)
            if end_time is not None:
                q = q.filter(AnomalyEvent.timestamp <= end_time)
            q = q.order_by(AnomalyEvent.timestamp.desc()).offset(offset).limit(limit)
            # Expunge so objects can be used outside the session
            results = q.all()
            for r in results:
                session.expunge(r)
            return results

    def get_event(self, event_id: int) -> Optional[AnomalyEvent]:
        """Fetch a single event by primary key. Returns None if not found."""
        with self._Session() as session:
            event = session.get(AnomalyEvent, event_id)
            if event is not None:
                session.expunge(event)
            return event

    def get_event_by_faiss_id(self, faiss_id: int) -> Optional[AnomalyEvent]:
        """Fetch a single event by its FAISS index ID.

        This is used by the similarity search endpoint to resolve FAISS
        result IDs back to event metadata in O(1) per result, avoiding
        the previous O(N) full-table scan.

        Returns None if no event has the given ``faiss_index_id``.
        """
        with self._Session() as session:
            event = (
                session.query(AnomalyEvent)
                .filter(AnomalyEvent.faiss_index_id == faiss_id)
                .first()
            )
            if event is not None:
                session.expunge(event)
            return event

    def delete_event(self, event_id: int) -> bool:
        """Delete a single event row by primary key. Returns True if it existed."""
        with self._Session() as session:
            event = session.get(AnomalyEvent, event_id)
            if event is None:
                return False
            session.delete(event)
            session.commit()
            return True

    def clear_events(self) -> int:
        """Delete all event rows. Returns the number of rows removed."""
        with self._Session() as session:
            count = session.query(AnomalyEvent).count()
            session.query(AnomalyEvent).delete()
            session.commit()
            return count
