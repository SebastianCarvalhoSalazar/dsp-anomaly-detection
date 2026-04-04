from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .config import StorageConfig
from .models import AnomalyEvent, Base


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
        """Create all tables if they do not exist."""
        Base.metadata.create_all(self._engine)

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
