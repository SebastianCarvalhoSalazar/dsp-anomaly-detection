from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from src.storage import Database, EventStore, FAISSStore, StorageConfig
from src.api.routers import events, search, websocket

logger = logging.getLogger(__name__)


def create_app(
    db: Optional[Database] = None,
    faiss_store: Optional[FAISSStore] = None,
    event_store: Optional[EventStore] = None,
) -> FastAPI:
    """FastAPI application factory.

    Accepts optional store instances for dependency injection in tests.
    When not provided, stores are initialized from environment variables
    or default StorageConfig values.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        config = StorageConfig(
            events_dir=os.getenv("EVENTS_DIR", "eventos"),
            db_path=os.getenv("DB_PATH", "data/events.db"),
            faiss_path=os.getenv("FAISS_PATH", "data/faiss.index"),
        )
        application.state.db = db or Database(config)
        application.state.faiss_store = faiss_store or FAISSStore(config)
        application.state.event_store = event_store or EventStore(config)
        application.state.db.init()
        application.state.faiss_store.init()
        application.state.detector_reset_pending = False
        # Live fusion config the dashboard sets and the pipeline polls.
        application.state.fusion_config = {
            "strategy": "weighted",
            "audio_weight": 0.5,
            "gates": False,
        }

        # Pre-load Wav2Vec2 + DINOv2 in a background thread so the first
        # similarity-search request does not block for ~60 s.
        def _preload_encoder() -> None:
            try:
                from src.api.routers.search import _get_encoder
                _get_encoder()
                logger.info("Encoder models preloaded successfully.")
            except Exception as exc:
                logger.warning("Encoder preload failed: %s", exc)

        threading.Thread(target=_preload_encoder, daemon=True).start()

        yield

    app = FastAPI(
        title="DSP Anomaly Detection API",
        description="Real-time audiovisual anomaly detection with multimodal embeddings",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(events.router)
    app.include_router(search.router)
    app.include_router(websocket.router)

    return app


# Entry point for `uvicorn src.api.main:app`
app = create_app()
