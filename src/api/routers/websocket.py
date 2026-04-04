from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from src.api.schemas import AnomalyScoreMessage

logger = logging.getLogger(__name__)
router = APIRouter()

# Active WebSocket connections; updated under asyncio — no lock needed
_connections: set[WebSocket] = set()


@router.websocket("/ws/stream")
async def stream_scores(websocket: WebSocket) -> None:
    """Push AnomalyScoreMessage JSON to all connected dashboard clients."""
    await websocket.accept()
    _connections.add(websocket)
    try:
        while True:
            # Keep-alive: wait for any client message (ping) or disconnection
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _connections.discard(websocket)


@router.post("/internal/score", include_in_schema=False)
async def receive_score(message: AnomalyScoreMessage) -> dict:
    """Internal endpoint called by the pipeline process to broadcast scores.

    The pipeline sends anomaly scores via HTTP POST; the API forwards them
    to all connected WebSocket clients. This avoids shared event-loop state
    between the synchronous pipeline process and the async API process.
    """
    await broadcast(message.model_dump())
    return {"ok": True}


async def broadcast(message: dict) -> None:
    """Serialize and send a message to all connected WebSocket clients."""
    if not _connections:
        return
    payload = json.dumps(message)
    dead = set()
    for ws in list(_connections):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    _connections.difference_update(dead)


@router.post("/internal/reset-detector", include_in_schema=False)
async def request_detector_reset(request: Request) -> dict:
    """Signal the pipeline to reset its AnomalyDetector and audio buffer.

    Sets a flag on app.state that the pipeline polls via GET /internal/reset-pending.
    """
    request.app.state.detector_reset_pending = True
    return {"ok": True}


@router.get("/internal/reset-pending", include_in_schema=False)
async def check_reset_pending(request: Request) -> dict:
    """Pipeline polls this endpoint; returns pending=True once then auto-clears."""
    pending = getattr(request.app.state, "detector_reset_pending", False)
    if pending:
        request.app.state.detector_reset_pending = False
    return {"pending": pending}
