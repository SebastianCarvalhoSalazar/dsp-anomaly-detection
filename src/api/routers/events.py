from __future__ import annotations

import json
from pathlib import Path

import anyio
import cv2
import librosa
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PyEMD import EMD

from src.api.dependencies import get_db, get_event_store, get_faiss_store
from src.api.schemas import EventResponse, OfflineAnalysisResponse
from src.storage import Database, EventStore, FAISSStore
from src.storage.models import AnomalyEvent

router = APIRouter(prefix="/events", tags=["events"])


def _event_to_response(event: AnomalyEvent) -> EventResponse:
    event_dir = Path(event.event_dir)
    return EventResponse(
        id=event.id,
        timestamp=event.timestamp,
        anomaly_score=event.anomaly_score,
        event_dir=str(event.event_dir),
        faiss_index_id=event.faiss_index_id,
        has_audio=(event_dir / "audio.wav").exists(),
        has_frame=(event_dir / "frame.jpg").exists(),
        has_embedding=(event_dir / "embedding.npy").exists(),
    )


@router.get("/", response_model=list[EventResponse])
async def list_events(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    db: Database = Depends(get_db),
) -> list[EventResponse]:
    """List anomaly events ordered by timestamp descending."""
    events = await anyio.to_thread.run_sync(
        lambda: db.list_events(limit=limit, offset=offset, min_score=min_score)
    )
    return [_event_to_response(e) for e in events]


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(
    event_id: int,
    db: Database = Depends(get_db),
) -> EventResponse:
    event = await anyio.to_thread.run_sync(lambda: db.get_event(event_id))
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return _event_to_response(event)


@router.get("/{event_id}/audio")
async def get_audio(
    event_id: int,
    db: Database = Depends(get_db),
) -> FileResponse:
    """Stream the audio.wav file for an event."""
    event = await anyio.to_thread.run_sync(lambda: db.get_event(event_id))
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    audio_path = Path(event.event_dir) / "audio.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(str(audio_path), media_type="audio/wav")


@router.get("/{event_id}/frame")
async def get_frame(
    event_id: int,
    db: Database = Depends(get_db),
) -> FileResponse:
    """Stream the frame.jpg file for an event."""
    event = await anyio.to_thread.run_sync(lambda: db.get_event(event_id))
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    frame_path = Path(event.event_dir) / "frame.jpg"
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(str(frame_path), media_type="image/jpeg")


@router.get("/{event_id}/frame/annotated")
async def get_annotated_frame(
    event_id: int,
    db: Database = Depends(get_db),
) -> Response:
    """Return the event frame with bounding boxes drawn on it.

    Loads the stored JPEG frame and the motion bounding boxes from
    ``source_region_json``.  Each box is drawn as a coloured
    rectangle with a small label showing its area percentage.
    If no boxes are stored the plain frame is returned.
    """
    event = await anyio.to_thread.run_sync(
        lambda: db.get_event(event_id)
    )
    if event is None:
        raise HTTPException(
            status_code=404, detail="Event not found"
        )
    frame_path = Path(event.event_dir) / "frame.jpg"
    if not frame_path.exists():
        raise HTTPException(
            status_code=404, detail="Frame not found"
        )

    def _annotate() -> bytes:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            with open(str(frame_path), "rb") as f:
                return f.read()

        boxes: list[dict] = []
        if event.source_region_json:
            try:
                boxes = json.loads(event.source_region_json)
            except (json.JSONDecodeError, TypeError):
                pass

        if not boxes:
            with open(str(frame_path), "rb") as f:
                return f.read()

        h, w = frame.shape[:2]

        # Pick the single best box (highest source_score)
        best = max(
            boxes,
            key=lambda b: b.get("source_score", 0),
        )
        bx, by = int(best["x"]), int(best["y"])
        bw, bh = int(best["w"]), int(best["h"])
        ss = best.get("source_score", 0)

        colour = (60, 60, 230)  # red BGR
        cv2.rectangle(
            frame,
            (bx, by),
            (bx + bw, by + bh),
            colour,
            2,
        )

        label = f"source {ss:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(
            label, font, 0.5, 1,
        )
        cv2.rectangle(
            frame,
            (bx, by - th - 6),
            (bx + tw + 4, by),
            colour,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (bx + 2, by - 4),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        _, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        return buf.tobytes()

    img_bytes = await anyio.to_thread.run_sync(_annotate)
    return Response(
        content=img_bytes,
        media_type="image/jpeg",
    )


@router.delete("/{event_id}", status_code=204)
async def delete_event(
    event_id: int,
    db: Database = Depends(get_db),
    event_store: EventStore = Depends(get_event_store),
) -> None:
    """Delete a single event: removes the DB row and filesystem directory.

    The FAISS entry for this event becomes orphaned but is harmless — the
    search router already skips FAISS IDs with no matching DB row.
    """
    event = await anyio.to_thread.run_sync(lambda: db.get_event(event_id))
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    event_dir = Path(event.event_dir)
    await anyio.to_thread.run_sync(lambda: db.delete_event(event_id))
    await anyio.to_thread.run_sync(lambda: event_store.delete_event(event_dir))


@router.delete("/", status_code=204)
async def clear_events(
    db: Database = Depends(get_db),
    event_store: EventStore = Depends(get_event_store),
    faiss_store: FAISSStore = Depends(get_faiss_store),
) -> None:
    """Delete all events: clears DB, all event directories, and the FAISS index."""
    events = await anyio.to_thread.run_sync(lambda: db.list_events(limit=10_000))
    await anyio.to_thread.run_sync(lambda: db.clear_events())
    for event in events:
        await anyio.to_thread.run_sync(
            lambda d=Path(event.event_dir): event_store.delete_event(d)
        )
    await anyio.to_thread.run_sync(faiss_store.clear)


@router.get("/{event_id}/offline_analysis", response_model=OfflineAnalysisResponse)
async def get_offline_analysis(
    event_id: int,
    db: Database = Depends(get_db),
    event_store: EventStore = Depends(get_event_store),
) -> OfflineAnalysisResponse:
    """Run EMD and compute mel spectrogram on the stored audio for an event."""
    event = await anyio.to_thread.run_sync(lambda: db.get_event(event_id))
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")

    def _analyze() -> OfflineAnalysisResponse:
        audio, sr = event_store.load_audio(Path(event.event_dir))
        signal = audio.astype(np.float64)

        emd = EMD()
        imfs = emd.emd(signal)  # (n_imfs, time) or (time,) if single IMF
        if imfs.ndim == 1:
            imfs = imfs[np.newaxis, :]

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        return OfflineAnalysisResponse(
            imfs=[row.tolist() for row in imfs],
            n_imfs=len(imfs),
            sample_rate=sr,
            spectrogram=[row.tolist() for row in mel_db],
        )

    return await anyio.to_thread.run_sync(_analyze)
