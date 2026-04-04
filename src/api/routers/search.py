from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import anyio
import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile

from src.api.dependencies import get_db, get_faiss_store
from src.api.routers.events import _event_to_response
from src.api.schemas import SimilarEventResponse
from src.storage import Database, FAISSStore

router = APIRouter(prefix="/search", tags=["search"])


def _encode_audio_query(audio_bytes: bytes) -> np.ndarray:
    """Encode uploaded audio to a 1536-dim embedding for similarity search.

    Uses the MultimodalEncoder with no image (image half will be zeros).
    Import is deferred to avoid loading models at API startup.
    """
    from src.embeddings import MultimodalEncoder
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]  # take first channel
    encoder = MultimodalEncoder()
    return encoder.encode(audio, frame=None, sample_rate=sr)


def _encode_image_query(image_bytes: bytes) -> np.ndarray:
    """Encode uploaded image to a 1536-dim embedding (audio half zeros)."""
    import cv2
    from src.embeddings import MultimodalEncoder
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")
    # Pad audio half with zeros; normalize in encoder
    audio_zeros = np.zeros(16000, dtype=np.float32)
    encoder = MultimodalEncoder()
    # Build manually: zero audio embedding + real image embedding
    from src.embeddings.encoder import _build_multimodal
    return _build_multimodal(np.zeros(768, dtype=np.float32), encoder.image_encoder.encode(frame))


@router.post("/similar", response_model=list[SimilarEventResponse])
async def search_similar(
    file: UploadFile,
    modality: Literal["audio", "image"] = Query("audio"),
    k: int = Query(5, ge=1, le=20),
    db: Database = Depends(get_db),
    faiss_store: FAISSStore = Depends(get_faiss_store),
) -> list[SimilarEventResponse]:
    """Search for the k most similar events to an uploaded audio or image file."""
    file_bytes = await file.read()

    def _search() -> list[SimilarEventResponse]:
        if faiss_store.get_total() == 0:
            return []

        if modality == "audio":
            query_emb = _encode_audio_query(file_bytes)
        else:
            query_emb = _encode_image_query(file_bytes)

        distances, ids = faiss_store.search(query_emb, k=k)
        results = []
        for dist, fid in zip(distances[0], ids[0]):
            if fid < 0:
                continue
            # faiss_index_id maps directly to FAISS sequential id
            events = db.list_events(limit=1000)
            matching = [e for e in events if e.faiss_index_id == int(fid)]
            if not matching:
                continue
            results.append(
                SimilarEventResponse(
                    event=_event_to_response(matching[0]),
                    cosine_similarity=float(dist),
                )
            )
        return results

    return await anyio.to_thread.run_sync(_search)
