from __future__ import annotations

from typing import Any, Optional

import httpx


class APIClient:
    """Synchronous HTTP client for the FastAPI backend.

    Uses httpx in sync mode — appropriate for Streamlit, which runs in its
    own thread model and does not use asyncio natively.
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base_url = base_url.rstrip("/")

    def list_events(
        self,
        limit: int = 50,
        min_score: float = 0.0,
        offset: int = 0,
    ) -> list[dict]:
        """Fetch anomaly events ordered by timestamp descending."""
        resp = httpx.get(
            f"{self._base_url}/events/",
            params={"limit": limit, "min_score": min_score, "offset": offset},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()

    def get_event(self, event_id: int) -> dict:
        """Fetch a single event by ID. Raises httpx.HTTPStatusError if not found."""
        resp = httpx.get(f"{self._base_url}/events/{event_id}", timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def get_audio_url(self, event_id: int) -> str:
        """Return the URL to stream the audio file for an event."""
        return f"{self._base_url}/events/{event_id}/audio"

    def get_frame_url(self, event_id: int) -> str:
        """Return the URL to stream the frame image for an event."""
        return f"{self._base_url}/events/{event_id}/frame"

    def get_annotated_frame_url(self, event_id: int) -> str:
        """Return the URL for the frame with bounding boxes drawn."""
        return f"{self._base_url}/events/{event_id}/frame/annotated"

    def search_similar(
        self,
        file_bytes: bytes,
        filename: str,
        modality: str = "audio",
        k: int = 5,
    ) -> list[dict]:
        """Upload a file and return the k most similar events."""
        content_type = "audio/wav" if modality == "audio" else "image/jpeg"
        resp = httpx.post(
            f"{self._base_url}/search/similar",
            params={"modality": modality, "k": k},
            files={"file": (filename, file_bytes, content_type)},
            timeout=120.0,  # first call loads Wav2Vec2+DINOv2 (~60s); subsequent calls are fast
        )
        resp.raise_for_status()
        return resp.json()

    def search_by_event(
        self,
        event_id: int,
        k: int = 5,
    ) -> list[dict]:
        """Find events similar to an already-stored event.

        Uses the pre-computed embedding on the server, so no model
        loading is needed and the response is near-instant.
        """
        resp = httpx.get(
            f"{self._base_url}/search/similar/by-event/{event_id}",
            params={"k": k},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()

    def get_offline_analysis(self, event_id: int) -> dict:
        """Run EMD analysis on a stored event. Returns IMFs and spectrogram."""
        resp = httpx.get(
            f"{self._base_url}/events/{event_id}/offline_analysis",
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()

    def delete_event(self, event_id: int) -> None:
        """Delete a single event by ID (DB row + filesystem artifacts)."""
        resp = httpx.delete(f"{self._base_url}/events/{event_id}", timeout=10.0)
        resp.raise_for_status()

    def clear_events(self) -> None:
        """Delete all events and reset the FAISS index."""
        resp = httpx.delete(f"{self._base_url}/events/", timeout=30.0)
        resp.raise_for_status()

    def reset_detector(self) -> None:
        """Signal the pipeline to reset its AnomalyDetector (restart warmup)."""
        resp = httpx.post(f"{self._base_url}/internal/reset-detector", timeout=5.0)
        resp.raise_for_status()

    def set_fusion_config(
        self,
        strategy: str,
        audio_weight: float,
        gates: bool,
    ) -> dict:
        """Push the live fusion config to the pipeline (strategy, weight, gating)."""
        resp = httpx.post(
            f"{self._base_url}/internal/fusion-config",
            json={
                "strategy": strategy,
                "audio_weight": audio_weight,
                "gates": gates,
            },
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json()

    def get_fusion_config(self) -> dict:
        """Fetch the current live fusion config."""
        resp = httpx.get(f"{self._base_url}/internal/fusion-config", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
