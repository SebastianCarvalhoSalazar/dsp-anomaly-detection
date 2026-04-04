from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import soundfile as sf

from .config import StorageConfig


class EventStore:
    """Filesystem-based persistence for anomaly event artifacts.

    Directory layout::

        eventos/
          2024-01-15T10-30-00.123456+00-00/
            audio.wav
            frame.jpg        (optional)
            embedding.npy    (optional, added later)
            metadata.json
    """

    def __init__(self, config: StorageConfig | None = None) -> None:
        self._config = config or StorageConfig()

    def save_event(
        self,
        timestamp: datetime,
        audio: np.ndarray,
        sample_rate: int,
        frame: Optional[np.ndarray],
        anomaly_score: float,
        extra_metadata: Optional[dict] = None,
    ) -> Path:
        """Write event artifacts to eventos/<timestamp>/ and return the directory.

        Parameters
        ----------
        timestamp : datetime
            Event time. Should be timezone-aware.
        audio : np.ndarray
            1-D float32 audio samples.
        sample_rate : int
        frame : np.ndarray or None
            BGR frame (H, W, 3) uint8. Skipped if None.
        anomaly_score : float
        extra_metadata : dict, optional
            Additional fields merged into metadata.json.

        Returns
        -------
        Path
            The created event directory.
        """
        # Replace colons for OS compatibility in directory names
        dir_name = timestamp.isoformat().replace(":", "-")
        event_dir = Path(self._config.events_dir) / dir_name
        event_dir.mkdir(parents=True, exist_ok=True)

        audio_path = event_dir / "audio.wav"
        # Use IEEE_FLOAT subtype to preserve full float32 range (PCM_16 would clip >1.0)
        sf.write(str(audio_path), audio, sample_rate, subtype="FLOAT")

        frame_path = None
        if frame is not None:
            frame_path = event_dir / "frame.jpg"
            _, buf = cv2.imencode(".jpg", frame)
            frame_path.write_bytes(buf.tobytes())

        metadata = {
            "timestamp": timestamp.isoformat(),
            "anomaly_score": anomaly_score,
            "sample_rate": sample_rate,
            "audio_samples": len(audio),
            **(extra_metadata or {}),
        }
        (event_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        return event_dir

    def load_audio(self, event_dir: Path) -> tuple[np.ndarray, int]:
        """Load audio.wav. Returns (samples, sample_rate)."""
        data, sr = sf.read(str(event_dir / "audio.wav"), dtype="float32")
        return data, sr

    def load_frame(self, event_dir: Path) -> Optional[np.ndarray]:
        """Load frame.jpg as BGR array. Returns None if not present."""
        p = event_dir / "frame.jpg"
        if not p.exists():
            return None
        return cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)

    def load_metadata(self, event_dir: Path) -> dict:
        """Load metadata.json."""
        return json.loads((event_dir / "metadata.json").read_text())

    def save_embedding(self, event_dir: Path, embedding: np.ndarray) -> Path:
        """Write embedding.npy to event_dir and return the file path."""
        path = event_dir / "embedding.npy"
        np.save(str(path), embedding)
        return path

    def load_embedding(self, event_dir: Path) -> Optional[np.ndarray]:
        """Load embedding.npy. Returns None if not present."""
        p = event_dir / "embedding.npy"
        if not p.exists():
            return None
        return np.load(str(p))

    def delete_event(self, event_dir: Path) -> None:
        """Remove the event directory and all its artifacts from the filesystem."""
        if event_dir.exists():
            shutil.rmtree(event_dir)
