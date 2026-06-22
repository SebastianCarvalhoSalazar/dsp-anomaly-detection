"""Pipeline orchestrator — ties all modules together into the main processing loop.

Run with:
    poetry run python -m src.pipeline

The pipeline reads audio from the microphone and video from the camera, runs
DSP feature extraction, scores each window for anomalies, and when an anomaly
is confirmed: generates multimodal embeddings, persists the event, indexes it
in FAISS, and notifies the API (which broadcasts to WebSocket clients).

Requires a running API server (uvicorn src.api.main:app) to broadcast scores.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import json

import httpx
import numpy as np
import sounddevice as sd

from src.detection import AnomalyDetector, DetectorConfig
from src.dsp import AudioProcessor, DSPConfig
from src.embeddings import MultimodalEncoder
from src.storage import Database, EventStore, FAISSStore, StorageConfig
from src.storage.models import AnomalyEvent
from src.vision import FrameCapture, MotionDetector, VisionConfig

logger = logging.getLogger(__name__)

_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_INTERNAL_URL = _API_BASE + "/internal/score"
API_RESET_PENDING_URL = _API_BASE + "/internal/reset-pending"


class Pipeline:
    """Main processing loop for real-time audiovisual anomaly detection.

    The pipeline is synchronous by design: no asyncio overhead in the hot path.
    Communication with the async FastAPI server happens via HTTP POST to the
    /internal/score endpoint, which the API then broadcasts over WebSocket.
    """

    def __init__(self) -> None:
        config = StorageConfig(
            events_dir=os.getenv("EVENTS_DIR", "eventos"),
            db_path=os.getenv("DB_PATH", "data/events.db"),
            faiss_path=os.getenv("FAISS_PATH", "data/faiss.index"),
        )
        self.dsp = AudioProcessor(DSPConfig())
        # Inject the real feature-name layout (H6) so drift / explainability
        # labels stay correct regardless of the DSP config.
        self.detector = AnomalyDetector(
            DetectorConfig(), feature_names=self.dsp.feature_names
        )
        self.encoder = MultimodalEncoder()

        # Persistent baseline (3.4): try to restore previous session
        if self.detector.load_state():
            logger.info(
                "Loaded detector state from disk "
                "(skipping warmup)"
            )
        self.event_store = EventStore(config)
        self.faiss_store = FAISSStore(config)
        self.db = Database(config)
        self.vision = MotionDetector(VisionConfig())
        self.frame_capture = FrameCapture(VisionConfig())

        # Queue for passing audio windows from the sounddevice callback
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False

        # Rolling buffer of raw audio samples: keeps the last 3 seconds
        # before an anomaly so saved events are meaningful for playback.
        _sr = DSPConfig().sample_rate
        self._pre_audio_buffer: deque[float] = deque(
            maxlen=3 * _sr
        )  # 48 000 samples @ 16 kHz

        # Cross-modal correlation state (3.2)
        self._last_motion_energy: float = 0.0

    def run(self) -> None:
        """Start the pipeline. Blocks until KeyboardInterrupt.

        Embedding models (Wav2Vec2 + DINOv2) are preloaded in a background
        thread so they are ready before the first anomaly is detected,
        without blocking the audio processing hot-path.
        """
        self.db.init()
        self.faiss_store.init()

        # Pre-download / warm-up embedding models in the background so the
        # first anomaly event does not incur a ~60 s model-load penalty.
        threading.Thread(
            target=self.encoder.ensure_downloaded, daemon=True
        ).start()
        logger.info("Background model preload started.")

        # Open camera in a background thread
        camera_thread = threading.Thread(target=self._camera_loop, daemon=True)

        dsp_config = self.dsp._config

        def _audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.warning("Audio callback status: %s", status)
            mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            try:
                self._audio_queue.put_nowait(mono.astype(np.float32))
            except queue.Full:
                pass  # drop the window under heavy load

        self._running = True
        camera_thread.start()
        logger.info("Pipeline started. Press Ctrl+C to stop.")

        try:
            with sd.InputStream(
                samplerate=dsp_config.sample_rate,
                blocksize=dsp_config.window_size,
                channels=1,
                dtype="float32",
                callback=_audio_callback,
            ):
                self._process_loop()
        except KeyboardInterrupt:
            logger.info("Pipeline stopped.")
        finally:
            self._running = False
            self.frame_capture.close()
            # Persistent baseline: save state for next run
            try:
                self.detector.save_state()
                logger.info("Detector state saved.")
            except Exception as exc:
                logger.warning(
                    "Failed to save detector state: %s", exc
                )

    def _check_reset(self) -> None:
        """Poll the API for a pending detector reset request and apply it."""
        try:
            resp = httpx.get(API_RESET_PENDING_URL, timeout=0.5)
            if resp.status_code == 200 and resp.json().get("pending"):
                self.detector.reset()
                self._pre_audio_buffer.clear()
                logger.info("Detector reset: buffer cleared, warmup restarted.")
        except Exception as exc:
            logger.debug("Reset poll failed (API may not be running): %s", exc)

    def _process_loop(self) -> None:
        """Consume audio windows, score them, and handle anomalies."""
        _reset_check_interval = 50  # check for reset every N windows
        _window_count = 0

        while self._running:
            try:
                window = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            _window_count += 1
            if _window_count % _reset_check_interval == 0:
                self._check_reset()

            # Accumulate raw samples for pre-event context before scoring
            self._pre_audio_buffer.extend(window.tolist())

            feature_vec = self.dsp.process_window(window)
            rms = float(np.sqrt(np.mean(window ** 2)))
            result = self.detector.score(feature_vec)

            with self._frame_lock:
                frame = self._latest_frame.copy() if self._latest_frame is not None else None

            boxes = self.vision.detect(frame) if frame is not None else []

            # Cross-modal correlation (3.2)
            if boxes and frame is not None:
                frame_area = float(
                    frame.shape[0] * frame.shape[1]
                )
                self._last_motion_energy = (
                    sum(b.area for b in boxes) / frame_area
                )
                # Source-score ranking (Option B):
                # score = anomaly_score × area_ratio × temporal_weight
                for b in boxes:
                    area_ratio = b.area / frame_area
                    b.source_score = (
                        result.anomaly_score
                        * area_ratio
                        * b.source_score  # temporal weight
                    )
                boxes.sort(
                    key=lambda b: b.source_score,
                    reverse=True,
                )
                boxes = boxes[:1]  # best match only
            else:
                self._last_motion_energy *= 0.9  # decay

            # Notify API (non-blocking fire-and-forget)
            self._notify_api(result, boxes, rms=rms)

            if result.is_anomaly:
                # Snapshot the pre-event buffer (last 3s) as the event audio
                audio_clip = np.array(list(self._pre_audio_buffer), dtype=np.float32)
                self._handle_anomaly(
                    result, audio_clip, frame, boxes,
                    motion_energy=self._last_motion_energy,
                )

    def _handle_anomaly(
        self,
        result,
        audio_window: np.ndarray,
        frame: Optional[np.ndarray],
        boxes,
        *,
        motion_energy: float = 0.0,
    ) -> None:
        """Persist an anomaly event: filesystem → FAISS → SQLite.

        Parameters
        ----------
        motion_energy : float
            Normalised ratio (0–1) of total bounding-box area to frame area
            at the moment the anomaly was confirmed.  Stored in the event
            metadata for later cross-modal analysis.
        """
        ts = datetime.fromtimestamp(result.timestamp, tz=timezone.utc)
        event_dir = self.event_store.save_event(
            timestamp=ts,
            audio=audio_window,
            sample_rate=self.dsp._config.sample_rate,
            frame=frame,
            anomaly_score=result.anomaly_score,
            extra_metadata={
                "window_index": result.window_index,
                "raw_score": result.raw_score,
                "motion_energy": motion_energy,
                "bounding_boxes": [
                    {
                        "x": b.x, "y": b.y,
                        "w": b.w, "h": b.h,
                        "source_score": round(
                            b.source_score, 4,
                        ),
                    }
                    for b in boxes
                ],
            },
        )

        embedding = self.encoder.encode(audio_window, frame=frame)
        self.event_store.save_embedding(event_dir, embedding)
        faiss_id = self.faiss_store.add(embedding)

        orm_event = AnomalyEvent(
            timestamp=ts,
            anomaly_score=result.anomaly_score,
            event_dir=str(event_dir),
            faiss_index_id=faiss_id,
            audio_path=str(event_dir / "audio.wav"),
            frame_path=str(event_dir / "frame.jpg") if frame is not None else None,
            embedding_path=str(event_dir / "embedding.npy"),
            source_region_json=json.dumps([
                {
                    "x": b.x, "y": b.y,
                    "w": b.w, "h": b.h,
                    "source_score": round(
                        b.source_score, 4,
                    ),
                }
                for b in boxes
            ]),
        )
        self.db.save_event(orm_event)
        logger.info("Anomaly event saved: %s (score=%.3f)", event_dir, result.anomaly_score)

    def _camera_loop(self) -> None:
        """Background thread: capture frames and update self._latest_frame."""
        try:
            self.frame_capture.open()
            while self._running:
                frame = self.frame_capture.read()
                if frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame
                time.sleep(1.0 / 25)
        except Exception as exc:
            logger.warning("Camera unavailable: %s", exc)

    def _notify_api(self, result, boxes, *, rms: float = 0.0) -> None:
        """POST anomaly score to API for WebSocket broadcast (fire-and-forget).

        The payload includes ``motion_energy`` and ``rms`` so the dashboard
        can correlate visual activity and audio amplitude in real time.
        """
        try:
            payload = {
                "anomaly_score": result.anomaly_score,
                "is_anomaly": result.is_anomaly,
                "is_fitted": result.is_fitted,
                "timestamp": datetime.fromtimestamp(
                    result.timestamp, tz=timezone.utc
                ).isoformat(),
                "window_index": result.window_index,
                "bounding_boxes": [
                    {
                        "x": b.x, "y": b.y,
                        "w": b.w, "h": b.h,
                        "source_score": round(
                            b.source_score, 4,
                        ),
                    }
                    for b in boxes
                ],
                "motion_energy": self._last_motion_energy,
                "rms": round(rms, 6),
            }
            # Drift detection metrics
            drift = self.detector.get_drift_metrics()
            payload.update(drift)
            httpx.post(API_INTERNAL_URL, json=payload, timeout=0.5)
        except Exception:
            pass  # API may not be running; score notification is best-effort


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Pipeline().run()
