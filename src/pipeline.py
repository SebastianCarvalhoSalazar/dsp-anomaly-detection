"""Pipeline orchestrator — ties all modules together into the main processing loop.

Run with:
    poetry run python -m src.pipeline

The pipeline reads audio from the microphone and frames from the camera, runs
DSP feature extraction and scores each audio window for anomalies. Consecutive
anomalous windows form one event, which is persisted off the audio loop (audio
clip, frame, multimodal embedding, FAISS index, SQLite row). Every window's
score is sent to the API, which broadcasts it to WebSocket clients.

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
from dotenv import load_dotenv

# Carga variables desde el .env de la raíz del repo (si existe) para que
# EVENTS_DIR, DB_PATH, CORS_ORIGINS, ENABLE_DRIFT, etc. surtan efecto sin
# tener que exportarlas manualmente. Una variable ya presente en el entorno
# tiene prioridad (load_dotenv no sobrescribe por defecto).
from pathlib import Path as _Path

load_dotenv(_Path(__file__).resolve().parent.parent / ".env")

from src.detection import AnomalyDetector, DetectorConfig, SnapshotStore
from src.dsp import AudioProcessor, DSPConfig
from src.embeddings import MultimodalEncoder
from src.events import EventSegmenter, EventWriter, SegmentedEvent
from src.storage import Database, EventStore, FAISSStore, StorageConfig
from src.storage.models import AnomalyEvent
from src.sync import FrameRingBuffer
from src.vision import FrameCapture, MotionDetector, VisionConfig

logger = logging.getLogger(__name__)

_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_INTERNAL_URL = _API_BASE + "/internal/score"
API_RESET_PENDING_URL = _API_BASE + "/internal/reset-pending"

def _env_flag(name: str, default: bool = False) -> bool:
    """Boolean env var: 1/true/yes (case-insensitive) → True."""
    return os.getenv(name, "true" if default else "false").lower() in (
        "1", "true", "yes",
    )


# Max |audio window midpoint − frame timestamp| for a frame to count as
# simultaneous (F4). Larger gaps yield "no frame" instead of a wrong frame.
FRAME_MAX_DELTA_S = 0.2


def capture_midpoint(now: float, time_info, block_seconds: float) -> float:
    """Wall-clock instant at the middle of an audio block (F4).

    PortAudio reports, in stream time, when the block's first sample hit the
    ADC (``inputBufferAdcTime``) and when the callback was invoked
    (``currentTime``). Their difference is how long ago the block started.
    Some host APIs report zeros; then the block is assumed to have ended at
    ``now``.
    """
    try:
        age = float(time_info.currentTime) - float(time_info.inputBufferAdcTime)
    except (AttributeError, TypeError, ValueError):
        age = -1.0
    if not 0.0 < age < 10.0 * block_seconds:
        age = block_seconds
    return now - age + block_seconds / 2.0


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
        # ADR-0014: components that do not drive the event decision are
        # opt-in. Drift (C2ST + drift-aware refits) is biased by overlapping
        # buffers and works against the anomaly quarantine; snapshots write a
        # model to disk on every refit.
        self._enable_drift = _env_flag("ENABLE_DRIFT")
        self._enable_snapshots = _env_flag("ENABLE_SNAPSHOTS")
        snap_root = os.getenv("SNAPSHOTS_DIR", "data/snapshots")

        def _snapshots(sub: str) -> Optional[SnapshotStore]:
            if not self._enable_snapshots:
                return None
            return SnapshotStore(os.path.join(snap_root, sub))

        audio_cfg = DetectorConfig(
            enable_drift_detection=self._enable_drift,
            enable_drift_aware_refit=self._enable_drift,
        )
        # Inject the real feature-name layout (H6) so drift / explainability
        # labels stay correct regardless of the DSP config.
        self.detector = AnomalyDetector(
            audio_cfg,
            feature_names=self.dsp.feature_names,
            snapshot_store=_snapshots("audio"),
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
        # Items are (samples, capture_midpoint_wallclock) — F4.
        self._audio_queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(
            maxsize=64
        )
        # Ring buffer of timestamped frames for audio-video temporal
        # alignment: each audio window is matched to the nearest frame in
        # time (not simply "the last frame captured").
        self._frame_buffer = FrameRingBuffer(maxlen=64)
        self._running = False

        # Rolling buffer of raw audio samples: keeps the last 3 seconds
        # before an anomaly so saved events are meaningful for playback.
        _sr = DSPConfig().sample_rate
        self._pre_audio_buffer: deque[float] = deque(
            maxlen=3 * _sr
        )  # 48 000 samples @ 16 kHz

        # Fraction of the frame covered by motion boxes (decays when idle)
        self._last_motion_energy: float = 0.0

        # F5: one event per anomaly (segmenter) persisted off the audio loop
        # (writer). Onset = confirming window minus the hysteresis delay.
        _win_s = DSPConfig().window_size / _sr
        self.segmenter = EventSegmenter(
            window_s=_win_s,
            release_s=float(os.getenv("EVENT_RELEASE_S", "2.0")),
            max_duration_s=float(os.getenv("EVENT_MAX_DURATION_S", "60.0")),
            onset_lookback_s=(audio_cfg.hysteresis_count - 1) * _win_s,
        )
        self.event_writer = EventWriter()

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

        block_seconds = dsp_config.window_size / dsp_config.sample_rate

        def _audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            # Timestamp at capture, not at dequeue (F4): the processing loop
            # may lag behind the callback by several windows.
            ts_capture = capture_midpoint(time.time(), time_info, block_seconds)
            if status:
                logger.warning("Audio callback status: %s", status)
            mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            try:
                self._audio_queue.put_nowait((mono.astype(np.float32), ts_capture))
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
            try:
                self.flush_events()
            except Exception as exc:
                logger.warning("Failed to flush pending events: %s", exc)
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
                window, ts_window = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            _window_count += 1
            if _window_count % _reset_check_interval == 0:
                self._check_reset()

            self.process_window(window, ts_window)

    def process_window(self, window: np.ndarray, ts_window: float) -> dict:
        """Process one audio window end-to-end (features → scores → events).

        Separated from the capture loop so it can be driven offline (tests,
        ``scripts/replay.py``) without a microphone. ``ts_window`` is the
        wall-clock midpoint of the window at *capture* (see
        :func:`capture_midpoint`). Returns a summary of the step.
        """
        # Accumulate raw samples for pre-event context before scoring
        self._pre_audio_buffer.extend(window.tolist())

        feature_vec = self.dsp.process_window(window)
        rms = float(np.sqrt(np.mean(window ** 2)))
        result = self.detector.score(feature_vec)

        # Temporal alignment: pick the frame nearest in time to this
        # audio window (frames are already copies — no copy needed here).
        captured = self._frame_buffer.nearest(
            ts_window, max_delta=FRAME_MAX_DELTA_S
        )
        frame = captured.frame if captured is not None else None

        boxes = self.vision.detect(frame) if frame is not None else []

        # --- Explainability (Req 9): top contributing features --------
        mm = {"top_audio_features": self.detector.top_features(feature_vec)}

        # Motion context for the window/event
        if boxes and frame is not None:
            frame_area = float(
                frame.shape[0] * frame.shape[1]
            )
            # Clamp to [0,1]: overlapping boxes can sum past frame_area (M7).
            self._last_motion_energy = min(
                sum(b.area for b in boxes) / frame_area, 1.0
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
        self._notify_api(result, boxes, rms=rms, mm=mm)

        # Event decision: the audio detector (confirmed with hysteresis).
        gate = result.is_anomaly
        # F5: consecutive anomalous windows form ONE event (start/end/peak);
        # it is persisted off the audio loop when it closes.
        event = self.segmenter.update(
            ts_window,
            bool(gate),
            result.anomaly_score,
            window,
            payload={
                "result": result, "frame": frame, "boxes": boxes,
                "motion_energy": self._last_motion_energy, "mm": mm,
            },
            pre_roll=lambda: np.array(self._pre_audio_buffer, dtype=np.float32),
        )
        if event is not None:
            self.event_writer.submit(self._persist_event, event)

        return {"result": result, "boxes": boxes, "mm": mm, "gate": bool(gate)}

    def flush_events(self) -> None:
        """Close any open event and wait until all events are persisted."""
        event = self.segmenter.flush()
        if event is not None:
            self.event_writer.submit(self._persist_event, event)
        self.event_writer.flush()

    def _persist_event(self, event: SegmentedEvent) -> None:
        """Event-writer job: persist a segmented event (runs off the loop)."""
        peak = event.peak
        self._handle_anomaly(
            peak["result"], event.audio, peak["frame"], peak["boxes"],
            motion_energy=peak["motion_energy"], mm=peak["mm"],
            start_ts=event.start_ts, end_ts=event.end_ts,
        )

    def _handle_anomaly(
        self,
        result,
        audio_window: np.ndarray,
        frame: Optional[np.ndarray],
        boxes,
        *,
        motion_energy: float = 0.0,
        mm: Optional[dict] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> None:
        """Persist an anomaly event: filesystem → FAISS → SQLite.

        Parameters
        ----------
        motion_energy : float
            Normalised ratio (0–1) of total bounding-box area to frame area
            at the moment the anomaly was confirmed.  Stored in the event
            metadata for later cross-modal analysis.
        """
        mm = mm or {}
        ts = datetime.fromtimestamp(
            start_ts if start_ts is not None else result.timestamp,
            tz=timezone.utc,
        )
        end = (
            datetime.fromtimestamp(end_ts, tz=timezone.utc)
            if end_ts is not None else None
        )
        duration_s = (
            round(end_ts - start_ts, 3)
            if start_ts is not None and end_ts is not None else None
        )
        boxes_json = [
            {
                "x": b.x, "y": b.y,
                "w": b.w, "h": b.h,
                "source_score": round(b.source_score, 4),
            }
            for b in boxes
        ]
        extra_meta = {
            "window_index": result.window_index,
            "raw_score": result.raw_score,
            "motion_energy": motion_energy,
            "bounding_boxes": boxes_json,
            "end_timestamp": end.isoformat() if end is not None else None,
            "duration_s": duration_s,
        }
        event_dir = self.event_store.save_event(
            timestamp=ts,
            audio=audio_window,
            sample_rate=self.dsp._config.sample_rate,
            frame=frame,
            anomaly_score=result.anomaly_score,
            extra_metadata=extra_meta,
        )
        # H3: store an absolute, CWD-independent path (pipeline and API are
        # separate processes that may run from different directories).
        event_dir = event_dir.resolve()

        embedding = self.encoder.encode(audio_window, frame=frame)
        self.event_store.save_embedding(event_dir, embedding)

        orm_event = AnomalyEvent(
            timestamp=ts,
            end_timestamp=end,
            duration_s=duration_s,
            anomaly_score=result.anomaly_score,
            event_dir=str(event_dir),
            faiss_index_id=None,  # set after FAISS add succeeds
            audio_path=str(event_dir / "audio.wav"),
            frame_path=str(event_dir / "frame.jpg") if frame is not None else None,
            embedding_path=str(event_dir / "embedding.npy"),
            source_region_json=json.dumps(boxes_json),
            # M13: persist the same metadata to the DB column, not just JSON.
            extra_json=json.dumps(extra_meta),
            top_audio_features=json.dumps(mm.get("top_audio_features", [])),
        )
        # C3/L1: persist the row first to get its PK, then index the embedding
        # in FAISS under that *stable* ID. If FAISS fails, the row simply has a
        # NULL faiss_index_id (no orphaned vector, no ID desync).
        event_id = self.db.save_event(orm_event)
        self.faiss_store.add(embedding, faiss_id=event_id)
        self.db.update_faiss_id(event_id, event_id)
        logger.info("Anomaly event saved: %s (score=%.3f)", event_dir, result.anomaly_score)

    def _camera_loop(self) -> None:
        """Background thread: capture timestamped frames into the ring buffer."""
        try:
            self.frame_capture.open()
        except Exception as exc:
            logger.warning("Camera unavailable: %s", exc)
            return
        # A single read error must not kill the thread for the whole process
        # lifetime — catch per iteration and keep going.
        while self._running:
            try:
                frame = self.frame_capture.read()
                if frame is not None:
                    # Store a copy: cv2 may reuse the read buffer (C5).
                    self._frame_buffer.push(frame.copy(), time.time())
            except Exception as exc:
                logger.debug("Camera read failed: %s", exc)
            time.sleep(1.0 / 25)

    def _notify_api(
        self, result, boxes, *, rms: float = 0.0, mm: Optional[dict] = None,
    ) -> None:
        """POST anomaly score to API for WebSocket broadcast (fire-and-forget).

        The payload carries the score, decision and warmup state, the motion
        context (boxes, ``motion_energy``), ``rms``, the top contributing audio
        features and the detector's threshold/drift metrics.
        """
        mm = mm or {}
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
                "top_audio_features": mm.get("top_audio_features", []),
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
