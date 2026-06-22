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

from src.detection import AnomalyDetector, DetectorConfig, SnapshotStore
from src.dsp import AudioProcessor, DSPConfig
from src.embeddings import MultimodalEncoder
from src.fusion import PercentileCalibrator, WeightedAverage
from src.storage import Database, EventStore, FAISSStore, StorageConfig
from src.storage.models import AnomalyEvent
from src.sync import FrameRingBuffer
from src.vision import FrameCapture, MotionDetector, VisionConfig
from src.vision_detection import VideoAnomalyDetector, VideoFeatureExtractor

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
        # Model snapshots for auditability (Req 8) + drift-aware refits (Req 7).
        snap_root = os.getenv("SNAPSHOTS_DIR", "data/snapshots")
        audio_cfg = DetectorConfig(enable_drift_aware_refit=True)
        # Inject the real feature-name layout (H6) so drift / explainability
        # labels stay correct regardless of the DSP config.
        self.detector = AnomalyDetector(
            audio_cfg,
            feature_names=self.dsp.feature_names,
            snapshot_store=SnapshotStore(os.path.join(snap_root, "audio")),
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

        # --- Multimodal detection (v0.3) ---------------------------------
        # Independent video detector + per-modality score calibration +
        # configurable late fusion. The final event-gating decision still
        # follows the audio detector (the fast, established path); the fused
        # combined_score is calibrated, exposed and stored for observability.
        self.video_detector = VideoAnomalyDetector(
            snapshot_store=SnapshotStore(os.path.join(snap_root, "video")),
        )
        if self.video_detector.load_state():
            logger.info("Loaded video detector state from disk.")
        self.video_extractor = VideoFeatureExtractor()
        self.audio_calibrator = PercentileCalibrator()
        self.video_calibrator = PercentileCalibrator()
        self.fusion_strategy = WeightedAverage(audio_weight=0.5)

        # Dual horizon (Req 6): an optional "slow" model per modality with a
        # large buffer reflects long-term behaviour. The final decision still
        # uses the fast model; slow scores are exposed for observability.
        # Opt-in (extra compute) via ENABLE_SLOW_MODELS.
        self._enable_slow = os.getenv(
            "ENABLE_SLOW_MODELS", "false"
        ).lower() in ("1", "true", "yes")
        self.slow_detector: Optional[AnomalyDetector] = None
        self.slow_video_detector: Optional[VideoAnomalyDetector] = None
        if self._enable_slow:
            self.slow_detector = AnomalyDetector(
                DetectorConfig(
                    buffer_size=5000, refit_every=2000,
                    enable_drift_detection=False,
                ),
                feature_names=self.dsp.feature_names,
            )
            self.slow_video_detector = VideoAnomalyDetector(
                DetectorConfig(
                    buffer_size=3000, refit_every=1500,
                    enable_pca=False, enable_drift_detection=False,
                    state_path="data/slow_video_detector_state.pkl",
                )
            )
            logger.info("Dual-horizon slow models enabled.")

        # Queue for passing audio windows from the sounddevice callback
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
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
                self.video_detector.save_state()
                if self._enable_slow:
                    self.slow_detector.save_state()
                    self.slow_video_detector.save_state()
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
                self.video_detector.reset()
                if self._enable_slow:
                    self.slow_detector.reset()
                    self.slow_video_detector.reset()
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

            # Timestamp the audio window at dequeue time so it can be matched
            # to the nearest captured frame.
            ts_window = time.time()

            _window_count += 1
            if _window_count % _reset_check_interval == 0:
                self._check_reset()

            # Accumulate raw samples for pre-event context before scoring
            self._pre_audio_buffer.extend(window.tolist())

            feature_vec = self.dsp.process_window(window)
            rms = float(np.sqrt(np.mean(window ** 2)))
            result = self.detector.score(feature_vec)

            # Temporal alignment: pick the frame nearest in time to this
            # audio window (frames are already copies — no copy needed here).
            captured = self._frame_buffer.nearest(ts_window)
            frame = captured.frame if captured is not None else None

            boxes = self.vision.detect(frame) if frame is not None else []

            # --- Video modality -------------------------------------------
            # Extract motion features while boxes still carry their IoU
            # temporal weights (the source-ranking below overwrites them).
            video_array = None
            if frame is not None:
                video_fv = self.video_extractor.extract(boxes, frame.shape)
                video_array = video_fv.to_array()
                video_result = self.video_detector.score(video_array)
                video_raw = video_result.anomaly_score
            else:
                video_raw = 0.0

            # --- Dual horizon (Req 6): slow models, decision uses fast ----
            slow_audio = 0.0
            slow_video = 0.0
            if self._enable_slow:
                slow_audio = self.slow_detector.score(feature_vec).anomaly_score
                if video_array is not None:
                    slow_video = self.slow_video_detector.score(
                        video_array
                    ).anomaly_score

            # --- Calibration + fusion -------------------------------------
            audio_cal = self.audio_calibrator.calibrate_and_update(
                result.anomaly_score
            )
            video_cal = self.video_calibrator.calibrate_and_update(video_raw)
            fusion = self.fusion_strategy.combine(audio_cal, video_cal)

            # --- Explainability (Req 9): top contributing features --------
            top_audio = self.detector.top_features(feature_vec)
            top_video = (
                self.video_detector.top_features(video_array)
                if video_array is not None else []
            )

            mm = {
                "audio_score": audio_cal,
                "video_score": video_cal,
                "combined_score": fusion.combined_score,
                "dominant_modality": fusion.dominant_modality,
                "fast_audio_score": result.anomaly_score,
                "slow_audio_score": slow_audio,
                "fast_video_score": video_raw,
                "slow_video_score": slow_video,
                "top_audio_features": top_audio,
                "top_video_features": top_video,
            }

            # Cross-modal correlation (3.2)
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

            # Event gating still follows the audio detector (fast path).
            if result.is_anomaly:
                # Snapshot the pre-event buffer (last 3s) as the event audio
                audio_clip = np.array(list(self._pre_audio_buffer), dtype=np.float32)
                self._handle_anomaly(
                    result, audio_clip, frame, boxes,
                    motion_energy=self._last_motion_energy, mm=mm,
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
        ts = datetime.fromtimestamp(result.timestamp, tz=timezone.utc)
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
        }
        event_dir = self.event_store.save_event(
            timestamp=ts,
            audio=audio_window,
            sample_rate=self.dsp._config.sample_rate,
            frame=frame,
            anomaly_score=result.anomaly_score,
            extra_metadata=extra_meta,
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
            source_region_json=json.dumps(boxes_json),
            # M13: persist the same metadata to the DB column, not just JSON.
            extra_json=json.dumps(extra_meta),
            # Calibrated per-modality + fused scores (v0.3).
            audio_score=mm.get("audio_score", 0.0),
            video_score=mm.get("video_score", 0.0),
            combined_score=mm.get("combined_score", 0.0),
            dominant_modality=mm.get("dominant_modality", "audio"),
            top_audio_features=json.dumps(mm.get("top_audio_features", [])),
            top_video_features=json.dumps(mm.get("top_video_features", [])),
        )
        self.db.save_event(orm_event)
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

        The payload includes ``motion_energy`` and ``rms`` plus the calibrated
        per-modality, fused, fast/slow and explainability fields so the
        dashboard can correlate signals and recompute fusion strategies live.
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
                # Calibrated per-modality + fused + fast/slow + explainability.
                "audio_score": round(mm.get("audio_score", 0.0), 6),
                "video_score": round(mm.get("video_score", 0.0), 6),
                "combined_score": round(mm.get("combined_score", 0.0), 6),
                "dominant_modality": mm.get("dominant_modality", "audio"),
                "fast_audio_score": round(mm.get("fast_audio_score", 0.0), 6),
                "slow_audio_score": round(mm.get("slow_audio_score", 0.0), 6),
                "fast_video_score": round(mm.get("fast_video_score", 0.0), 6),
                "slow_video_score": round(mm.get("slow_video_score", 0.0), 6),
                "top_audio_features": mm.get("top_audio_features", []),
                "top_video_features": mm.get("top_video_features", []),
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
