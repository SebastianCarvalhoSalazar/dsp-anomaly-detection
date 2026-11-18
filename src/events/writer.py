"""Background writer for anomaly events (plan F5).

Persisting an event (WAV + JPEG + SQLite + Wav2Vec2/DINOv2 embedding + FAISS)
takes from hundreds of milliseconds to seconds on CPU. Running it inside the
audio loop stalled processing and filled the capture queue, dropping audio.
``EventWriter`` runs those jobs on a single worker thread fed by a bounded
queue; the audio loop only enqueues.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class EventWriter:
    """Single-thread job runner with a bounded queue.

    Jobs run in submission order. If the queue is full the job is dropped and
    counted in :attr:`dropped` (the audio loop must never block on I/O).
    """

    def __init__(self, maxsize: int = 32) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.failed = 0
        self._thread = threading.Thread(
            target=self._run, name="event-writer", daemon=True
        )
        self._thread.start()

    def submit(self, fn: Callable, *args, **kwargs) -> bool:
        """Enqueue a job without blocking. Returns False if it was dropped."""
        try:
            self._q.put_nowait((fn, args, kwargs))
            return True
        except queue.Full:
            self.dropped += 1
            logger.warning("Event writer queue full; event dropped.")
            return False

    def flush(self) -> None:
        """Block until every submitted job has finished."""
        self._q.join()

    def pending(self) -> int:
        return self._q.unfinished_tasks

    def _run(self) -> None:
        while True:
            fn, args, kwargs = self._q.get()
            try:
                fn(*args, **kwargs)
            except Exception:  # keep the worker alive
                self.failed += 1
                logger.exception("Event writer job failed")
            finally:
                self._q.task_done()
