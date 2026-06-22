from __future__ import annotations

import os
import threading
from contextlib import contextmanager

import faiss
import numpy as np

from .config import StorageConfig

try:  # POSIX advisory file locking; absent on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


class FAISSStore:
    """FAISS-based vector store for similarity search.

    Uses ``IndexIDMap2`` over ``IndexFlatIP`` (inner product). Vectors are
    L2-normalized before insertion and query, making inner product equivalent
    to cosine similarity.

    IDs are **explicit and caller-supplied** (the SQLite event PK), never
    positional — so they stay stable across deletions and index rebuilds
    (fix C3: positional IDs desynced from SQLite after any delete/clear).
    Vectors can be removed by ID, avoiding orphaned entries.

    Cross-process safety (fix C4): the pipeline and API are separate processes
    that share the index file. Persistence is **atomic** (write to a temp file
    then ``os.replace``) and guarded by a POSIX file lock, so a reader never
    sees a truncated index and two writers cannot interleave.

    The in-process ``threading.Lock`` guards the in-memory index; the file
    lock guards the on-disk file across processes.
    """

    def __init__(self, config: StorageConfig | None = None) -> None:
        self._config = config or StorageConfig()
        self._dim = self._config.embedding_dim
        self._lock = threading.Lock()
        self._index: faiss.IndexIDMap2 | None = None

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def _new_index(self) -> faiss.IndexIDMap2:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self._dim))

    def init(self) -> None:
        """Load existing index from disk or create a new one."""
        path = self._config.faiss_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._lock:
            self._index = self._read_or_new(path)

    def reload(self) -> None:
        """Re-read the index from disk, picking up vectors added by other
        processes. The API must call this before searching to see vectors the
        pipeline added."""
        with self._lock:
            self._index = self._read_or_new(self._config.faiss_path)

    def _read_or_new(self, path: str) -> faiss.IndexIDMap2:
        """Read the persisted index under the file lock, or create a new one."""
        with self._interprocess_lock():
            if os.path.exists(path):
                return faiss.read_index(path)
            return self._new_index()

    # ------------------------------------------------------------------ #
    #  Mutations                                                          #
    # ------------------------------------------------------------------ #

    def add(self, embedding: np.ndarray, faiss_id: int) -> int:
        """L2-normalize and add a vector under an explicit, stable ID.

        Parameters
        ----------
        embedding : np.ndarray
            Shape (embedding_dim,), float32.
        faiss_id : int
            Stable ID for the vector (the SQLite event PK). Must be unique.

        Returns
        -------
        int
            ``faiss_id`` (echoed back for caller convenience).
        """
        vec = self._normalize(embedding).reshape(1, -1).astype(np.float32)
        if vec.shape[1] != self._dim:
            raise ValueError(
                f"embedding dim {vec.shape[1]} != index dim {self._dim}"
            )
        ids = np.array([int(faiss_id)], dtype=np.int64)
        with self._lock:
            self._index.add_with_ids(vec, ids)
            self._persist()
        return int(faiss_id)

    def remove(self, faiss_id: int) -> int:
        """Remove the vector with the given ID. Returns the count removed (0/1)."""
        ids = np.array([int(faiss_id)], dtype=np.int64)
        with self._lock:
            removed = int(self._index.remove_ids(ids))
            if removed:
                self._persist()
        return removed

    def clear(self) -> None:
        """Reset the index to empty and overwrite the persisted file."""
        with self._lock:
            self._index = self._new_index()
            self._persist()

    # ------------------------------------------------------------------ #
    #  Queries                                                            #
    # ------------------------------------------------------------------ #

    def search(self, query: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Search for the k nearest vectors by cosine similarity.

        Returns ``(distances, ids)`` each of shape (1, k); ``ids`` are the
        explicit IDs supplied at insertion (the SQLite PKs). Returns empty
        arrays when the index is empty.
        """
        vec = self._normalize(query).reshape(1, -1).astype(np.float32)
        if vec.shape[1] != self._dim:
            raise ValueError(
                f"query dim {vec.shape[1]} != index dim {self._dim}"
            )
        with self._lock:
            if self._index.ntotal == 0:
                return (
                    np.empty((1, 0), dtype=np.float32),
                    np.empty((1, 0), dtype=np.int64),
                )
            k_actual = min(k, self._index.ntotal)
            distances, ids = self._index.search(vec, k_actual)
        return distances, ids

    def get_total(self) -> int:
        """Return the number of vectors currently in the index."""
        with self._lock:
            return self._index.ntotal

    # ------------------------------------------------------------------ #
    #  Internal                                                           #
    # ------------------------------------------------------------------ #

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        v = vec.astype(np.float32)
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return v
        return v / norm

    @contextmanager
    def _interprocess_lock(self):
        """Exclusive cross-process lock around index file read/write.

        No-op where ``fcntl`` is unavailable (non-POSIX); atomic ``os.replace``
        still prevents torn reads there.
        """
        if fcntl is None:
            yield
            return
        lock_path = self._config.faiss_path + ".lock"
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        f = open(lock_path, "w")  # noqa: SIM115
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()

    def _persist(self) -> None:
        # Caller must hold self._lock. Atomic write under the file lock so a
        # concurrent reader never sees a partially-written index.
        path = self._config.faiss_path
        tmp = f"{path}.tmp.{os.getpid()}"
        with self._interprocess_lock():
            faiss.write_index(self._index, tmp)
            os.replace(tmp, path)
