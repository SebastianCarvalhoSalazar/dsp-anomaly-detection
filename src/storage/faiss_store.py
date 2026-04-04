from __future__ import annotations

import os
import threading

import faiss
import numpy as np

from .config import StorageConfig


class FAISSStore:
    """FAISS-based vector store for similarity search.

    Uses IndexFlatIP (inner product). Vectors are L2-normalized before
    insertion and query, making inner product equivalent to cosine similarity.

    The index is persisted to disk after every add() call.
    All public methods are thread-safe.
    """

    def __init__(self, config: StorageConfig | None = None) -> None:
        self._config = config or StorageConfig()
        self._dim = self._config.embedding_dim
        self._lock = threading.Lock()
        self._index: faiss.IndexFlatIP | None = None

    def init(self) -> None:
        """Load existing index from disk or create a new one."""
        path = self._config.faiss_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._lock:
            if os.path.exists(path):
                self._index = faiss.read_index(path)
            else:
                self._index = faiss.IndexFlatIP(self._dim)

    def add(self, embedding: np.ndarray) -> int:
        """L2-normalize and add a vector to the index.

        Parameters
        ----------
        embedding : np.ndarray
            Shape (embedding_dim,), float32.

        Returns
        -------
        int
            The FAISS sequential integer ID (index.ntotal - 1 after add).
        """
        vec = self._normalize(embedding).reshape(1, -1).astype(np.float32)
        with self._lock:
            self._index.add(vec)
            idx = self._index.ntotal - 1
            self._persist()
        return idx

    def search(self, query: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Search for the k nearest vectors by cosine similarity.

        Parameters
        ----------
        query : np.ndarray
            Shape (embedding_dim,), float32.
        k : int
            Number of results.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (distances, ids) each of shape (1, k). Returns empty arrays if
            the index has fewer than k vectors.
        """
        with self._lock:
            if self._index.ntotal == 0:
                return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)
            k_actual = min(k, self._index.ntotal)
            vec = self._normalize(query).reshape(1, -1).astype(np.float32)
            distances, ids = self._index.search(vec, k_actual)
        return distances, ids

    def get_total(self) -> int:
        """Return the number of vectors currently in the index."""
        with self._lock:
            return self._index.ntotal

    def reload(self) -> None:
        """Re-read the index from disk, picking up vectors added by other processes.

        The pipeline and API run as separate processes and each hold their own
        in-memory FAISSStore. The pipeline writes to disk after every add(); the
        API must call reload() before searching to see those new vectors.
        """
        path = self._config.faiss_path
        with self._lock:
            if os.path.exists(path):
                self._index = faiss.read_index(path)
            else:
                self._index = faiss.IndexFlatIP(self._dim)

    def clear(self) -> None:
        """Reset the index to empty and overwrite the persisted file."""
        with self._lock:
            self._index = faiss.IndexFlatIP(self._dim)
            self._persist()

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        v = vec.astype(np.float32)
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return v
        return v / norm

    def _persist(self) -> None:
        # Caller must hold self._lock
        faiss.write_index(self._index, self._config.faiss_path)
