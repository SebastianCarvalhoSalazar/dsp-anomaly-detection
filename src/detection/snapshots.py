"""Model snapshots for auditability, debugging and reproducibility (Req 8).

On every refit a detector can persist the fitted PCA + IsolationForest plus
metadata (timestamp, drift AUC, buffer statistics, refit reason). A bounded
retention policy keeps only the most recent ``max_snapshots`` so disk usage
stays predictable. Saving happens off the scoring hot-path (after the model
swap, outside the detector lock).
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Optional


class SnapshotStore:
    """Bounded, append-only store of fitted-model snapshots.

    Layout::

        <directory>/
          snapshot_000001/
            model.pkl        (dict: {"model": IsolationForest, "pca": PCA|None})
            metadata.json
          snapshot_000002/
          ...
    """

    _NAME_RE = re.compile(r"snapshot_(\d+)$")

    def __init__(self, directory: str, max_snapshots: int = 10) -> None:
        self._dir = Path(directory)
        self._max = max_snapshots

    def _existing(self) -> list[Path]:
        if not self._dir.exists():
            return []
        snaps = [p for p in self._dir.iterdir() if self._NAME_RE.search(p.name)]
        return sorted(snaps, key=lambda p: int(self._NAME_RE.search(p.name).group(1)))

    def _next_index(self) -> int:
        existing = self._existing()
        if not existing:
            return 1
        return int(self._NAME_RE.search(existing[-1].name).group(1)) + 1

    def save(self, *, model, pca, metadata: dict) -> Path:
        """Persist a snapshot and prune old ones beyond the retention limit."""
        self._dir.mkdir(parents=True, exist_ok=True)
        idx = self._next_index()
        snap_dir = self._dir / f"snapshot_{idx:06d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        with open(snap_dir / "model.pkl", "wb") as f:
            pickle.dump({"model": model, "pca": pca}, f)
        (snap_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        self._prune()
        return snap_dir

    def _prune(self) -> None:
        snaps = self._existing()
        excess = len(snaps) - self._max
        if excess <= 0:
            return
        for old in snaps[:excess]:
            for child in old.iterdir():
                child.unlink()
            old.rmdir()

    def list_snapshots(self) -> list[Path]:
        return self._existing()

    def load_metadata(self, snap_dir: Path) -> dict:
        return json.loads((snap_dir / "metadata.json").read_text())

    def latest(self) -> Optional[Path]:
        existing = self._existing()
        return existing[-1] if existing else None
