from dataclasses import dataclass


@dataclass
class StorageConfig:
    """Configuration for all storage backends."""

    events_dir: str = "eventos"
    db_path: str = "data/events.db"
    faiss_path: str = "data/faiss.index"
    # Wav2Vec2 768-dim + DINOv2 768-dim concatenated and L2-normalized
    embedding_dim: int = 1536
