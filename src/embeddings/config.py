from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    """Configuration for multimodal embedding encoders."""

    wav2vec2_model_id: str = "facebook/wav2vec2-base"
    dinov2_model_id: str = "facebook/dinov2-base"
    audio_embedding_dim: int = 768
    image_embedding_dim: int = 768
    multimodal_dim: int = 1536   # audio_dim + image_dim
    sample_rate: int = 16000
    # Try local cache first (offline); falls back to download on first run
    local_files_only: bool = False
    device: str = "cpu"
