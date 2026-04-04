from __future__ import annotations

from typing import Optional

import numpy as np

from .audio_encoder import AudioEncoder
from .config import EmbeddingConfig
from .image_encoder import ImageEncoder


def _build_multimodal(audio_emb: np.ndarray, image_emb: np.ndarray) -> np.ndarray:
    """Concatenate two 768-dim embeddings and L2-normalize the result."""
    combined = np.concatenate([audio_emb, image_emb]).astype(np.float32)
    norm = np.linalg.norm(combined)
    if norm < 1e-8:
        return combined
    return combined / norm


class MultimodalEncoder:
    """Produces 1536-dim multimodal embeddings from audio and image inputs.

    Both modalities are encoded independently (Wav2Vec 2.0 for audio,
    DINOv2 for image) and the resulting 768-dim L2-normalized vectors are
    concatenated and re-normalized to unit length.

    If no frame is provided, the image half is zero-padded before concatenation
    and final normalization.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self._config = config or EmbeddingConfig()
        self.audio_encoder = AudioEncoder(self._config)
        self.image_encoder = ImageEncoder(self._config)

    def encode(
        self,
        audio: np.ndarray,
        frame: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Produce a 1536-dim multimodal embedding.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 audio samples.
        frame : np.ndarray or None
            BGR image frame (H, W, 3), uint8. If None, the image half of the
            embedding is zero-padded.
        sample_rate : int
            Sample rate of the audio input.

        Returns
        -------
        np.ndarray
            Shape (1536,), float32, L2-normalized.
        """
        audio_emb = self.audio_encoder.encode(audio, sample_rate=sample_rate)

        if frame is not None:
            image_emb = self.image_encoder.encode(frame)
        else:
            image_emb = np.zeros(self._config.image_embedding_dim, dtype=np.float32)

        return _build_multimodal(audio_emb, image_emb)

    def ensure_downloaded(self) -> None:
        """Download both models if not already cached."""
        self.audio_encoder.ensure_downloaded()
        self.image_encoder.ensure_downloaded()
