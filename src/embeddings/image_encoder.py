from __future__ import annotations

import numpy as np
import torch

from .config import EmbeddingConfig


class ImageEncoder:
    """Encodes image frames to 768-dim L2-normalized embeddings via DINOv2.

    Uses the CLS token output from the last hidden state. Input frames are
    expected in BGR format (OpenCV convention); conversion to RGB is applied
    internally before passing to the HuggingFace processor.

    Model is loaded lazily on the first call to encode().
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self._config = config or EmbeddingConfig()
        self._model = None
        self._processor = None

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """Encode a BGR or RGB image frame to a 768-dim L2-normalized embedding.

        Parameters
        ----------
        frame : np.ndarray
            Shape (H, W, 3), uint8. Expected BGR (OpenCV) — converted to RGB
            internally.

        Returns
        -------
        np.ndarray
            Shape (768,), float32, L2-normalized.
        """
        if self._model is None:
            self._load_model()

        # OpenCV produces BGR; DINOv2 processor expects RGB.
        # HuggingFace AutoImageProcessor accepts numpy arrays directly,
        # avoiding the PIL dependency that causes segfaults on macOS with torch+OpenCV.
        rgb_frame = frame[:, :, ::-1].astype(np.uint8)

        inputs = self._processor(images=rgb_frame, return_tensors="pt")
        inputs = {k: v.to(self._config.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
        # CLS token is the first token of last_hidden_state: (1, T, 768) → (768,)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
        return self._l2_normalize(embedding)

    def ensure_downloaded(self) -> None:
        """Force model download/cache verification. Call once at startup."""
        if self._model is None:
            self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        model_id = self._config.dinov2_model_id
        try:
            self._processor = AutoImageProcessor.from_pretrained(
                model_id, local_files_only=True
            )
            self._model = AutoModel.from_pretrained(
                model_id, local_files_only=True
            )
        except EnvironmentError:
            self._processor = AutoImageProcessor.from_pretrained(model_id)
            self._model = AutoModel.from_pretrained(model_id)

        self._model.eval()
        self._model.to(self._config.device)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)
