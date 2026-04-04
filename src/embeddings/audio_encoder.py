from __future__ import annotations

import numpy as np
import torch

from .config import EmbeddingConfig


class AudioEncoder:
    """Encodes audio signals to 768-dim L2-normalized embeddings via Wav2Vec 2.0.

    Model is loaded lazily on the first call to encode(). The loading strategy
    tries ``local_files_only=True`` first (fast path for offline use), then falls
    back to downloading on the initial run.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self._config = config or EmbeddingConfig()
        self._model = None
        self._processor = None

    def encode(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Encode an audio signal to a 768-dim L2-normalized embedding.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 audio samples.
        sample_rate : int
            Sample rate of the input. Resampled to 16000 Hz if different.

        Returns
        -------
        np.ndarray
            Shape (768,), float32, L2-normalized.
        """
        if self._model is None:
            self._load_model()

        # Resample if needed
        if sample_rate != self._config.sample_rate:
            import torchaudio
            waveform = torch.tensor(audio).unsqueeze(0)
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=self._config.sample_rate
            )
            audio = waveform.squeeze(0).numpy()

        audio_float = audio.astype(np.float32)
        inputs = self._processor(
            audio_float,
            sampling_rate=self._config.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        # Mean-pool over time dimension: (1, T, 768) → (768,)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
        return self._l2_normalize(embedding)

    def ensure_downloaded(self) -> None:
        """Force model download/cache verification. Call once at startup."""
        if self._model is None:
            self._load_model()

    def _load_model(self) -> None:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        model_id = self._config.wav2vec2_model_id
        # Try offline-first; download on first run if cache is empty
        try:
            self._processor = Wav2Vec2Processor.from_pretrained(
                model_id, local_files_only=True
            )
            self._model = Wav2Vec2Model.from_pretrained(
                model_id, local_files_only=True
            )
        except EnvironmentError:
            self._processor = Wav2Vec2Processor.from_pretrained(model_id)
            self._model = Wav2Vec2Model.from_pretrained(model_id)

        self._model.eval()
        self._model.to(self._config.device)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)
