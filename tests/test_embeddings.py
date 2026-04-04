"""Tests for embedding encoders — all models are mocked; no real model downloads."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.embeddings.audio_encoder import AudioEncoder
from src.embeddings.config import EmbeddingConfig
from src.embeddings.encoder import MultimodalEncoder, _build_multimodal
from src.embeddings.image_encoder import ImageEncoder


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_fake_tensor(dim: int = 768, batch: int = 1, time: int = 10):
    """Return a fake last_hidden_state tensor shaped (batch, time, dim)."""
    t = torch.randn(batch, time, dim)
    return MagicMock(last_hidden_state=t)


def _make_audio(n: int = 16000) -> np.ndarray:
    return np.random.randn(n).astype(np.float32)


def _make_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# AudioEncoder
# ---------------------------------------------------------------------------

class TestAudioEncoder:
    def _patched_encoder(self):
        enc = AudioEncoder()
        fake_model = MagicMock(return_value=_make_fake_tensor())
        fake_processor = MagicMock(return_value={"input_values": torch.zeros(1, 16000)})
        enc._model = fake_model
        enc._processor = fake_processor
        return enc

    def test_encode_shape(self):
        enc = self._patched_encoder()
        out = enc.encode(_make_audio())
        assert out.shape == (768,)

    def test_encode_dtype(self):
        enc = self._patched_encoder()
        out = enc.encode(_make_audio())
        assert out.dtype == np.float32

    def test_encode_l2_normalized(self):
        enc = self._patched_encoder()
        out = enc.encode(_make_audio())
        assert abs(np.linalg.norm(out) - 1.0) < 1e-5

    def test_lazy_load_model_is_none_before_encode(self):
        enc = AudioEncoder()
        assert enc._model is None

    def test_model_loaded_after_encode(self):
        enc = self._patched_encoder()
        # _model already set (simulates post-load state)
        enc.encode(_make_audio())
        assert enc._model is not None

    def test_load_model_called_once(self):
        enc = AudioEncoder()
        call_count = []

        def fake_load():
            call_count.append(1)
            enc._model = MagicMock(return_value=_make_fake_tensor())
            enc._processor = MagicMock(return_value={"input_values": torch.zeros(1, 16000)})

        enc._load_model = fake_load
        enc.encode(_make_audio())
        enc.encode(_make_audio())
        assert len(call_count) == 1  # loaded only once


# ---------------------------------------------------------------------------
# ImageEncoder
# ---------------------------------------------------------------------------

class TestImageEncoder:
    def _patched_encoder(self):
        enc = ImageEncoder()
        # last_hidden_state shape: (1, T, 768) — CLS token is index 0
        fake_hs = torch.randn(1, 197, 768)  # 197 = 196 patches + 1 CLS
        fake_model = MagicMock(return_value=MagicMock(last_hidden_state=fake_hs))
        fake_processor = MagicMock(return_value={"pixel_values": torch.zeros(1, 3, 224, 224)})
        enc._model = fake_model
        enc._processor = fake_processor
        return enc

    def test_encode_shape(self):
        enc = self._patched_encoder()
        out = enc.encode(_make_frame())
        assert out.shape == (768,)

    def test_encode_dtype(self):
        enc = self._patched_encoder()
        out = enc.encode(_make_frame())
        assert out.dtype == np.float32

    def test_encode_l2_normalized(self):
        enc = self._patched_encoder()
        out = enc.encode(_make_frame())
        assert abs(np.linalg.norm(out) - 1.0) < 1e-5

    def test_lazy_load_model_is_none_before_encode(self):
        enc = ImageEncoder()
        assert enc._model is None


# ---------------------------------------------------------------------------
# MultimodalEncoder
# ---------------------------------------------------------------------------

class TestMultimodalEncoder:
    def _patched_encoder(self):
        enc = MultimodalEncoder()
        # Patch sub-encoders to return fixed normalized vectors
        audio_vec = np.ones(768, dtype=np.float32)
        audio_vec /= np.linalg.norm(audio_vec)
        image_vec = np.ones(768, dtype=np.float32) * 0.5
        image_vec /= np.linalg.norm(image_vec)
        enc.audio_encoder.encode = MagicMock(return_value=audio_vec)
        enc.image_encoder.encode = MagicMock(return_value=image_vec)
        return enc

    def test_encode_shape_with_frame(self):
        enc = self._patched_encoder()
        out = enc.encode(_make_audio(), frame=_make_frame())
        assert out.shape == (1536,)

    def test_encode_shape_without_frame(self):
        enc = self._patched_encoder()
        out = enc.encode(_make_audio(), frame=None)
        assert out.shape == (1536,)

    def test_encode_l2_normalized_with_frame(self):
        enc = self._patched_encoder()
        out = enc.encode(_make_audio(), frame=_make_frame())
        assert abs(np.linalg.norm(out) - 1.0) < 1e-5

    def test_encode_l2_normalized_without_frame(self):
        enc = self._patched_encoder()
        out = enc.encode(_make_audio(), frame=None)
        assert abs(np.linalg.norm(out) - 1.0) < 1e-5

    def test_no_frame_zeros_image_half_before_normalization(self):
        """When frame=None, image_encoder.encode must not be called."""
        enc = MultimodalEncoder()
        audio_vec = np.ones(768, dtype=np.float32) / np.sqrt(768)
        enc.audio_encoder.encode = MagicMock(return_value=audio_vec)
        enc.image_encoder.encode = MagicMock()

        out = enc.encode(_make_audio(), frame=None)

        enc.image_encoder.encode.assert_not_called()
        # Image half of the raw (pre-normalization) concat was zeros
        # After normalization the first 768 values should be non-zero
        # and the vector must be unit length
        assert abs(np.linalg.norm(out) - 1.0) < 1e-5

    def test_image_encoder_called_when_frame_provided(self):
        enc = self._patched_encoder()
        frame = _make_frame()
        enc.encode(_make_audio(), frame=frame)
        enc.image_encoder.encode.assert_called_once()


# ---------------------------------------------------------------------------
# _build_multimodal helper
# ---------------------------------------------------------------------------

def test_build_multimodal_shape():
    a = np.random.randn(768).astype(np.float32)
    b = np.random.randn(768).astype(np.float32)
    out = _build_multimodal(a, b)
    assert out.shape == (1536,)


def test_build_multimodal_l2_normalized():
    a = np.ones(768, dtype=np.float32)
    b = np.ones(768, dtype=np.float32)
    out = _build_multimodal(a, b)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-5


def test_build_multimodal_zero_vector():
    a = np.zeros(768, dtype=np.float32)
    b = np.zeros(768, dtype=np.float32)
    out = _build_multimodal(a, b)
    assert out.shape == (1536,)
    assert not np.any(np.isnan(out))
