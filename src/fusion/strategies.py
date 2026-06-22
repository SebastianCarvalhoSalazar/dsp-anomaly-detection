"""Configurable score-fusion strategies (Strategy pattern, ADR-0005).

Each strategy combines two **calibrated** scores (audio, video) in [0, 1]
into a ``FusionResult``. A single ``threshold`` on the calibrated scale
decides whether the fused result is anomalous, so all strategies share a
consistent decision rule. New strategies (e.g. learned fusion) only need to
implement :meth:`FusionStrategy.combine`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Calibrated scores within this band of each other are deemed equally
# responsible → the event is "multimodal" rather than driven by one modality.
_DOMINANCE_EPS = 0.1


@dataclass
class FusionResult:
    combined_score: float
    is_anomaly: bool
    dominant_modality: str  # "audio-driven" | "video-driven" | "multimodal"


def _dominant(audio: float, video: float) -> str:
    if abs(audio - video) <= _DOMINANCE_EPS:
        return "multimodal"
    return "audio-driven" if audio > video else "video-driven"


class FusionStrategy(ABC):
    """Combine two calibrated scores into a fused decision."""

    name: str = "base"

    def __init__(self, threshold: float = 0.9) -> None:
        self.threshold = threshold

    @abstractmethod
    def combine(self, audio: float, video: float) -> FusionResult: ...


class WeightedAverage(FusionStrategy):
    """``audio_weight * audio + video_weight * video`` (weights sum to 1)."""

    name = "weighted"

    def __init__(self, audio_weight: float = 0.5, threshold: float = 0.9) -> None:
        super().__init__(threshold)
        audio_weight = float(min(max(audio_weight, 0.0), 1.0))
        self.audio_weight = audio_weight
        self.video_weight = 1.0 - audio_weight

    def combine(self, audio: float, video: float) -> FusionResult:
        combined = self.audio_weight * audio + self.video_weight * video
        return FusionResult(combined, combined >= self.threshold, _dominant(audio, video))


class Maximum(FusionStrategy):
    """``max(audio, video)`` — fires if either modality is anomalous."""

    name = "max"

    def combine(self, audio: float, video: float) -> FusionResult:
        combined = max(audio, video)
        return FusionResult(combined, combined >= self.threshold, _dominant(audio, video))


class AndStrategy(FusionStrategy):
    """Anomaly only when BOTH modalities exceed the threshold."""

    name = "and"

    def combine(self, audio: float, video: float) -> FusionResult:
        is_anom = audio >= self.threshold and video >= self.threshold
        # The combined magnitude is the weaker of the two (the binding one).
        return FusionResult(min(audio, video), is_anom, _dominant(audio, video))


class OrStrategy(FusionStrategy):
    """Anomaly when EITHER modality exceeds the threshold."""

    name = "or"

    def combine(self, audio: float, video: float) -> FusionResult:
        is_anom = audio >= self.threshold or video >= self.threshold
        return FusionResult(max(audio, video), is_anom, _dominant(audio, video))


_REGISTRY: dict[str, type[FusionStrategy]] = {
    WeightedAverage.name: WeightedAverage,
    Maximum.name: Maximum,
    AndStrategy.name: AndStrategy,
    OrStrategy.name: OrStrategy,
}


def make_strategy(name: str, **kwargs) -> FusionStrategy:
    """Build a strategy by name (``weighted`` | ``max`` | ``and`` | ``or``)."""
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown fusion strategy '{name}'. "
            f"Available: {sorted(_REGISTRY)}"
        ) from exc
    return cls(**kwargs)
