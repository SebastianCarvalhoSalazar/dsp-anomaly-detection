"""Tests for the fusion strategies (Fase 2)."""

import pytest

from src.fusion import (
    AndStrategy,
    Maximum,
    OrStrategy,
    WeightedAverage,
    make_strategy,
)


# --------------------------------------------------------------------------- #
# Weighted average
# --------------------------------------------------------------------------- #

def test_weighted_average_combines_and_normalizes_weights():
    s = WeightedAverage(audio_weight=0.75)
    assert s.video_weight == pytest.approx(0.25)
    r = s.combine(audio=0.8, video=0.4)
    assert r.combined_score == pytest.approx(0.75 * 0.8 + 0.25 * 0.4)


def test_weighted_average_clamps_weight():
    s = WeightedAverage(audio_weight=1.5)
    assert s.audio_weight == 1.0 and s.video_weight == 0.0


def test_weighted_average_anomaly_threshold():
    s = WeightedAverage(audio_weight=0.5, threshold=0.9)
    assert s.combine(0.95, 0.95).is_anomaly
    assert not s.combine(0.5, 0.5).is_anomaly


# --------------------------------------------------------------------------- #
# Maximum
# --------------------------------------------------------------------------- #

def test_maximum_takes_higher_score():
    r = Maximum(threshold=0.9).combine(0.3, 0.95)
    assert r.combined_score == 0.95
    assert r.is_anomaly


# --------------------------------------------------------------------------- #
# AND / OR
# --------------------------------------------------------------------------- #

def test_and_requires_both_above_threshold():
    s = AndStrategy(threshold=0.9)
    assert s.combine(0.95, 0.95).is_anomaly
    assert not s.combine(0.95, 0.5).is_anomaly
    assert not s.combine(0.5, 0.95).is_anomaly
    # combined magnitude is the weaker (binding) score
    assert s.combine(0.95, 0.92).combined_score == pytest.approx(0.92)


def test_or_requires_either_above_threshold():
    s = OrStrategy(threshold=0.9)
    assert s.combine(0.95, 0.1).is_anomaly
    assert s.combine(0.1, 0.95).is_anomaly
    assert not s.combine(0.5, 0.5).is_anomaly


# --------------------------------------------------------------------------- #
# Dominant modality
# --------------------------------------------------------------------------- #

def test_dominant_modality_labels():
    s = Maximum()
    assert s.combine(0.9, 0.2).dominant_modality == "audio-driven"
    assert s.combine(0.2, 0.9).dominant_modality == "video-driven"
    assert s.combine(0.8, 0.82).dominant_modality == "multimodal"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_make_strategy_by_name():
    assert isinstance(make_strategy("weighted", audio_weight=0.3), WeightedAverage)
    assert isinstance(make_strategy("max"), Maximum)
    assert isinstance(make_strategy("and"), AndStrategy)
    assert isinstance(make_strategy("or"), OrStrategy)


def test_make_strategy_unknown_raises():
    with pytest.raises(ValueError):
        make_strategy("bogus")
