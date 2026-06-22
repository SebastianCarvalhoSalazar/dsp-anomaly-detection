"""Tests for percentile-based score calibration (Fase 2)."""

from src.fusion import PercentileCalibrator


def test_passthrough_during_warmup():
    cal = PercentileCalibrator(window=500, min_samples=50)
    # Fewer than min_samples → returns the (clamped) raw score.
    assert cal.calibrate(0.42) == 0.42
    assert cal.calibrate(2.0) == 1.0
    assert cal.calibrate(-1.0) == 0.0


def test_percentile_rank_after_history():
    cal = PercentileCalibrator(window=500, min_samples=10)
    for v in [i / 100 for i in range(100)]:  # 0.00 .. 0.99
        cal.update(v)
    # ~half the history is below 0.5.
    assert 0.45 <= cal.calibrate(0.5) <= 0.55
    # A value above everything seen → rank ~1.0.
    assert cal.calibrate(5.0) == 1.0
    # Below everything → rank ~0.0.
    assert cal.calibrate(-5.0) == 0.0


def test_calibrate_and_update_records_history():
    cal = PercentileCalibrator(min_samples=1)
    cal.calibrate_and_update(0.1)
    cal.calibrate_and_update(0.2)
    assert len(cal) == 2


def test_dynamic_update_shifts_distribution():
    """After the distribution shifts up, an old 'high' value calibrates low."""
    cal = PercentileCalibrator(window=100, min_samples=10)
    for _ in range(100):
        cal.update(0.1)
    high_before = cal.calibrate(0.5)  # 0.5 >> history of 0.1 → ~1.0
    for _ in range(100):
        cal.update(0.9)  # window rolls over to high values
    high_after = cal.calibrate(0.5)  # now 0.5 is low relative to 0.9
    assert high_before > high_after
