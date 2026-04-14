"""Phase 90 US-402: Tests for confidence_penalty_ceiling.py verification."""
from __future__ import annotations

import pytest

from src.scanner.confidence_penalty_ceiling import (
    DEFAULT_CEILING_FLOOR,
    DEFAULT_CEILING_FLOOR_NORM,
    ConfidencePenaltyCeiling,
)


def test_default_ceiling_floor_is_040():
    """Regression guard: DEFAULT_CEILING_FLOOR must be 0.40, not 40.0."""
    assert DEFAULT_CEILING_FLOOR == pytest.approx(0.40)


def test_default_ceiling_floor_norm_is_040():
    """DEFAULT_CEILING_FLOOR_NORM must also be 0.40."""
    assert DEFAULT_CEILING_FLOOR_NORM == pytest.approx(0.40)


# ── check_subtraction ──────────────────────────────────────────────────────


def test_subtraction_ceiling_active_when_near_floor():
    """Confidence near floor → ceiling clips penalty → ceiling_active=True."""
    ceiling = ConfidencePenaltyCeiling(ceiling_floor=0.40)
    result = ceiling.check_subtraction(current_confidence=0.41, proposed_sub=0.05)
    # Only 0.01 of headroom before floor
    assert result.applied_penalty <= 0.01 + 1e-9
    assert result.ceiling_active is True


def test_subtraction_no_ceiling_when_plenty_of_headroom():
    """Confidence well above floor → full penalty applied → ceiling_active=False."""
    ceiling = ConfidencePenaltyCeiling(ceiling_floor=0.40)
    result = ceiling.check_subtraction(current_confidence=0.70, proposed_sub=0.03)
    assert result.applied_penalty == pytest.approx(0.03)
    assert result.ceiling_active is False


def test_subtraction_confidence_after_stays_at_or_above_floor():
    """After applying penalty, confidence_after must not drop below ceiling_floor."""
    ceiling = ConfidencePenaltyCeiling(ceiling_floor=0.40)
    result = ceiling.check_subtraction(current_confidence=0.43, proposed_sub=0.10)
    assert result.confidence_after >= 0.40 - 1e-9


def test_subtraction_full_clip_at_floor():
    """If already at floor, applied_penalty should be 0."""
    ceiling = ConfidencePenaltyCeiling(ceiling_floor=0.40)
    result = ceiling.check_subtraction(current_confidence=0.40, proposed_sub=0.05)
    assert result.applied_penalty == pytest.approx(0.0)
    assert result.ceiling_active is True


# ── check_multiplier ───────────────────────────────────────────────────────


def test_multiplier_ceiling_active_when_near_floor():
    """Multiplier would push confidence below floor → ceiling active."""
    ceiling = ConfidencePenaltyCeiling(ceiling_floor=0.40)
    result = ceiling.check_multiplier(current_confidence=0.41, proposed_mult=0.85, scale="0-100")
    # 0.41 * 0.85 = 0.3485 < 0.40 → ceiling fires
    assert result.ceiling_active is True
    assert result.confidence_after >= 0.40 - 1e-9


def test_multiplier_no_ceiling_when_safe():
    """Multiplier keeps confidence above floor → ceiling_active=False."""
    ceiling = ConfidencePenaltyCeiling(ceiling_floor=0.40)
    result = ceiling.check_multiplier(current_confidence=0.80, proposed_mult=0.85, scale="0-100")
    # 0.80 * 0.85 = 0.68 > 0.40 → safe
    assert result.ceiling_active is False
    assert result.applied_penalty == pytest.approx(0.85)


def test_multiplier_0to1_scale_ceiling_active():
    """check_multiplier with scale='0-1' uses ceiling_floor_norm."""
    ceiling = ConfidencePenaltyCeiling(ceiling_floor=0.40, ceiling_floor_norm=0.40)
    result = ceiling.check_multiplier(current_confidence=0.41, proposed_mult=0.85, scale="0-1")
    assert result.ceiling_active is True
    assert result.confidence_after >= 0.40 - 1e-9


def test_multiplier_0to1_scale_no_ceiling():
    """check_multiplier with scale='0-1' and confidence well above floor."""
    ceiling = ConfidencePenaltyCeiling(ceiling_floor_norm=0.40)
    result = ceiling.check_multiplier(current_confidence=0.80, proposed_mult=0.85, scale="0-1")
    assert result.ceiling_active is False


# ── Stats tracking ─────────────────────────────────────────────────────────


def test_stats_track_ceiling_activations():
    """Ceiling activation counter increments correctly."""
    ceiling = ConfidencePenaltyCeiling(ceiling_floor=0.40)
    ceiling.check_subtraction(current_confidence=0.41, proposed_sub=0.05)  # activates
    ceiling.check_subtraction(current_confidence=0.70, proposed_sub=0.03)  # no activation
    stats = ceiling.get_stats()
    assert stats["total_checks"] == 2
    assert stats["ceiling_activations"] == 1
