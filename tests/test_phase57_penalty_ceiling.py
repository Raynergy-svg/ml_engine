"""Tests for Phase 57 US-352: ConfidencePenaltyCeiling.

Covers:
  - Basic subtraction ceiling (normal case — penalty applies fully)
  - Subtraction ceiling activates when confidence near floor
  - Subtraction capped so confidence doesn't go below floor
  - At-floor pair receives zero subtraction penalty
  - Multiplier ceiling — normal case multiplier applies
  - Multiplier ceiling activates when projected confidence < floor
  - Safe multiplier computed so confidence lands exactly at floor
  - Confidence already at/below floor — multiplier held at 1.0
  - get_stats tracks total checks and activations
  - reset_stats clears counters
"""

import pytest

from src.scanner.confidence_penalty_ceiling import (
    ConfidencePenaltyCeiling,
    CeilingResult,
    DEFAULT_CEILING_FLOOR,
)


def make_ceiling(floor: float = 40.0) -> ConfidencePenaltyCeiling:
    return ConfidencePenaltyCeiling(ceiling_floor=floor, ceiling_floor_norm=floor / 100.0)


# ---------------------------------------------------------------------------
# 1. Additive subtraction checks
# ---------------------------------------------------------------------------

class TestCheckSubtraction:

    def test_normal_penalty_applies_fully(self):
        ceiling = make_ceiling(floor=40.0)
        result = ceiling.check_subtraction(current_confidence=70.0, proposed_sub=3.0)
        assert result.applied_penalty == pytest.approx(3.0)
        assert result.ceiling_active is False
        assert result.confidence_after == pytest.approx(67.0)

    def test_ceiling_activates_when_near_floor(self):
        """confidence=42, proposed_sub=3 → can only subtract 2 before hitting floor=40."""
        ceiling = make_ceiling(floor=40.0)
        result = ceiling.check_subtraction(current_confidence=42.0, proposed_sub=3.0)
        assert result.ceiling_active is True
        assert result.applied_penalty == pytest.approx(2.0)
        assert result.confidence_after == pytest.approx(40.0)

    def test_at_floor_pair_gets_zero_subtraction(self):
        """Confidence already at floor → headroom=0 → zero penalty."""
        ceiling = make_ceiling(floor=40.0)
        result = ceiling.check_subtraction(current_confidence=40.0, proposed_sub=3.0)
        assert result.applied_penalty == pytest.approx(0.0)
        assert result.ceiling_active is True
        assert result.confidence_after == pytest.approx(40.0)

    def test_below_floor_pair_gets_zero_subtraction(self):
        """Confidence below floor (already suppressed by another system) → no further penalty."""
        ceiling = make_ceiling(floor=40.0)
        result = ceiling.check_subtraction(current_confidence=35.0, proposed_sub=3.0)
        assert result.applied_penalty == pytest.approx(0.0)
        assert result.confidence_after == pytest.approx(35.0)

    def test_custom_floor_respected(self):
        ceiling = make_ceiling(floor=50.0)
        result = ceiling.check_subtraction(current_confidence=52.0, proposed_sub=5.0)
        assert result.ceiling_active is True
        assert result.applied_penalty == pytest.approx(2.0)

    def test_zero_proposed_penalty_is_noop(self):
        ceiling = make_ceiling()
        result = ceiling.check_subtraction(current_confidence=60.0, proposed_sub=0.0)
        assert result.applied_penalty == pytest.approx(0.0)
        assert result.ceiling_active is False


# ---------------------------------------------------------------------------
# 2. Multiplicative checks
# ---------------------------------------------------------------------------

class TestCheckMultiplier:

    def test_normal_multiplier_applies_fully(self):
        """confidence=70, mult=0.90 → 63 > floor=40 → mult unchanged."""
        ceiling = make_ceiling(floor=40.0)
        result = ceiling.check_multiplier(current_confidence=70.0, proposed_mult=0.90)
        assert result.applied_penalty == pytest.approx(0.90)
        assert result.ceiling_active is False
        assert result.confidence_after == pytest.approx(63.0)

    def test_multiplier_ceiling_activates(self):
        """confidence=45, mult=0.85 → 38.25 < floor=40 → ceiling fires."""
        ceiling = make_ceiling(floor=40.0)
        result = ceiling.check_multiplier(current_confidence=45.0, proposed_mult=0.85)
        assert result.ceiling_active is True
        # Safe multiplier = 40/45 ≈ 0.8889
        assert result.applied_penalty == pytest.approx(40.0 / 45.0, abs=0.001)
        assert result.confidence_after == pytest.approx(40.0, abs=0.01)

    def test_confidence_at_floor_mult_held_at_1(self):
        """Confidence already at floor → no further multiplier reduction."""
        ceiling = make_ceiling(floor=40.0)
        result = ceiling.check_multiplier(current_confidence=40.0, proposed_mult=0.85)
        # confidence at floor → applied multiplier = 1.0 (no change)
        assert result.applied_penalty == pytest.approx(1.0)

    def test_normalised_scale_multiplier(self):
        """0-1 scale: conf=0.45, mult=0.85, floor_norm=0.40 → ceiling fires."""
        ceiling = ConfidencePenaltyCeiling(ceiling_floor=40.0, ceiling_floor_norm=0.40)
        result = ceiling.check_multiplier(
            current_confidence=0.45, proposed_mult=0.85, scale="0-1"
        )
        assert result.ceiling_active is True
        assert result.confidence_after == pytest.approx(0.40, abs=0.001)


# ---------------------------------------------------------------------------
# 3. Stats tracking
# ---------------------------------------------------------------------------

class TestStats:

    def test_stats_track_total_checks(self):
        ceiling = make_ceiling()
        ceiling.check_subtraction(70.0, 3.0)
        ceiling.check_subtraction(42.0, 3.0)
        stats = ceiling.get_stats()
        assert stats["total_checks"] == 2

    def test_stats_track_activations(self):
        ceiling = make_ceiling()
        ceiling.check_subtraction(70.0, 3.0)   # No ceiling
        ceiling.check_subtraction(41.0, 5.0)   # Ceiling fires
        stats = ceiling.get_stats()
        assert stats["ceiling_activations"] == 1
        assert stats["activation_rate"] == pytest.approx(0.5, abs=0.01)

    def test_reset_stats_clears_counters(self):
        ceiling = make_ceiling()
        ceiling.check_subtraction(41.0, 5.0)
        ceiling.reset_stats()
        stats = ceiling.get_stats()
        assert stats["total_checks"] == 0
        assert stats["ceiling_activations"] == 0
