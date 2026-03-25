"""
Test suite for Ensemble Model Conflict Resolution System.

Tests cover:
- Direction extraction (BUY/SELL/HOLD thresholds)
- Conflict detection (models disagreeing on direction)
- Graduated penalty interpolation
- Trade blocking conditions
- Edge cases (zero/identical/extreme scores)
- Config validation
- Graceful error fallback
- Single/dual/triple model scenarios
"""

import logging
import pytest
from typing import Dict, List

from src.scanner.ensemble_conflict import (
    ConflictConfig,
    ConflictResult,
    EnsembleConflictResolver,
    ModelPrediction,
    create_default_conflict_resolver,
)


logger = logging.getLogger(__name__)


class TestModelPrediction:
    """Tests for ModelPrediction dataclass."""

    def test_model_prediction_creation(self):
        """Test basic ModelPrediction instantiation."""
        pred = ModelPrediction(
            model_name="TCN",
            score=0.75,
            direction="BUY",
            confidence=0.92,
        )
        assert pred.model_name == "TCN"
        assert pred.score == 0.75
        assert pred.direction == "BUY"
        assert pred.confidence == 0.92


class TestConflictConfig:
    """Tests for ConflictConfig dataclass."""

    def test_default_config(self):
        """Test default ConflictConfig values."""
        config = ConflictConfig()
        assert config.enabled is True
        assert config.direction_threshold == 0.1
        assert len(config.penalty_curve) == 5
        assert config.penalty_curve[0] == (0.30, -0.08)
        assert config.penalty_curve[-1] == (0.50, -1.00)

    def test_custom_config(self):
        """Test custom ConflictConfig with overrides."""
        custom_curve = [(0.25, -0.05), (0.50, -0.50)]
        config = ConflictConfig(
            enabled=False,
            direction_threshold=0.15,
            penalty_curve=custom_curve,
            block_on_direction_conflict=False,
        )
        assert config.enabled is False
        assert config.direction_threshold == 0.15
        assert config.penalty_curve == custom_curve
        assert config.block_on_direction_conflict is False


class TestEnsembleConflictResolverValidation:
    """Tests for config validation."""

    def test_valid_config_passes(self):
        """Test that valid config passes validation."""
        config = ConflictConfig()
        resolver = EnsembleConflictResolver(config)
        assert resolver.config == config

    def test_invalid_direction_threshold_too_high(self):
        """Test validation rejects direction_threshold > 1.0."""
        config = ConflictConfig(direction_threshold=1.5)
        with pytest.raises(ValueError, match="direction_threshold must be"):
            EnsembleConflictResolver(config)

    def test_invalid_direction_threshold_negative(self):
        """Test validation rejects negative direction_threshold."""
        config = ConflictConfig(direction_threshold=-0.1)
        with pytest.raises(ValueError, match="direction_threshold must be"):
            EnsembleConflictResolver(config)

    def test_invalid_penalty_curve_empty(self):
        """Test validation rejects empty penalty_curve."""
        config = ConflictConfig(penalty_curve=[])
        with pytest.raises(ValueError, match="penalty_curve cannot be empty"):
            EnsembleConflictResolver(config)

    def test_invalid_penalty_out_of_range(self):
        """Test validation rejects penalty outside [-1.0, 0.0]."""
        config = ConflictConfig(penalty_curve=[(0.35, 0.5)])  # positive penalty
        with pytest.raises(ValueError, match="penalty must be"):
            EnsembleConflictResolver(config)

    def test_invalid_penalty_curve_not_increasing(self):
        """Test validation rejects non-increasing disagreement thresholds."""
        config = ConflictConfig(
            penalty_curve=[(0.35, -0.10), (0.30, -0.15)]  # not increasing
        )
        with pytest.raises(ValueError, match="strictly increasing"):
            EnsembleConflictResolver(config)


class TestDirectionExtraction:
    """Tests for direction extraction from prediction scores."""

    def test_direction_buy_above_threshold(self):
        """Test score > threshold → BUY."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.75, direction="BUY", confidence=0.9)
        ]
        directions = resolver._extract_directions(predictions)
        assert directions["TCN"] == "BUY"

    def test_direction_sell_below_threshold(self):
        """Test score < -threshold → SELL."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("Ridge", score=-0.65, direction="SELL", confidence=0.85)
        ]
        directions = resolver._extract_directions(predictions)
        assert directions["Ridge"] == "SELL"

    def test_direction_hold_near_zero(self):
        """Test score near zero (within threshold) → HOLD."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("RF", score=0.05, direction="BUY", confidence=0.6)
        ]
        directions = resolver._extract_directions(predictions)
        assert directions["RF"] == "HOLD"

    def test_direction_hold_exactly_zero(self):
        """Test score = 0.0 → HOLD."""
        resolver = EnsembleConflictResolver()
        predictions = [ModelPrediction("TCN", score=0.0, direction="BUY", confidence=0.5)]
        directions = resolver._extract_directions(predictions)
        assert directions["TCN"] == "HOLD"

    def test_direction_at_threshold_boundary(self):
        """Test score exactly at threshold (should be HOLD, not BUY)."""
        resolver = EnsembleConflictResolver(ConflictConfig(direction_threshold=0.1))
        predictions = [ModelPrediction("TCN", score=0.1, direction="BUY", confidence=0.5)]
        directions = resolver._extract_directions(predictions)
        # At exactly threshold boundary, still HOLD (not > threshold)
        assert directions["TCN"] == "HOLD"

    def test_direction_just_above_threshold(self):
        """Test score just above threshold → BUY."""
        resolver = EnsembleConflictResolver(ConflictConfig(direction_threshold=0.1))
        predictions = [ModelPrediction("TCN", score=0.101, direction="BUY", confidence=0.5)]
        directions = resolver._extract_directions(predictions)
        assert directions["TCN"] == "BUY"

    def test_direction_multiple_models(self):
        """Test extracting directions from multiple models."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.8, direction="BUY", confidence=0.9),
            ModelPrediction("Ridge", score=-0.6, direction="SELL", confidence=0.85),
            ModelPrediction("RF", score=0.02, direction="BUY", confidence=0.7),
        ]
        directions = resolver._extract_directions(predictions)
        assert directions["TCN"] == "BUY"
        assert directions["Ridge"] == "SELL"
        assert directions["RF"] == "HOLD"


class TestDirectionConflictDetection:
    """Tests for direction conflict detection."""

    def test_no_conflict_all_buy(self):
        """Test no conflict when all models agree on BUY."""
        resolver = EnsembleConflictResolver()
        directions = {"TCN": "BUY", "Ridge": "BUY", "RF": "BUY"}
        conflict = resolver._detect_direction_conflict(directions)
        assert conflict is False

    def test_no_conflict_all_sell(self):
        """Test no conflict when all models agree on SELL."""
        resolver = EnsembleConflictResolver()
        directions = {"TCN": "SELL", "Ridge": "SELL", "RF": "SELL"}
        conflict = resolver._detect_direction_conflict(directions)
        assert conflict is False

    def test_no_conflict_all_hold(self):
        """Test no conflict when all models say HOLD."""
        resolver = EnsembleConflictResolver()
        directions = {"TCN": "HOLD", "Ridge": "HOLD", "RF": "HOLD"}
        conflict = resolver._detect_direction_conflict(directions)
        assert conflict is False

    def test_conflict_buy_vs_sell(self):
        """Test conflict detected when one model BUY, another SELL."""
        resolver = EnsembleConflictResolver()
        directions = {"TCN": "BUY", "Ridge": "SELL", "RF": "BUY"}
        conflict = resolver._detect_direction_conflict(directions)
        assert conflict is True

    def test_no_conflict_buy_and_hold(self):
        """Test no conflict when some BUY and some HOLD (no SELL)."""
        resolver = EnsembleConflictResolver()
        directions = {"TCN": "BUY", "Ridge": "HOLD", "RF": "BUY"}
        conflict = resolver._detect_direction_conflict(directions)
        assert conflict is False

    def test_no_conflict_sell_and_hold(self):
        """Test no conflict when some SELL and some HOLD (no BUY)."""
        resolver = EnsembleConflictResolver()
        directions = {"TCN": "SELL", "Ridge": "HOLD", "RF": "SELL"}
        conflict = resolver._detect_direction_conflict(directions)
        assert conflict is False


class TestDisagreementCalculation:
    """Tests for numeric disagreement score calculation."""

    def test_disagreement_identical_scores(self):
        """Test disagreement = 0 when all scores identical."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.5, direction="BUY", confidence=0.9),
            ModelPrediction("Ridge", score=0.5, direction="BUY", confidence=0.85),
            ModelPrediction("RF", score=0.5, direction="BUY", confidence=0.88),
        ]
        disagreement = resolver._calculate_disagreement(predictions)
        assert disagreement == pytest.approx(0.0, abs=0.001)

    def test_disagreement_high_spread(self):
        """Test high disagreement with wide score spread."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.9, direction="BUY", confidence=0.9),
            ModelPrediction("Ridge", score=-0.8, direction="SELL", confidence=0.85),
            ModelPrediction("RF", score=0.1, direction="BUY", confidence=0.7),
        ]
        disagreement = resolver._calculate_disagreement(predictions)
        # Should be > 0.5 due to wide spread
        assert disagreement > 0.5

    def test_disagreement_all_zero(self):
        """Test disagreement with all zero scores."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.0, direction="HOLD", confidence=0.5),
            ModelPrediction("Ridge", score=0.0, direction="HOLD", confidence=0.5),
            ModelPrediction("RF", score=0.0, direction="HOLD", confidence=0.5),
        ]
        disagreement = resolver._calculate_disagreement(predictions)
        assert disagreement == pytest.approx(0.0, abs=0.001)

    def test_disagreement_single_model(self):
        """Test disagreement with only one model (no disagreement possible)."""
        resolver = EnsembleConflictResolver()
        predictions = [ModelPrediction("TCN", score=0.75, direction="BUY", confidence=0.9)]
        disagreement = resolver._calculate_disagreement(predictions)
        assert disagreement == 0.0

    def test_disagreement_empty_list(self):
        """Test disagreement with empty predictions list."""
        resolver = EnsembleConflictResolver()
        disagreement = resolver._calculate_disagreement([])
        assert disagreement == 0.0

    def test_disagreement_two_models(self):
        """Test disagreement with exactly two models."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.6, direction="BUY", confidence=0.9),
            ModelPrediction("Ridge", score=0.2, direction="BUY", confidence=0.6),
        ]
        disagreement = resolver._calculate_disagreement(predictions)
        # mean = 0.4, std = 0.2, disagreement = 0.2 / 0.4 = 0.5
        assert disagreement == pytest.approx(0.5, abs=0.01)

    def test_disagreement_clamped_to_one(self):
        """Test that disagreement is clamped to [0, 1]."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=1.0, direction="BUY", confidence=0.9),
            ModelPrediction("Ridge", score=-1.0, direction="SELL", confidence=0.85),
        ]
        disagreement = resolver._calculate_disagreement(predictions)
        assert 0.0 <= disagreement <= 1.0


class TestPenaltyInterpolation:
    """Tests for graduated penalty curve interpolation."""

    def test_penalty_at_curve_point_35(self):
        """Test penalty at exact curve point (disagreement=0.35)."""
        resolver = EnsembleConflictResolver()
        penalty = resolver._interpolate_penalty(0.35)
        assert penalty == pytest.approx(-0.12, abs=0.001)

    def test_penalty_at_curve_point_40(self):
        """Test penalty at exact curve point (disagreement=0.40)."""
        resolver = EnsembleConflictResolver()
        penalty = resolver._interpolate_penalty(0.40)
        assert penalty == pytest.approx(-0.20, abs=0.001)

    def test_penalty_below_minimum(self):
        """Test penalty below lowest curve point (no penalty)."""
        resolver = EnsembleConflictResolver()
        penalty = resolver._interpolate_penalty(0.20)
        assert penalty == pytest.approx(0.0, abs=0.001)

    def test_penalty_above_maximum(self):
        """Test penalty above highest curve point (max penalty)."""
        resolver = EnsembleConflictResolver()
        penalty = resolver._interpolate_penalty(0.60)
        assert penalty == pytest.approx(-1.0, abs=0.001)

    def test_penalty_interpolation_between_points(self):
        """Test linear interpolation between two curve points."""
        resolver = EnsembleConflictResolver()
        # Between 0.35 (-0.12) and 0.40 (-0.20), at 0.375 should be -0.16
        penalty = resolver._interpolate_penalty(0.375)
        expected = -0.12 + 0.5 * (-0.20 - (-0.12))  # -0.16
        assert penalty == pytest.approx(expected, abs=0.001)

    def test_penalty_all_curve_points(self):
        """Test penalty at all graduated curve points."""
        resolver = EnsembleConflictResolver()
        test_cases = [
            (0.30, -0.08),
            (0.35, -0.12),
            (0.40, -0.20),
            (0.45, -0.30),
            (0.50, -1.00),
        ]
        for disagreement, expected_penalty in test_cases:
            penalty = resolver._interpolate_penalty(disagreement)
            assert penalty == pytest.approx(expected_penalty, abs=0.001), \
                f"At disagreement={disagreement}, expected {expected_penalty}, got {penalty}"

    def test_penalty_clamped_to_valid_range(self):
        """Test that penalty is always in [-1.0, 0.0]."""
        resolver = EnsembleConflictResolver()
        for disagreement in [0.0, 0.25, 0.50, 0.75, 1.0]:
            penalty = resolver._interpolate_penalty(disagreement)
            assert -1.0 <= penalty <= 0.0, \
                f"Penalty {penalty} out of range for disagreement={disagreement}"


class TestShouldBlock:
    """Tests for trade blocking logic."""

    def test_block_on_direction_conflict_enabled(self):
        """Test trade blocked when direction conflict detected and flag enabled."""
        config = ConflictConfig(block_on_direction_conflict=True)
        resolver = EnsembleConflictResolver(config)
        directions = {"TCN": "BUY", "Ridge": "SELL"}  # conflict
        should_block = resolver._should_block(directions, disagreement=0.35)
        assert should_block is True

    def test_no_block_when_direction_conflict_disabled(self):
        """Test trade not blocked if direction conflict flag disabled."""
        config = ConflictConfig(block_on_direction_conflict=False)
        resolver = EnsembleConflictResolver(config)
        directions = {"TCN": "BUY", "Ridge": "SELL"}  # conflict exists
        should_block = resolver._should_block(directions, disagreement=0.35)
        assert should_block is False

    def test_block_on_high_disagreement(self):
        """Test trade blocked when disagreement >= 0.50."""
        config = ConflictConfig(
            block_on_direction_conflict=False,
            block_on_high_disagreement=True,
        )
        resolver = EnsembleConflictResolver(config)
        directions = {"TCN": "BUY", "Ridge": "BUY"}  # no conflict
        should_block = resolver._should_block(directions, disagreement=0.50)
        assert should_block is True

    def test_no_block_below_disagreement_threshold(self):
        """Test trade not blocked when disagreement < 0.50."""
        config = ConflictConfig(block_on_high_disagreement=True)
        resolver = EnsembleConflictResolver(config)
        directions = {"TCN": "BUY", "Ridge": "BUY"}
        should_block = resolver._should_block(directions, disagreement=0.45)
        assert should_block is False

    def test_no_block_all_hold(self):
        """Test no block when all models HOLD."""
        resolver = EnsembleConflictResolver()
        directions = {"TCN": "HOLD", "Ridge": "HOLD"}
        should_block = resolver._should_block(directions, disagreement=0.40)
        assert should_block is False

    def test_no_block_buy_and_hold_only(self):
        """Test no block when only BUY and HOLD (no SELL)."""
        resolver = EnsembleConflictResolver()
        directions = {"TCN": "BUY", "Ridge": "HOLD"}
        should_block = resolver._should_block(directions, disagreement=0.45)
        assert should_block is False


class TestDirectionVoting:
    """Tests for direction voting and consensus."""

    def test_count_direction_votes_all_buy(self):
        """Test vote counting when all models BUY."""
        resolver = EnsembleConflictResolver()
        directions = {"TCN": "BUY", "Ridge": "BUY", "RF": "BUY"}
        votes = resolver._count_direction_votes(directions)
        assert votes == {"BUY": 3, "SELL": 0, "HOLD": 0}

    def test_count_direction_votes_mixed(self):
        """Test vote counting with mixed directions."""
        resolver = EnsembleConflictResolver()
        directions = {"TCN": "BUY", "Ridge": "SELL", "RF": "HOLD"}
        votes = resolver._count_direction_votes(directions)
        assert votes == {"BUY": 1, "SELL": 1, "HOLD": 1}

    def test_consensus_direction_buy_majority(self):
        """Test consensus direction when BUY has majority."""
        resolver = EnsembleConflictResolver()
        votes = {"BUY": 2, "SELL": 1, "HOLD": 0}
        consensus = resolver._consensus_direction(votes)
        assert consensus == "BUY"

    def test_consensus_direction_sell_majority(self):
        """Test consensus direction when SELL has majority."""
        resolver = EnsembleConflictResolver()
        votes = {"BUY": 0, "SELL": 2, "HOLD": 1}
        consensus = resolver._consensus_direction(votes)
        assert consensus == "SELL"

    def test_consensus_direction_buy_tie_with_sell(self):
        """Test consensus direction: BUY wins tie vs SELL."""
        resolver = EnsembleConflictResolver()
        votes = {"BUY": 1, "SELL": 1, "HOLD": 1}
        consensus = resolver._consensus_direction(votes)
        assert consensus == "BUY"  # BUY priority in tie

    def test_consensus_direction_all_hold(self):
        """Test consensus direction when all votes are HOLD."""
        resolver = EnsembleConflictResolver()
        votes = {"BUY": 0, "SELL": 0, "HOLD": 3}
        consensus = resolver._consensus_direction(votes)
        assert consensus == "HOLD"


class TestResolveIntegration:
    """Integration tests for the full resolve() method."""

    def test_resolve_all_models_agree_buy(self):
        """Test resolve() when all models strongly agree on BUY."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.8, direction="BUY", confidence=0.92),
            ModelPrediction("Ridge", score=0.75, direction="BUY", confidence=0.88),
            ModelPrediction("RF", score=0.82, direction="BUY", confidence=0.90),
        ]
        result = resolver.resolve(predictions)
        assert result.has_direction_conflict is False
        assert result.penalty == pytest.approx(0.0, abs=0.001)
        assert result.should_block is False
        assert result.consensus_direction == "BUY"

    def test_resolve_direction_conflict_buy_vs_sell(self):
        """Test resolve() with clear direction conflict."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.7, direction="BUY", confidence=0.90),
            ModelPrediction("Ridge", score=-0.6, direction="SELL", confidence=0.85),
            ModelPrediction("RF", score=0.05, direction="BUY", confidence=0.65),
        ]
        result = resolver.resolve(predictions)
        assert result.has_direction_conflict is True
        assert result.should_block is True  # 2+ active directions
        assert result.penalty == -1.0

    def test_resolve_moderate_disagreement(self):
        """Test resolve() with moderate disagreement (penalty applied)."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.6, direction="BUY", confidence=0.9),
            ModelPrediction("Ridge", score=0.2, direction="BUY", confidence=0.7),
            ModelPrediction("RF", score=0.3, direction="BUY", confidence=0.75),
        ]
        result = resolver.resolve(predictions)
        assert result.has_direction_conflict is False
        assert result.should_block is False
        assert result.penalty < 0.0  # Should have penalty
        assert result.consensus_direction == "BUY"

    def test_resolve_with_disabled_resolver(self):
        """Test resolve() returns no penalty when resolver disabled."""
        config = ConflictConfig(enabled=False)
        resolver = EnsembleConflictResolver(config)
        predictions = [
            ModelPrediction("TCN", score=-0.8, direction="SELL", confidence=0.92),
            ModelPrediction("Ridge", score=0.7, direction="BUY", confidence=0.88),
        ]
        result = resolver.resolve(predictions)
        assert result.penalty == 0.0
        assert result.should_block is False

    def test_resolve_empty_predictions(self):
        """Test resolve() with empty predictions list."""
        resolver = EnsembleConflictResolver()
        result = resolver.resolve([])
        assert result.penalty == 0.0
        assert result.has_direction_conflict is False

    def test_resolve_single_model(self):
        """Test resolve() with single model (no conflict possible)."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.75, direction="BUY", confidence=0.92)
        ]
        result = resolver.resolve(predictions)
        assert result.has_direction_conflict is False
        assert result.should_block is False
        assert result.disagreement_score == 0.0

    def test_resolve_two_models_agree(self):
        """Test resolve() with two models in agreement."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.8, direction="BUY", confidence=0.92),
            ModelPrediction("Ridge", score=0.7, direction="BUY", confidence=0.88),
        ]
        result = resolver.resolve(predictions)
        assert result.has_direction_conflict is False
        assert result.should_block is False
        assert result.direction_votes["BUY"] == 2

    def test_resolve_all_hold(self):
        """Test resolve() when all models HOLD."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.05, direction="HOLD", confidence=0.6),
            ModelPrediction("Ridge", score=-0.02, direction="HOLD", confidence=0.55),
            ModelPrediction("RF", score=0.03, direction="HOLD", confidence=0.58),
        ]
        result = resolver.resolve(predictions)
        assert result.consensus_direction == "HOLD"
        assert result.has_direction_conflict is False


class TestErrorHandling:
    """Tests for error handling and graceful fallback."""

    def test_resolve_graceful_fallback_on_error(self):
        """Test graceful fallback to -15% penalty on exception.

        Note: This test verifies that resolve() catches exceptions during
        analysis and returns a safe fallback. We can't easily trigger an error
        with normal inputs, but the try/except is in place for robustness.
        """
        resolver = EnsembleConflictResolver()
        # Monkey-patch to force an error during resolution
        original_method = resolver._calculate_disagreement

        def broken_disagreement(*args, **kwargs):
            raise RuntimeError("Simulated error in disagreement calculation")

        resolver._calculate_disagreement = broken_disagreement

        predictions = [
            ModelPrediction("TCN", score=0.7, direction="BUY", confidence=0.9)
        ]
        result = resolver.resolve(predictions)

        # Should return fallback penalty
        assert result.penalty == pytest.approx(-0.15, abs=0.001)
        assert result.should_block is False
        assert "fallback_on_error" in result.details.get("reason", "")

    def test_resolve_handles_invalid_direction_gracefully(self):
        """Test resolve() handles invalid direction strings."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.7, direction="INVALID", confidence=0.9),
        ]
        # Should not crash
        result = resolver.resolve(predictions)
        assert isinstance(result, ConflictResult)


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_default_conflict_resolver(self):
        """Test factory function creates resolver with defaults."""
        resolver = create_default_conflict_resolver()
        assert isinstance(resolver, EnsembleConflictResolver)
        assert resolver.config.enabled is True
        assert resolver.config.direction_threshold == 0.1
        assert len(resolver.config.penalty_curve) == 5

    def test_default_resolver_accepts_predictions(self):
        """Test default resolver can process predictions."""
        resolver = create_default_conflict_resolver()
        predictions = [
            ModelPrediction("TCN", score=0.8, direction="BUY", confidence=0.92),
            ModelPrediction("Ridge", score=0.75, direction="BUY", confidence=0.88),
        ]
        result = resolver.resolve(predictions)
        assert isinstance(result, ConflictResult)
        assert result.has_direction_conflict is False


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_extreme_positive_scores(self):
        """Test with maximum positive scores."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=1.0, direction="BUY", confidence=1.0),
            ModelPrediction("Ridge", score=1.0, direction="BUY", confidence=1.0),
        ]
        result = resolver.resolve(predictions)
        assert result.disagreement_score == 0.0
        assert result.has_direction_conflict is False

    def test_extreme_negative_scores(self):
        """Test with minimum negative scores."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=-1.0, direction="SELL", confidence=1.0),
            ModelPrediction("Ridge", score=-1.0, direction="SELL", confidence=1.0),
        ]
        result = resolver.resolve(predictions)
        assert result.disagreement_score == 0.0
        assert result.consensus_direction == "SELL"

    def test_mixed_extreme_scores(self):
        """Test with extreme positive and negative scores (maximum disagreement)."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=1.0, direction="BUY", confidence=0.95),
            ModelPrediction("Ridge", score=-1.0, direction="SELL", confidence=0.95),
        ]
        result = resolver.resolve(predictions)
        assert result.has_direction_conflict is True
        assert result.disagreement_score > 0.7
        assert result.should_block is True

    def test_confidence_zero(self):
        """Test model with zero confidence."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.8, direction="BUY", confidence=0.0),
            ModelPrediction("Ridge", score=0.75, direction="BUY", confidence=0.9),
        ]
        result = resolver.resolve(predictions)
        # Should process normally despite zero confidence
        assert isinstance(result, ConflictResult)

    def test_very_small_score_differences(self):
        """Test with scores that are very close together."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.5001, direction="BUY", confidence=0.9),
            ModelPrediction("Ridge", score=0.5002, direction="BUY", confidence=0.9),
            ModelPrediction("RF", score=0.5003, direction="BUY", confidence=0.9),
        ]
        result = resolver.resolve(predictions)
        assert result.disagreement_score < 0.01
        assert result.has_direction_conflict is False

    def test_many_models(self):
        """Test with more than 3 models."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction(f"Model{i}", score=0.7 + i*0.01, direction="BUY", confidence=0.85)
            for i in range(10)
        ]
        result = resolver.resolve(predictions)
        assert isinstance(result, ConflictResult)
        assert result.direction_votes["BUY"] == 10

    def test_custom_direction_threshold(self):
        """Test with custom direction threshold."""
        config = ConflictConfig(direction_threshold=0.5)
        resolver = EnsembleConflictResolver(config)
        predictions = [
            ModelPrediction("TCN", score=0.3, direction="BUY", confidence=0.9)  # Would be BUY at 0.1 threshold
        ]
        directions = resolver._extract_directions(predictions)
        assert directions["TCN"] == "HOLD"  # Not BUY at 0.5 threshold


class TestResultMetadata:
    """Tests for result metadata and logging."""

    def test_result_details_contain_predictions(self):
        """Test that result.details contains model predictions."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.8, direction="BUY", confidence=0.92),
        ]
        result = resolver.resolve(predictions)
        assert "model_predictions" in result.details
        assert len(result.details["model_predictions"]) == 1
        assert result.details["model_predictions"][0]["model"] == "TCN"

    def test_result_contains_direction_votes(self):
        """Test that result contains direction vote counts."""
        resolver = EnsembleConflictResolver()
        predictions = [
            ModelPrediction("TCN", score=0.8, direction="BUY", confidence=0.92),
            ModelPrediction("Ridge", score=-0.6, direction="SELL", confidence=0.88),
        ]
        result = resolver.resolve(predictions)
        assert "BUY" in result.direction_votes
        assert "SELL" in result.direction_votes
        assert result.direction_votes["BUY"] == 1
        assert result.direction_votes["SELL"] == 1


class TestConfigDisabled:
    """Tests for disabled resolver behavior."""

    def test_disabled_resolver_ignores_conflicts(self):
        """Test that disabled resolver returns zero penalty regardless of conflicts."""
        config = ConflictConfig(enabled=False)
        resolver = EnsembleConflictResolver(config)
        predictions = [
            ModelPrediction("TCN", score=0.9, direction="BUY", confidence=0.95),
            ModelPrediction("Ridge", score=-0.9, direction="SELL", confidence=0.95),
        ]
        result = resolver.resolve(predictions)
        assert result.penalty == 0.0
        assert result.should_block is False
        assert result.has_direction_conflict is False

    def test_disabled_resolver_still_runs_analysis(self):
        """Test that disabled resolver completes without error."""
        config = ConflictConfig(enabled=False)
        resolver = EnsembleConflictResolver(config)
        predictions = [
            ModelPrediction("TCN", score=0.5, direction="BUY", confidence=0.9),
        ]
        result = resolver.resolve(predictions)
        assert isinstance(result, ConflictResult)
        assert result.penalty == 0.0
