"""
Unit tests for regime_gates.py module.

Tests cover:
- RegimeGateProfile validation
- RegimeGateConfig initialization and defaults
- get_regime_profile() for all 4 regimes
- apply_regime_gates() threshold adaptation
- Graceful fallback to static thresholds
- Edge cases: None regimes, unknown regimes, boundary values
"""

import pytest

from src.scanner.regime_gates import (
    REGIME_EXTREME,
    REGIME_HIGH,
    REGIME_LOW,
    REGIME_NORMAL,
    VALID_REGIMES,
    RegimeGateConfig,
    RegimeGateProfile,
    apply_regime_gates,
    get_regime_profile,
)


class TestRegimeGateProfile:
    """Test RegimeGateProfile dataclass and validation."""

    def test_profile_creation_valid(self):
        """Test creating a valid profile."""
        profile = RegimeGateProfile(
            regime_name="NORMAL",
            confidence_threshold=50.0,
            momentum_threshold=0.35,
            risk_threshold=0.80,
            min_rr_ratio=1.2,
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Standard thresholds",
        )
        assert profile.regime_name == "NORMAL"
        assert profile.confidence_threshold == 50.0
        assert profile.position_size_multiplier == 1.0

    def test_profile_validate_invalid_regime_name(self):
        """Test validation rejects invalid regime name."""
        profile = RegimeGateProfile(
            regime_name="INVALID",
            confidence_threshold=50.0,
            momentum_threshold=0.35,
            risk_threshold=0.80,
            min_rr_ratio=1.2,
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Invalid",
        )
        with pytest.raises(ValueError, match="regime_name"):
            profile.validate()

    def test_profile_validate_confidence_out_of_range(self):
        """Test validation rejects confidence outside 0-100."""
        profile = RegimeGateProfile(
            regime_name="NORMAL",
            confidence_threshold=150.0,  # Invalid
            momentum_threshold=0.35,
            risk_threshold=0.80,
            min_rr_ratio=1.2,
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Invalid",
        )
        with pytest.raises(ValueError, match="confidence_threshold"):
            profile.validate()

    def test_profile_validate_momentum_out_of_range(self):
        """Test validation rejects momentum outside 0-1."""
        profile = RegimeGateProfile(
            regime_name="NORMAL",
            confidence_threshold=50.0,
            momentum_threshold=1.5,  # Invalid
            risk_threshold=0.80,
            min_rr_ratio=1.2,
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Invalid",
        )
        with pytest.raises(ValueError, match="momentum_threshold"):
            profile.validate()

    def test_profile_validate_risk_out_of_range(self):
        """Test validation rejects risk outside 0-1."""
        profile = RegimeGateProfile(
            regime_name="NORMAL",
            confidence_threshold=50.0,
            momentum_threshold=0.35,
            risk_threshold=1.5,  # Invalid
            min_rr_ratio=1.2,
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Invalid",
        )
        with pytest.raises(ValueError, match="risk_threshold"):
            profile.validate()

    def test_profile_validate_min_rr_ratio_too_low(self):
        """Test validation rejects R:R ratio below 0.5."""
        profile = RegimeGateProfile(
            regime_name="NORMAL",
            confidence_threshold=50.0,
            momentum_threshold=0.35,
            risk_threshold=0.80,
            min_rr_ratio=0.3,  # Invalid
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Invalid",
        )
        with pytest.raises(ValueError, match="min_rr_ratio"):
            profile.validate()

    def test_profile_validate_position_size_multiplier_invalid(self):
        """Test validation rejects position_size_multiplier <= 0."""
        profile = RegimeGateProfile(
            regime_name="NORMAL",
            confidence_threshold=50.0,
            momentum_threshold=0.35,
            risk_threshold=0.80,
            min_rr_ratio=1.2,
            position_size_multiplier=0.0,  # Invalid
            require_momentum_alignment=False,
            description="Invalid",
        )
        with pytest.raises(ValueError, match="position_size_multiplier"):
            profile.validate()

    def test_profile_validate_confidence_boundary_0(self):
        """Test validation accepts confidence = 0."""
        profile = RegimeGateProfile(
            regime_name="NORMAL",
            confidence_threshold=0.0,
            momentum_threshold=0.35,
            risk_threshold=0.80,
            min_rr_ratio=1.2,
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Edge case",
        )
        profile.validate()  # Should not raise

    def test_profile_validate_confidence_boundary_100(self):
        """Test validation accepts confidence = 100."""
        profile = RegimeGateProfile(
            regime_name="NORMAL",
            confidence_threshold=100.0,
            momentum_threshold=0.35,
            risk_threshold=0.80,
            min_rr_ratio=1.2,
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Edge case",
        )
        profile.validate()  # Should not raise

    def test_profile_validate_momentum_boundary_0(self):
        """Test validation accepts momentum = 0."""
        profile = RegimeGateProfile(
            regime_name="NORMAL",
            confidence_threshold=50.0,
            momentum_threshold=0.0,
            risk_threshold=0.80,
            min_rr_ratio=1.2,
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Edge case",
        )
        profile.validate()  # Should not raise

    def test_profile_validate_momentum_boundary_1(self):
        """Test validation accepts momentum = 1."""
        profile = RegimeGateProfile(
            regime_name="NORMAL",
            confidence_threshold=50.0,
            momentum_threshold=1.0,
            risk_threshold=0.80,
            min_rr_ratio=1.2,
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Edge case",
        )
        profile.validate()  # Should not raise


class TestRegimeGateConfig:
    """Test RegimeGateConfig dataclass initialization."""

    def test_config_default_initialization(self):
        """Test default RegimeGateConfig initialization."""
        config = RegimeGateConfig()
        assert config.enabled is True
        assert config.version == "1.0"
        assert len(config.profiles) == 4
        assert REGIME_LOW in config.profiles
        assert REGIME_NORMAL in config.profiles
        assert REGIME_HIGH in config.profiles
        assert REGIME_EXTREME in config.profiles

    def test_config_disabled(self):
        """Test RegimeGateConfig with enabled=False."""
        config = RegimeGateConfig(enabled=False)
        assert config.enabled is False
        assert len(config.profiles) == 4  # Still has defaults

    def test_config_custom_version(self):
        """Test RegimeGateConfig with custom version."""
        config = RegimeGateConfig(version="2.0")
        assert config.version == "2.0"

    def test_config_profile_validation(self):
        """Test that invalid profiles are rejected during __post_init__."""
        bad_profile = RegimeGateProfile(
            regime_name="INVALID",
            confidence_threshold=50.0,
            momentum_threshold=0.35,
            risk_threshold=0.80,
            min_rr_ratio=1.2,
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Invalid",
        )
        with pytest.raises(ValueError):
            RegimeGateConfig(profiles={"INVALID": bad_profile})


class TestGetRegimeProfile:
    """Test get_regime_profile() function."""

    def test_get_profile_low(self):
        """Test retrieving LOW profile."""
        profile = get_regime_profile(REGIME_LOW)
        assert profile is not None
        assert profile.regime_name == REGIME_LOW
        assert profile.confidence_threshold == 0.60
        assert profile.momentum_threshold == 0.40
        assert profile.risk_threshold == 0.70
        assert profile.min_rr_ratio == 1.5
        assert profile.position_size_multiplier == 1.3
        assert profile.require_momentum_alignment is False

    def test_get_profile_normal(self):
        """Test retrieving NORMAL profile."""
        profile = get_regime_profile(REGIME_NORMAL)
        assert profile is not None
        assert profile.regime_name == REGIME_NORMAL
        assert profile.confidence_threshold == 0.50
        assert profile.momentum_threshold == 0.35
        assert profile.risk_threshold == 0.80
        assert profile.min_rr_ratio == 1.2
        assert profile.position_size_multiplier == 1.0
        assert profile.require_momentum_alignment is False

    def test_get_profile_high(self):
        """Test retrieving HIGH profile."""
        profile = get_regime_profile(REGIME_HIGH)
        assert profile is not None
        assert profile.regime_name == REGIME_HIGH
        assert profile.confidence_threshold == 0.55
        assert profile.momentum_threshold == 0.45
        assert profile.risk_threshold == 0.65
        assert profile.min_rr_ratio == 1.3
        assert profile.position_size_multiplier == 0.65
        assert profile.require_momentum_alignment is True

    def test_get_profile_extreme(self):
        """Test retrieving EXTREME profile."""
        profile = get_regime_profile(REGIME_EXTREME)
        assert profile is not None
        assert profile.regime_name == REGIME_EXTREME
        assert profile.confidence_threshold == 0.65
        assert profile.momentum_threshold == 0.50
        assert profile.risk_threshold == 0.50
        assert profile.min_rr_ratio == 1.5
        assert profile.position_size_multiplier == 0.40
        assert profile.require_momentum_alignment is True

    def test_get_profile_none(self):
        """Test get_regime_profile returns None for None regime."""
        profile = get_regime_profile(None)
        assert profile is None

    def test_get_profile_empty_string(self):
        """Test get_regime_profile returns None for empty string."""
        profile = get_regime_profile("")
        assert profile is None

    def test_get_profile_unknown_defaults_to_normal(self):
        """Test unknown regime defaults to NORMAL profile."""
        profile = get_regime_profile("UNKNOWN_REGIME")
        assert profile is not None
        assert profile.regime_name == REGIME_NORMAL


class TestApplyRegimeGates:
    """Test apply_regime_gates() function."""

    def test_apply_gates_low_regime(self):
        """Test applying gates for LOW regime."""
        result = apply_regime_gates(REGIME_LOW)
        assert result["confidence_threshold"] == 60.0
        assert result["momentum_threshold"] == 0.40
        assert result["risk_threshold"] == 0.70
        assert result["min_rr_ratio"] == 1.5
        assert result["position_size_multiplier"] == 1.3
        assert result["require_momentum_alignment"] is False

    def test_apply_gates_normal_regime(self):
        """Test applying gates for NORMAL regime."""
        result = apply_regime_gates(REGIME_NORMAL)
        assert result["confidence_threshold"] == 50.0
        assert result["momentum_threshold"] == 0.35
        assert result["risk_threshold"] == 0.80
        assert result["min_rr_ratio"] == 1.2
        assert result["position_size_multiplier"] == 1.0
        assert result["require_momentum_alignment"] is False

    def test_apply_gates_high_regime(self):
        """Test applying gates for HIGH regime."""
        result = apply_regime_gates(REGIME_HIGH)
        assert abs(result["confidence_threshold"] - 55.0) < 0.01
        assert result["momentum_threshold"] == 0.45
        assert result["risk_threshold"] == 0.65
        assert result["min_rr_ratio"] == 1.3
        assert result["position_size_multiplier"] == 0.65
        assert result["require_momentum_alignment"] is True

    def test_apply_gates_extreme_regime(self):
        """Test applying gates for EXTREME regime."""
        result = apply_regime_gates(REGIME_EXTREME)
        assert result["confidence_threshold"] == 65.0
        assert result["momentum_threshold"] == 0.50
        assert result["risk_threshold"] == 0.50
        assert result["min_rr_ratio"] == 1.5
        assert result["position_size_multiplier"] == 0.40
        assert result["require_momentum_alignment"] is True

    def test_apply_gates_none_regime(self):
        """Test apply_regime_gates with None regime uses static defaults."""
        result = apply_regime_gates(None)
        # Should return static defaults
        assert result["confidence_threshold"] == 50.0
        assert result["momentum_threshold"] == 0.35
        assert result["risk_threshold"] == 0.80
        assert result["min_rr_ratio"] == 1.2
        assert result["position_size_multiplier"] == 1.0
        assert result["require_momentum_alignment"] is False

    def test_apply_gates_empty_string_regime(self):
        """Test apply_regime_gates with empty string uses static defaults."""
        result = apply_regime_gates("")
        assert result["confidence_threshold"] == 50.0

    def test_apply_gates_unknown_regime_defaults_to_normal(self):
        """Test unknown regime defaults to NORMAL thresholds."""
        result = apply_regime_gates("UNKNOWN_REGIME")
        assert result["confidence_threshold"] == 50.0  # NORMAL default
        assert result["momentum_threshold"] == 0.35

    def test_apply_gates_with_base_thresholds(self):
        """Test apply_regime_gates respects base thresholds if provided."""
        base = {
            "confidence_threshold": 60.0,  # Will be overridden by regime
            "momentum_threshold": 0.50,    # Will be overridden by regime
        }
        result = apply_regime_gates(REGIME_HIGH, base)
        # Regime thresholds should override base thresholds
        assert abs(result["confidence_threshold"] - 55.0) < 0.01  # HIGH regime, not 60
        assert result["momentum_threshold"] == 0.45    # HIGH regime, not 0.50

    def test_apply_gates_fills_missing_keys(self):
        """Test apply_regime_gates fills in missing keys with defaults."""
        base = {
            "confidence_threshold": 45.0,  # Only one key provided
        }
        result = apply_regime_gates(None, base)
        # Should have all keys with defaults for missing ones
        assert "momentum_threshold" in result
        assert "risk_threshold" in result
        assert "position_size_multiplier" in result
        assert "require_momentum_alignment" in result

    def test_apply_gates_all_keys_present(self):
        """Test apply_regime_gates returns all required keys."""
        result = apply_regime_gates(REGIME_HIGH)
        required_keys = {
            "confidence_threshold",
            "momentum_threshold",
            "risk_threshold",
            "min_rr_ratio",
            "position_size_multiplier",
            "require_momentum_alignment",
        }
        assert set(result.keys()) >= required_keys

    def test_apply_gates_confidence_scaled_to_100(self):
        """Test that confidence is scaled to 0-100 range."""
        # LOW regime has confidence_threshold = 0.60, should scale to 60.0
        result = apply_regime_gates(REGIME_LOW)
        assert result["confidence_threshold"] == 60.0

        # NORMAL regime has 0.50, should scale to 50.0
        result = apply_regime_gates(REGIME_NORMAL)
        assert result["confidence_threshold"] == 50.0

    def test_apply_gates_momentum_stays_0_to_1(self):
        """Test that momentum stays in 0-1 range."""
        result = apply_regime_gates(REGIME_HIGH)
        assert 0.0 <= result["momentum_threshold"] <= 1.0

    def test_apply_gates_size_multiplier_positive(self):
        """Test that position_size_multiplier is always positive."""
        for regime in [REGIME_LOW, REGIME_NORMAL, REGIME_HIGH, REGIME_EXTREME]:
            result = apply_regime_gates(regime)
            assert result["position_size_multiplier"] > 0.0

    def test_apply_gates_low_confidence_low_size(self):
        """Test trade-off: LOW regime raises confidence but boosts size."""
        result = apply_regime_gates(REGIME_LOW)
        # Confidence threshold is raised to filter noise
        assert result["confidence_threshold"] > 50.0
        # But position size is boosted for quality setups
        assert result["position_size_multiplier"] > 1.0

    def test_apply_gates_extreme_confidence_small_size(self):
        """Test trade-off: EXTREME regime high confidence and small sizes."""
        result = apply_regime_gates(REGIME_EXTREME)
        # Very high confidence threshold
        assert result["confidence_threshold"] == 65.0
        # Very small position size
        assert result["position_size_multiplier"] < 0.5


class TestRegimeGatesEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_confidence_threshold(self):
        """Test profile with zero confidence threshold."""
        profile = get_regime_profile(REGIME_LOW)
        # Create new profile with zero confidence
        zero_conf = RegimeGateProfile(
            regime_name="NORMAL",
            confidence_threshold=0.0,
            momentum_threshold=0.35,
            risk_threshold=0.80,
            min_rr_ratio=1.2,
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Zero confidence",
        )
        zero_conf.validate()  # Should not raise

    def test_zero_momentum_threshold(self):
        """Test profile with zero momentum threshold."""
        zero_mom = RegimeGateProfile(
            regime_name="NORMAL",
            confidence_threshold=50.0,
            momentum_threshold=0.0,
            risk_threshold=0.80,
            min_rr_ratio=1.2,
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Zero momentum",
        )
        zero_mom.validate()  # Should not raise

    def test_max_values_valid(self):
        """Test profile with maximum allowed values."""
        max_vals = RegimeGateProfile(
            regime_name="NORMAL",
            confidence_threshold=100.0,
            momentum_threshold=1.0,
            risk_threshold=1.0,
            min_rr_ratio=10.0,
            position_size_multiplier=10.0,
            require_momentum_alignment=True,
            description="Max values",
        )
        max_vals.validate()  # Should not raise

    def test_apply_gates_robustness_on_none(self):
        """Test apply_regime_gates gracefully handles None."""
        result = apply_regime_gates(None)
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_apply_gates_robustness_on_empty(self):
        """Test apply_regime_gates gracefully handles empty string."""
        result = apply_regime_gates("")
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) > 0


class TestRegimeConstantsConsistency:
    """Test that regime constants are consistent across module."""

    def test_valid_regimes_contains_all_regimes(self):
        """Test VALID_REGIMES includes all 4 regimes."""
        assert REGIME_LOW in VALID_REGIMES
        assert REGIME_NORMAL in VALID_REGIMES
        assert REGIME_HIGH in VALID_REGIMES
        assert REGIME_EXTREME in VALID_REGIMES
        assert len(VALID_REGIMES) == 4

    def test_get_profile_works_for_all_valid_regimes(self):
        """Test get_regime_profile returns profile for each valid regime."""
        for regime in VALID_REGIMES:
            profile = get_regime_profile(regime)
            assert profile is not None
            assert profile.regime_name == regime

    def test_apply_gates_works_for_all_valid_regimes(self):
        """Test apply_regime_gates works for each valid regime."""
        for regime in VALID_REGIMES:
            result = apply_regime_gates(regime)
            assert result is not None
            assert isinstance(result, dict)
            assert "confidence_threshold" in result


class TestRegimeGatesIntegration:
    """Integration tests combining multiple functions."""

    def test_workflow_get_profile_then_apply_gates(self):
        """Test typical workflow: get profile, then apply gates."""
        profile = get_regime_profile(REGIME_HIGH)
        assert profile is not None

        result = apply_regime_gates(REGIME_HIGH)
        assert result["confidence_threshold"] == profile.confidence_threshold * 100.0
        assert result["momentum_threshold"] == profile.momentum_threshold
        assert result["position_size_multiplier"] == profile.position_size_multiplier

    def test_workflow_multiple_regimes_sequential(self):
        """Test applying gates for multiple regimes sequentially."""
        regimes = [REGIME_LOW, REGIME_NORMAL, REGIME_HIGH, REGIME_EXTREME]
        results = [apply_regime_gates(regime) for regime in regimes]

        # Each should produce different thresholds
        assert results[0]["position_size_multiplier"] > results[3]["position_size_multiplier"]
        # LOW should have smaller confidence threshold than EXTREME
        assert results[0]["confidence_threshold"] < results[3]["confidence_threshold"]

    def test_workflow_profile_validation_during_config_init(self):
        """Test that profile validation happens during config initialization."""
        # Valid config should succeed
        config = RegimeGateConfig()
        assert len(config.profiles) == 4

        # Invalid profile should fail
        bad_profile = RegimeGateProfile(
            regime_name="INVALID",
            confidence_threshold=50.0,
            momentum_threshold=0.35,
            risk_threshold=0.80,
            min_rr_ratio=1.2,
            position_size_multiplier=1.0,
            require_momentum_alignment=False,
            description="Invalid",
        )
        with pytest.raises(ValueError):
            RegimeGateConfig(profiles={"INVALID": bad_profile})
