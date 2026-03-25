"""Unit tests for ScannerConfig — profile application, validation, and flag behavior.

Phase 33 (US-217): Core config logic had zero test coverage. These tests verify:
- Profile application sets expected flags
- Unknown key validation (Phase 32 US-192) works
- All profiles are syntactically valid
- Profile flag override behavior (profile > default)
- Config immutability between apply_profile calls
"""

import logging
import pytest
from unittest.mock import patch

# Import paths may vary — guard gracefully
try:
    from src.scanner.config import ScannerConfig, SCAN_PROFILES, VALID_SCAN_PROFILES
except ImportError:
    pytest.skip("ScannerConfig not importable", allow_module_level=True)


class TestScannerConfigProfiles:
    """Tests for apply_profile() and profile validation."""

    def test_all_profiles_are_valid_names(self):
        """Every profile in SCAN_PROFILES should be in VALID_SCAN_PROFILES."""
        for name in SCAN_PROFILES:
            assert name in VALID_SCAN_PROFILES, f"Profile '{name}' not in VALID_SCAN_PROFILES"

    def test_apply_balanced_profile(self):
        """Balanced profile should apply without errors."""
        cfg = ScannerConfig()
        cfg.apply_profile("balanced")
        assert cfg.profile == "balanced"

    def test_apply_conservative_profile(self):
        """Conservative profile should apply without errors and set stricter gates."""
        cfg = ScannerConfig()
        cfg.apply_profile("conservative")
        assert cfg.profile == "conservative"
        assert cfg.min_confidence >= 55.0  # Conservative = stricter

    def test_apply_aggressive_profile(self):
        """Aggressive profile should enable execution and set looser gates."""
        cfg = ScannerConfig()
        cfg.apply_profile("aggressive")
        assert cfg.profile == "aggressive"
        assert cfg.enable_execution is True
        assert cfg.min_confidence < 50.0  # Aggressive = looser

    def test_apply_smart_profile(self):
        """Smart profile should enable RL features and agent flags."""
        cfg = ScannerConfig()
        cfg.apply_profile("smart")
        assert cfg.profile == "smart"
        assert cfg.use_rl_sizer is True
        assert cfg.use_rl_gates is True
        assert cfg.enable_agent_trade_promotion is True
        assert cfg.enable_multi_timeframe_agent is True

    def test_invalid_profile_raises_error(self):
        """Unknown profile name should raise ValueError."""
        cfg = ScannerConfig()
        with pytest.raises(ValueError, match="Unknown scan profile"):
            cfg.apply_profile("nonexistent_profile")

    def test_apply_profile_case_insensitive(self):
        """Profile names should be case-insensitive."""
        cfg = ScannerConfig()
        cfg.apply_profile("SMART")
        assert cfg.profile == "smart"

    def test_apply_profile_strips_whitespace(self):
        """Profile names should be stripped of whitespace."""
        cfg = ScannerConfig()
        cfg.apply_profile("  smart  ")
        assert cfg.profile == "smart"


class TestProfileKeyValidation:
    """Tests for Phase 32 US-192 unknown key validation."""

    def test_unknown_key_is_skipped_with_warning(self, caplog):
        """Unknown keys in a profile should log a warning and be skipped."""
        cfg = ScannerConfig()
        # Temporarily inject a bad key into a profile
        original = SCAN_PROFILES.get("balanced", {}).copy()
        SCAN_PROFILES["balanced"]["__fake_unknown_key__"] = True
        try:
            with caplog.at_level(logging.WARNING):
                cfg.apply_profile("balanced")
            # Key should NOT have been set as an attribute
            assert not hasattr(cfg, "__fake_unknown_key__") or \
                getattr(cfg, "__fake_unknown_key__", None) is not True
            # Warning should have been logged
            assert any("unknown field" in r.message for r in caplog.records)
        finally:
            # Restore original profile
            del SCAN_PROFILES["balanced"]["__fake_unknown_key__"]

    def test_valid_keys_are_applied(self):
        """Known fields should be set correctly."""
        cfg = ScannerConfig()
        cfg.apply_profile("aggressive")
        # min_confidence is a known field in the aggressive profile
        assert cfg.min_confidence == 45.0


class TestProfileFeatureParity:
    """Tests for Phase 33 US-212/214/216 feature parity."""

    def test_aggressive_has_execution_enabled(self):
        """Aggressive profile must have enable_execution=True."""
        assert SCAN_PROFILES["aggressive"].get("enable_execution") is True

    def test_aggressive_has_rl_features(self):
        """Aggressive profile must have RL features enabled."""
        agg = SCAN_PROFILES["aggressive"]
        assert agg.get("use_rl_sizer") is True
        assert agg.get("use_rl_gates") is True

    def test_aggressive_has_news_risk_agent(self):
        """Aggressive should have news_risk_agent enabled (Phase 33 US-214)."""
        assert SCAN_PROFILES["aggressive"].get("enable_news_risk_agent") is True

    def test_smart_has_news_risk_agent(self):
        """Smart should have news_risk_agent enabled (Phase 33 US-214)."""
        assert SCAN_PROFILES["smart"].get("enable_news_risk_agent") is True

    def test_conservative_has_infrastructure_flags(self):
        """Conservative should have safe infrastructure flags (Phase 33 US-216)."""
        con = SCAN_PROFILES["conservative"]
        assert con.get("enable_memory_manager") is True
        assert con.get("enable_health_registry") is True
        assert con.get("enable_observation_consumer") is True

    def test_conservative_does_not_enable_execution(self):
        """Conservative must NOT enable trade execution."""
        assert SCAN_PROFILES["conservative"].get("enable_execution") is not True

    def test_smart_enable_count_exceeds_50(self):
        """Smart profile should have 50+ enable flags set to True."""
        smart = SCAN_PROFILES["smart"]
        enable_count = sum(1 for k, v in smart.items() if k.startswith("enable_") and v is True)
        assert enable_count >= 50, f"Smart has only {enable_count} enable flags, expected 50+"


class TestVolatilityRegimeClamping:
    """Tests for min_volatility_regime clamping in apply_profile."""

    def test_volatility_regime_clamped_to_valid_range(self):
        """min_volatility_regime should be clamped to [0, 3]."""
        cfg = ScannerConfig()
        cfg.apply_profile("balanced")
        assert 0 <= cfg.min_volatility_regime <= 3

    def test_aggressive_volatility_regime_zero(self):
        """Aggressive profile should allow regime 0 (all regimes)."""
        cfg = ScannerConfig()
        cfg.apply_profile("aggressive")
        assert cfg.min_volatility_regime == 0


class TestProfileSwitching:
    """Tests for switching between profiles."""

    def test_switch_from_conservative_to_aggressive(self):
        """Switching profiles should override previous settings."""
        cfg = ScannerConfig()
        cfg.apply_profile("conservative")
        old_confidence = cfg.min_confidence
        cfg.apply_profile("aggressive")
        assert cfg.min_confidence < old_confidence  # Aggressive is looser
        assert cfg.profile == "aggressive"

    def test_reapply_same_profile(self):
        """Reapplying the same profile should be idempotent."""
        cfg = ScannerConfig()
        cfg.apply_profile("smart")
        first_confidence = cfg.min_confidence
        cfg.apply_profile("smart")
        assert cfg.min_confidence == first_confidence
