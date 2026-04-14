"""Phase 37 (US-240): Unit tests for ScannerConfig validation.

Tests __post_init__ boundary clamping, path resolution, profile defaults,
get_pip_value, get_trading_session_status, and safety invariant checks.
Supplements existing test_scanner_config.py (profile application tests).
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.scanner.config import (
    ScannerConfig,
    SCAN_PROFILES,
    MAJOR_PAIRS,
    CROSS_PAIRS,
    DEFAULT_PAIRS,
    PROJECT_ROOT,
)
from src.brokers.registry import get_registry


# ---------------------------------------------------------------------------
# Tests: __post_init__ threshold clamping
# ---------------------------------------------------------------------------
class TestPostInitClamping:
    def test_min_tcn_probability_clamped_low(self):
        cfg = ScannerConfig(min_tcn_probability=0.10)
        assert cfg.min_tcn_probability >= 0.50

    def test_min_tcn_probability_clamped_high(self):
        cfg = ScannerConfig(min_tcn_probability=1.5)
        assert cfg.min_tcn_probability <= 0.99

    def test_final_score_threshold_clamped_low(self):
        cfg = ScannerConfig(final_score_threshold=0.01)
        assert cfg.final_score_threshold >= 0.20

    def test_final_score_threshold_clamped_high(self):
        cfg = ScannerConfig(final_score_threshold=5.0)
        assert cfg.final_score_threshold <= 0.99

    def test_max_uncertainty_std_clamped_low(self):
        cfg = ScannerConfig(max_uncertainty_std=0.001)
        assert cfg.max_uncertainty_std >= 0.01

    def test_max_uncertainty_std_clamped_high(self):
        cfg = ScannerConfig(max_uncertainty_std=10.0)
        assert cfg.max_uncertainty_std <= 0.50

    def test_weighted_vote_threshold_clamped(self):
        cfg = ScannerConfig(weighted_vote_threshold=0.10)
        assert cfg.weighted_vote_threshold >= 0.40

    def test_sub_inference_min_confidence_clamped(self):
        cfg = ScannerConfig(sub_inference_min_confidence=0.01)
        assert cfg.sub_inference_min_confidence >= 0.30

    def test_sub_inference_vote_threshold_clamped(self):
        cfg = ScannerConfig(sub_inference_vote_threshold=0.10)
        assert cfg.sub_inference_vote_threshold >= 0.34

    def test_sub_inference_window_checks_min_1(self):
        cfg = ScannerConfig(sub_inference_window_checks=0)
        assert cfg.sub_inference_window_checks >= 1

    def test_sub_inference_workers_min_1(self):
        cfg = ScannerConfig(sub_inference_workers=-5)
        assert cfg.sub_inference_workers >= 1

    def test_sub_inference_max_candidates_min_1(self):
        cfg = ScannerConfig(sub_inference_max_candidates=0)
        assert cfg.sub_inference_max_candidates >= 1

    def test_max_uncertainty_score_clamped(self):
        cfg = ScannerConfig(max_uncertainty_score=0.01)
        assert cfg.max_uncertainty_score >= 0.10

    def test_max_model_disagreement_clamped(self):
        cfg = ScannerConfig(max_model_disagreement=0.01)
        assert cfg.max_model_disagreement >= 0.10

    def test_agent_promotion_min_confidence_clamped(self):
        cfg = ScannerConfig(agent_promotion_min_confidence=0.01)
        assert cfg.agent_promotion_min_confidence >= 0.30


# ---------------------------------------------------------------------------
# Tests: __post_init__ path resolution
# ---------------------------------------------------------------------------
class TestPathResolution:
    def test_string_config_path_converted(self):
        cfg = ScannerConfig(config_path="config/test.yaml")
        assert isinstance(cfg.config_path, Path)

    def test_string_model_dir_converted(self):
        cfg = ScannerConfig(model_dir="trained_data/models")
        assert isinstance(cfg.model_dir, Path)

    def test_relative_config_path_resolved(self):
        cfg = ScannerConfig(config_path="config/test.yaml")
        assert cfg.config_path.is_absolute()

    def test_relative_model_dir_resolved(self):
        cfg = ScannerConfig(model_dir="trained_data/models")
        assert cfg.model_dir.is_absolute()

    def test_absolute_path_preserved(self):
        cfg = ScannerConfig(config_path="/tmp/test.yaml")
        assert str(cfg.config_path) == "/tmp/test.yaml"


# ---------------------------------------------------------------------------
# Tests: __post_init__ profile normalization
# ---------------------------------------------------------------------------
class TestProfileNormalization:
    def test_unknown_profile_defaults_to_balanced(self):
        cfg = ScannerConfig(profile="nonexistent")
        assert cfg.profile == "balanced"

    def test_profile_lowercase(self):
        cfg = ScannerConfig(profile="BALANCED")
        assert cfg.profile == "balanced"

    def test_profile_stripped(self):
        cfg = ScannerConfig(profile="  smart  ")
        assert cfg.profile == "smart"


# ---------------------------------------------------------------------------
# Tests: get_pip_value
# ---------------------------------------------------------------------------
class TestGetPipValue:
    def test_major_pair(self):
        cfg = ScannerConfig()
        assert cfg.get_pip_value("EUR_USD") == 0.0001

    def test_jpy_pair(self):
        cfg = ScannerConfig()
        assert cfg.get_pip_value("USD_JPY") == 0.01

    def test_unknown_pair_default(self):
        cfg = ScannerConfig()
        assert cfg.get_pip_value("FAKE_PAIR") == 0.0001

    def test_all_major_pairs_have_pip_values(self):
        registry = get_registry()
        for pair in MAJOR_PAIRS:
            assert registry.contains(pair), f"Missing instrument in registry for {pair}"
            inst = registry.get(pair)
            assert inst.pip_value > 0, f"Invalid pip_value for {pair}"

    def test_all_cross_pairs_have_pip_values(self):
        registry = get_registry()
        for pair in CROSS_PAIRS:
            assert registry.contains(pair), f"Missing instrument in registry for {pair}"
            inst = registry.get(pair)
            assert inst.pip_value > 0, f"Invalid pip_value for {pair}"


# ---------------------------------------------------------------------------
# Tests: get_trading_session_status
# ---------------------------------------------------------------------------
class TestTradingSessionStatus:
    def test_weekday_within_hours(self):
        cfg = ScannerConfig(enable_session_filter=True, session_start_utc=7, session_end_utc=21)
        # Monday 10:00 UTC
        now = datetime(2026, 3, 23, 10, 0, tzinfo=timezone.utc)
        status = cfg.get_trading_session_status(now)
        assert status["market_open"] is True
        assert status["session_allowed"] is True

    def test_saturday_market_closed(self):
        cfg = ScannerConfig(block_when_market_closed=True)
        # Saturday 12:00 UTC
        now = datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc)
        status = cfg.get_trading_session_status(now)
        assert status["market_open"] is False

    def test_session_filter_disabled(self):
        cfg = ScannerConfig(enable_session_filter=False)
        # Monday 3:00 UTC (outside typical session)
        now = datetime(2026, 3, 23, 3, 0, tzinfo=timezone.utc)
        status = cfg.get_trading_session_status(now)
        assert status["session_allowed"] is True


# ---------------------------------------------------------------------------
# Tests: Module constants
# ---------------------------------------------------------------------------
class TestModuleConstants:
    def test_default_pairs_includes_majors_and_crosses(self):
        assert len(DEFAULT_PAIRS) == len(MAJOR_PAIRS) + len(CROSS_PAIRS)

    def test_major_pairs_count(self):
        assert len(MAJOR_PAIRS) == 7

    def test_cross_pairs_count(self):
        assert len(CROSS_PAIRS) == 8

    def test_project_root_is_absolute(self):
        assert PROJECT_ROOT.is_absolute()


# ---------------------------------------------------------------------------
# Tests: Config defaults
# ---------------------------------------------------------------------------
class TestConfigDefaults:
    def test_default_profile_balanced(self):
        cfg = ScannerConfig()
        assert cfg.profile == "balanced"

    def test_default_execution_enabled(self):
        cfg = ScannerConfig()
        assert cfg.enable_execution is True

    def test_default_enable_execution_true(self):
        """enable_execution defaults to True (live execution mode)."""
        cfg = ScannerConfig()
        assert cfg.enable_execution is True

    def test_default_top_n(self):
        cfg = ScannerConfig()
        assert cfg.top_n >= 1

    def test_default_meta_labeler_threshold_in_range(self):
        cfg = ScannerConfig()
        assert 0.50 <= cfg.meta_labeler_threshold <= 0.60


# ---------------------------------------------------------------------------
# Tests: Profile safety invariants
# ---------------------------------------------------------------------------
class TestProfileSafetyInvariants:
    def test_conservative_never_enables_execution(self):
        """Conservative profile must NEVER enable trade execution."""
        assert "enable_execution" not in SCAN_PROFILES["conservative"] or \
               SCAN_PROFILES["conservative"]["enable_execution"] is False

    def test_all_profiles_have_blocked_pairs(self):
        """Profiles that include blocked_pairs should have it as a list."""
        for name, profile in SCAN_PROFILES.items():
            if "blocked_pairs" in profile:
                assert isinstance(profile["blocked_pairs"], list), (
                    f"Profile '{name}' blocked_pairs must be a list"
                )

    def test_aggressive_min_risk_reward_above_one(self):
        """Even aggressive profile should maintain minimum R:R ratio."""
        agg = SCAN_PROFILES["aggressive"]
        assert agg.get("min_risk_reward_ratio", 1.2) >= 1.0

    def test_no_profile_allows_negative_confidence(self):
        """No profile should set min_confidence below 0."""
        for name, profile in SCAN_PROFILES.items():
            if "min_confidence" in profile:
                assert profile["min_confidence"] >= 0, \
                    f"Profile '{name}' has negative min_confidence"
