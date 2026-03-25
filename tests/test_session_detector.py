"""
Unit tests for SessionDetector module.

Tests session detection, confidence modifiers, position sizing multipliers,
and edge cases including midnight-wrapping sessions.
"""

import pytest
from datetime import datetime, timezone, timedelta
from src.scanner.session_detector import (
    SessionDetector,
    SessionConfig,
    SessionInfo,
)


class TestSessionConfigValidation:
    """Test SessionConfig validation."""

    def test_config_defaults_validate_successfully(self):
        """Test that default config values are valid."""
        config = SessionConfig()
        config.validate()  # Should not raise

    def test_config_with_invalid_hour_raises_error(self):
        """Test that invalid hour values raise ValueError."""
        config = SessionConfig(tokyo_start=25)
        with pytest.raises(ValueError, match="tokyo_start must be an integer between 0 and 23"):
            config.validate()

    def test_config_with_negative_hour_raises_error(self):
        """Test that negative hour values raise ValueError."""
        config = SessionConfig(london_end=-1)
        with pytest.raises(ValueError, match="london_end must be an integer between 0 and 23"):
            config.validate()

    def test_config_with_invalid_multiplier_raises_error(self):
        """Test that invalid position multiplier raises ValueError."""
        config = SessionConfig(tokyo_position_multiplier=0)
        with pytest.raises(ValueError, match="tokyo_position_multiplier must be > 0"):
            config.validate()

    def test_config_with_negative_multiplier_raises_error(self):
        """Test that negative position multiplier raises ValueError."""
        config = SessionConfig(london_position_multiplier=-0.5)
        with pytest.raises(ValueError, match="london_position_multiplier must be > 0"):
            config.validate()

    def test_config_version_field_exists(self):
        """Test that config has version field for forward compatibility."""
        config = SessionConfig()
        assert hasattr(config, "version")
        assert config.version == "1.0"


class TestSessionDetectionTokyoSession:
    """Test Tokyo session detection (23:00-08:00 UTC, wraps midnight)."""

    def test_tokyo_session_23_00_utc(self):
        """Test Tokyo session at 23:00 UTC."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 23, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "TOKYO"
        assert session.description == "Tokyo Session"

    def test_tokyo_session_00_00_utc(self):
        """Test Tokyo session at 00:00 UTC (midnight)."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "TOKYO"

    def test_tokyo_session_07_59_utc(self):
        """Test Tokyo session at 07:59 UTC (before end)."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 7, 59, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "TOKYO"

    def test_tokyo_session_ends_at_08_00_utc(self):
        """Test Tokyo session ends at 08:00 UTC."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 8, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name != "TOKYO"  # Should be London or off-hours


class TestSessionDetectionLondonSession:
    """Test London session detection (08:00-17:00 UTC)."""

    def test_london_session_08_00_utc(self):
        """Test London session at 08:00 UTC."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 8, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "LONDON"
        assert session.description == "London Session"

    def test_london_session_12_00_utc(self):
        """Test London session at 12:00 UTC."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "LONDON"

    def test_london_session_16_59_utc(self):
        """Test London session at 16:59 UTC (still in overlap)."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 16, 59, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        # At 16:59, London-NY overlap is still active (ends at 17:00)
        assert session.name == "LONDON_NY_OVERLAP"

    def test_london_session_ends_at_17_00_utc(self):
        """Test London session ends at 17:00 UTC."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 17, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name != "LONDON"


class TestSessionDetectionNewYorkSession:
    """Test New York session detection (13:00-22:00 UTC)."""

    def test_new_york_session_13_00_utc(self):
        """Test New York session at 13:00 UTC."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 13, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        # At 13:00, London-NY overlap takes priority
        assert session.name == "LONDON_NY_OVERLAP"

    def test_new_york_session_17_01_utc(self):
        """Test New York session at 17:01 UTC (after overlap ends)."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 17, 1, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "NEW_YORK"

    def test_new_york_session_20_00_utc(self):
        """Test New York session at 20:00 UTC."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 20, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "NEW_YORK"

    def test_new_york_session_21_59_utc(self):
        """Test New York session at 21:59 UTC (before end)."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 21, 59, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "NEW_YORK"

    def test_new_york_session_ends_at_22_00_utc(self):
        """Test New York session ends at 22:00 UTC."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 22, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name != "NEW_YORK"


class TestSessionDetectionLondonNYOverlap:
    """Test London-NY overlap detection (13:00-17:00 UTC)."""

    def test_overlap_at_13_00_utc(self):
        """Test overlap at 13:00 UTC."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 13, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "LONDON_NY_OVERLAP"
        assert session.is_overlap is True
        assert session.overlap_name == "LONDON_NY"

    def test_overlap_at_15_00_utc(self):
        """Test overlap at 15:00 UTC (middle)."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 15, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "LONDON_NY_OVERLAP"
        assert session.is_overlap is True

    def test_overlap_at_16_59_utc(self):
        """Test overlap at 16:59 UTC (before end)."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 16, 59, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "LONDON_NY_OVERLAP"

    def test_overlap_ends_at_17_00_utc(self):
        """Test overlap ends at 17:00 UTC."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 17, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        # At 17:00, should be NEW_YORK (not overlap, not London)
        assert session.name == "NEW_YORK"


class TestSessionDetectionOffHours:
    """Test off-hours detection (22:00-23:00 UTC gap)."""

    def test_off_hours_at_22_00_utc(self):
        """Test off-hours at 22:00 UTC."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 22, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "OFF_HOURS"
        assert session.description == "Off Hours (Low Liquidity)"

    def test_off_hours_at_22_30_utc(self):
        """Test off-hours at 22:30 UTC."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 22, 30, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "OFF_HOURS"

    def test_off_hours_at_22_59_utc(self):
        """Test off-hours at 22:59 UTC (before Tokyo)."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 22, 59, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "OFF_HOURS"


class TestConfidenceModifiers:
    """Test confidence modifiers per session and strategy type."""

    def test_tokyo_mean_reversion_modifier(self):
        """Test Tokyo session boosts mean_reversion confidence."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        modifier = detector.get_confidence_modifier("mean_reversion", session)
        assert modifier == 0.10  # Tokyo favors mean reversion

    def test_tokyo_trend_modifier(self):
        """Test Tokyo session reduces trend confidence."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        modifier = detector.get_confidence_modifier("trend", session)
        assert modifier == -0.15  # Tokyo discourages trend

    def test_london_trend_modifier(self):
        """Test London session boosts trend confidence."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        modifier = detector.get_confidence_modifier("trend", session)
        assert modifier == 0.10  # London favors trend

    def test_london_ny_overlap_momentum_modifier(self):
        """Test London-NY overlap boosts momentum confidence."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 14, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        modifier = detector.get_confidence_modifier("momentum", session)
        assert modifier == 0.10  # Overlap favors momentum

    def test_off_hours_all_strategies_penalized(self):
        """Test off-hours reduces all strategy confidence."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 22, 30, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        for strategy in ["trend", "mean_reversion", "volatility", "momentum", "breakout"]:
            modifier = detector.get_confidence_modifier(strategy, session)
            assert modifier == -0.20

    def test_unknown_strategy_returns_zero(self):
        """Test unknown strategy type returns 0.0 modifier."""
        detector = SessionDetector()
        session = detector.get_current_session()
        modifier = detector.get_confidence_modifier("unknown_strategy", session)
        assert modifier == 0.0

    def test_confidence_modifier_without_explicit_session(self):
        """Test confidence modifier uses current session if not provided."""
        detector = SessionDetector()
        # Should not raise, should use current session
        modifier = detector.get_confidence_modifier("trend")
        assert isinstance(modifier, float)


class TestPositionSizeMultipliers:
    """Test position sizing multipliers per session."""

    def test_tokyo_position_multiplier(self):
        """Test Tokyo session position multiplier is 0.70."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 0, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        multiplier = detector.get_position_size_multiplier(session)
        assert multiplier == 0.70

    def test_london_position_multiplier(self):
        """Test London session position multiplier is 1.00."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        multiplier = detector.get_position_size_multiplier(session)
        assert multiplier == 1.00

    def test_new_york_position_multiplier(self):
        """Test New York session position multiplier is 0.85."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 20, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        multiplier = detector.get_position_size_multiplier(session)
        assert multiplier == 0.85

    def test_london_ny_overlap_position_multiplier(self):
        """Test London-NY overlap position multiplier is 0.75."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 14, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        multiplier = detector.get_position_size_multiplier(session)
        assert multiplier == 0.75

    def test_off_hours_position_multiplier(self):
        """Test off-hours position multiplier is 0.50."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 22, 30, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        multiplier = detector.get_position_size_multiplier(session)
        assert multiplier == 0.50

    def test_position_multiplier_without_explicit_session(self):
        """Test position multiplier uses current session if not provided."""
        detector = SessionDetector()
        # Should not raise, should use current session
        multiplier = detector.get_position_size_multiplier()
        assert isinstance(multiplier, float)
        assert multiplier > 0


class TestSessionSummary:
    """Test session summary functionality."""

    def test_session_summary_includes_required_fields(self):
        """Test that session summary includes all required fields."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
        summary = detector.get_session_summary(utc_time)

        assert "enabled" in summary
        assert "utc_time" in summary
        assert "utc_hour" in summary
        assert "current_session" in summary
        assert "current_description" in summary
        assert "active_sessions" in summary
        assert "position_size_multiplier" in summary
        assert "is_overlap" in summary

    def test_session_summary_at_london_session(self):
        """Test session summary during London session."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 10, 0, 0, tzinfo=timezone.utc)
        summary = detector.get_session_summary(utc_time)

        assert summary["enabled"] is True
        assert summary["current_session"] == "LONDON"
        assert "LONDON" in summary["active_sessions"]

    def test_session_summary_during_overlap(self):
        """Test session summary during London-NY overlap."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 15, 0, 0, tzinfo=timezone.utc)
        summary = detector.get_session_summary(utc_time)

        assert summary["current_session"] == "LONDON_NY_OVERLAP"
        assert summary["is_overlap"] is True

    def test_session_summary_when_disabled(self):
        """Test session summary when detection is disabled."""
        config = SessionConfig(enabled=False)
        detector = SessionDetector(config)
        summary = detector.get_session_summary()

        assert summary["enabled"] is False
        assert "disabled" in summary["message"].lower()


class TestConfigDisabled:
    """Test behavior when session detection is disabled."""

    def test_disabled_returns_neutral_session(self):
        """Test disabled config returns neutral session."""
        config = SessionConfig(enabled=False)
        detector = SessionDetector(config)
        session = detector.get_current_session()

        assert session.name == "DISABLED"
        assert session.position_size_multiplier == 1.0

    def test_disabled_returns_empty_modifiers(self):
        """Test disabled config returns empty modifiers dict."""
        config = SessionConfig(enabled=False)
        detector = SessionDetector(config)
        session = detector.get_current_session()

        assert session.confidence_modifiers == {}


class TestMidnightBoundaryHandling:
    """Test midnight-crossing sessions (Tokyo wraps 23:00 to 08:00)."""

    def test_tokyo_wraps_correctly_at_23_59(self):
        """Test Tokyo session wraps correctly at 23:59."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 23, 59, 59, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "TOKYO"

    def test_tokyo_wraps_correctly_at_00_30(self):
        """Test Tokyo session continues after midnight."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 24, 0, 30, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "TOKYO"

    def test_tokyo_ends_correctly_at_07_59(self):
        """Test Tokyo session ends correctly before London."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 24, 7, 59, 59, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "TOKYO"

    def test_london_starts_correctly_at_08_00(self):
        """Test London session starts when Tokyo ends."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 24, 8, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "LONDON"


class TestSessionBoundaryEdgeCases:
    """Test edge cases at session boundaries."""

    def test_exact_london_start_boundary(self):
        """Test exact boundary at London start (08:00)."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 8, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "LONDON"

    def test_one_minute_before_london_end(self):
        """Test one minute before London-NY overlap ends."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 16, 59, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        # At 16:59, still in overlap (overlap ends at 17:00)
        assert session.name == "LONDON_NY_OVERLAP"

    def test_exact_new_york_start_boundary(self):
        """Test exact boundary at New York start (13:00 in overlap)."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 13, 0, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "LONDON_NY_OVERLAP"

    def test_one_minute_after_new_york_end(self):
        """Test one minute after New York end."""
        detector = SessionDetector()
        utc_time = datetime(2026, 3, 23, 22, 1, 0, tzinfo=timezone.utc)
        session = detector.get_current_session(utc_time)
        assert session.name == "OFF_HOURS"


class TestSessionInfoDataclass:
    """Test SessionInfo dataclass."""

    def test_session_info_creation(self):
        """Test SessionInfo creation with all fields."""
        info = SessionInfo(
            name="LONDON",
            is_overlap=False,
            overlap_name=None,
            confidence_modifiers={"trend": 0.10},
            position_size_multiplier=1.0,
            description="London Session",
            start_hour_utc=8,
            end_hour_utc=17,
        )
        assert info.name == "LONDON"
        assert info.is_overlap is False
        assert info.position_size_multiplier == 1.0

    def test_session_info_with_overlap(self):
        """Test SessionInfo with overlap values."""
        info = SessionInfo(
            name="LONDON_NY_OVERLAP",
            is_overlap=True,
            overlap_name="LONDON_NY",
            confidence_modifiers={},
            position_size_multiplier=0.75,
            description="Overlap",
            start_hour_utc=13,
            end_hour_utc=17,
        )
        assert info.is_overlap is True
        assert info.overlap_name == "LONDON_NY"


class TestIsInSessionHelper:
    """Test _is_in_session helper method."""

    def test_is_in_session_simple_range(self):
        """Test simple range without wrapping."""
        assert SessionDetector._is_in_session(10, 8, 17, wrap=False) is True
        assert SessionDetector._is_in_session(7, 8, 17, wrap=False) is False

    def test_is_in_session_wrapping_range(self):
        """Test wrapping range (crossing midnight)."""
        # Hour 23 should be in 23-8 range
        assert SessionDetector._is_in_session(23, 23, 8, wrap=True) is True
        # Hour 0 should be in 23-8 range
        assert SessionDetector._is_in_session(0, 23, 8, wrap=True) is True
        # Hour 7 should be in 23-8 range
        assert SessionDetector._is_in_session(7, 23, 8, wrap=True) is True
        # Hour 8 should NOT be in 23-8 range
        assert SessionDetector._is_in_session(8, 23, 8, wrap=True) is False

    def test_is_in_session_boundary_conditions(self):
        """Test boundary conditions for is_in_session."""
        # Start hour should be included
        assert SessionDetector._is_in_session(8, 8, 17, wrap=False) is True
        # End hour should NOT be included
        assert SessionDetector._is_in_session(17, 8, 17, wrap=False) is False


class TestCurrentUTCTimeDefault:
    """Test default behavior when no UTC time is provided."""

    def test_get_current_session_uses_utc_now_when_none(self):
        """Test that None time parameter uses current UTC time."""
        detector = SessionDetector()
        session = detector.get_current_session(None)
        # Should return valid session (not raise)
        assert session.name in [
            "TOKYO",
            "LONDON",
            "NEW_YORK",
            "LONDON_NY_OVERLAP",
            "OFF_HOURS",
            "DISABLED",
        ]

    def test_get_session_summary_uses_utc_now_when_none(self):
        """Test that get_session_summary uses current UTC when none provided."""
        detector = SessionDetector()
        summary = detector.get_session_summary(None)
        # Should have valid structure
        assert isinstance(summary, dict)
        assert "enabled" in summary
