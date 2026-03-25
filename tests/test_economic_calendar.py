"""
Tests for EconomicCalendarFilter (US-301, Phase 48).

Covers: blackout detection, currency-pair matching, cache persistence,
post-event buffer, medium impact reduction, disabled filter, fallbacks.
"""

import json
import os
import pytest
import tempfile
from datetime import datetime, timezone, timedelta

from src.scanner.economic_calendar import (
    EconomicCalendarFilter,
    EconomicCalendarConfig,
    EconomicEvent,
    CalendarCheckResult,
    create_default_calendar_filter,
)


def _make_event(
    currency="USD",
    impact="HIGH",
    event_name="Non-Farm Payrolls",
    minutes_from_now=20,
    base_time=None,
):
    """Helper to create an event N minutes from a reference time."""
    now = base_time or datetime.now(timezone.utc)
    event_time = now + timedelta(minutes=minutes_from_now)
    return EconomicEvent(
        timestamp=event_time.isoformat(),
        currency=currency,
        impact=impact,
        event_name=event_name,
    )


class TestEconomicEvent:
    """Test EconomicEvent dataclass."""

    def test_event_time_parsing(self):
        """ISO timestamp parses correctly."""
        evt = EconomicEvent(
            timestamp="2026-03-24T14:30:00+00:00",
            currency="USD",
            impact="HIGH",
            event_name="NFP",
        )
        assert evt.event_time.year == 2026
        assert evt.event_time.month == 3

    def test_event_time_z_suffix(self):
        """Z suffix timestamps parse correctly."""
        evt = EconomicEvent(
            timestamp="2026-03-24T14:30:00Z",
            currency="USD",
            impact="HIGH",
            event_name="NFP",
        )
        assert evt.event_time.tzinfo is not None

    def test_event_time_invalid_fallback(self):
        """Invalid timestamp falls back to now."""
        evt = EconomicEvent(
            timestamp="not-a-date",
            currency="USD",
            impact="HIGH",
            event_name="NFP",
        )
        # Should not raise, falls back to now
        assert isinstance(evt.event_time, datetime)


class TestCalendarCheckResult:
    """Test CalendarCheckResult."""

    def test_clear_result(self):
        result = CalendarCheckResult(action="CLEAR", reason="ok", size_multiplier=1.0)
        assert result.action == "CLEAR"
        assert result.size_multiplier == 1.0

    def test_blackout_result(self):
        result = CalendarCheckResult(action="BLACKOUT", reason="nfp", size_multiplier=0.0)
        assert result.size_multiplier == 0.0


class TestEconomicCalendarFilter:
    """Test core filter logic."""

    def test_high_impact_blackout_before_event(self):
        """HIGH impact event within 30min triggers BLACKOUT."""
        now = datetime(2026, 3, 24, 14, 0, 0, tzinfo=timezone.utc)
        config = EconomicCalendarConfig(enabled=True)
        f = EconomicCalendarFilter(config)
        f._events = [_make_event("USD", "HIGH", "NFP", 20, now)]

        result = f.check_pair("EUR_USD", check_time=now)
        assert result.action == "BLACKOUT"
        assert result.size_multiplier == 0.0
        assert result.nearest_event is not None

    def test_high_impact_post_buffer(self):
        """HIGH impact event released 3min ago triggers BLACKOUT (5min buffer)."""
        now = datetime(2026, 3, 24, 14, 0, 0, tzinfo=timezone.utc)
        config = EconomicCalendarConfig(enabled=True)
        f = EconomicCalendarFilter(config)
        f._events = [_make_event("USD", "HIGH", "NFP", -3, now)]

        result = f.check_pair("EUR_USD", check_time=now)
        assert result.action == "BLACKOUT"

    def test_high_impact_post_buffer_expired(self):
        """HIGH impact event released 10min ago is CLEAR (past 5min buffer)."""
        now = datetime(2026, 3, 24, 14, 0, 0, tzinfo=timezone.utc)
        config = EconomicCalendarConfig(enabled=True)
        f = EconomicCalendarFilter(config)
        f._events = [_make_event("USD", "HIGH", "NFP", -10, now)]

        result = f.check_pair("EUR_USD", check_time=now)
        assert result.action == "CLEAR"

    def test_medium_impact_reduce_size(self):
        """MEDIUM impact within 15min triggers REDUCE_SIZE with 0.5x."""
        now = datetime(2026, 3, 24, 14, 0, 0, tzinfo=timezone.utc)
        config = EconomicCalendarConfig(enabled=True)
        f = EconomicCalendarFilter(config)
        f._events = [_make_event("EUR", "MEDIUM", "PMI", 10, now)]

        result = f.check_pair("EUR_USD", check_time=now)
        assert result.action == "REDUCE_SIZE"
        assert result.size_multiplier == 0.50

    def test_low_impact_no_action(self):
        """LOW impact events never trigger blackout."""
        now = datetime(2026, 3, 24, 14, 0, 0, tzinfo=timezone.utc)
        config = EconomicCalendarConfig(enabled=True)
        f = EconomicCalendarFilter(config)
        f._events = [_make_event("USD", "LOW", "Housing Starts", 5, now)]

        result = f.check_pair("EUR_USD", check_time=now)
        assert result.action == "CLEAR"

    def test_currency_pair_matching_usd(self):
        """USD event affects EUR_USD but not EUR_GBP."""
        now = datetime(2026, 3, 24, 14, 0, 0, tzinfo=timezone.utc)
        config = EconomicCalendarConfig(enabled=True)
        f = EconomicCalendarFilter(config)
        f._events = [_make_event("USD", "HIGH", "NFP", 10, now)]

        # EUR_USD has USD → affected
        result1 = f.check_pair("EUR_USD", check_time=now)
        assert result1.action == "BLACKOUT"

        # EUR_GBP has no USD → not affected
        result2 = f.check_pair("EUR_GBP", check_time=now)
        assert result2.action == "CLEAR"

    def test_currency_pair_matching_eur(self):
        """EUR event affects EUR_USD and EUR_JPY."""
        now = datetime(2026, 3, 24, 14, 0, 0, tzinfo=timezone.utc)
        config = EconomicCalendarConfig(enabled=True)
        f = EconomicCalendarFilter(config)
        f._events = [_make_event("EUR", "HIGH", "ECB Rate", 15, now)]

        assert f.check_pair("EUR_USD", check_time=now).action == "BLACKOUT"
        assert f.check_pair("EUR_JPY", check_time=now).action == "BLACKOUT"
        assert f.check_pair("GBP_USD", check_time=now).action == "CLEAR"

    def test_event_far_away_clear(self):
        """Event 60min away is CLEAR (outside 30min blackout)."""
        now = datetime(2026, 3, 24, 14, 0, 0, tzinfo=timezone.utc)
        config = EconomicCalendarConfig(enabled=True)
        f = EconomicCalendarFilter(config)
        f._events = [_make_event("USD", "HIGH", "NFP", 60, now)]

        result = f.check_pair("EUR_USD", check_time=now)
        assert result.action == "CLEAR"

    def test_disabled_filter(self):
        """Disabled filter always returns CLEAR."""
        config = EconomicCalendarConfig(enabled=False)
        f = EconomicCalendarFilter(config)
        f._events = [_make_event("USD", "HIGH", "NFP", 5)]

        result = f.check_pair("EUR_USD")
        assert result.action == "CLEAR"
        assert result.reason == "economic_calendar_disabled"

    def test_no_events_loaded(self):
        """Empty event list returns CLEAR."""
        config = EconomicCalendarConfig(enabled=True, cache_file="/tmp/nonexistent_cal_cache.json")
        f = EconomicCalendarFilter(config)
        assert len(f._events) == 0

        result = f.check_pair("EUR_USD")
        assert result.action == "CLEAR"
        assert result.reason == "no_events_loaded"

    def test_high_overrides_medium(self):
        """HIGH blackout takes priority over MEDIUM reduce."""
        now = datetime(2026, 3, 24, 14, 0, 0, tzinfo=timezone.utc)
        config = EconomicCalendarConfig(enabled=True)
        f = EconomicCalendarFilter(config)
        f._events = [
            _make_event("USD", "HIGH", "NFP", 10, now),
            _make_event("USD", "MEDIUM", "PMI", 12, now),
        ]

        result = f.check_pair("EUR_USD", check_time=now)
        assert result.action == "BLACKOUT"
        assert result.size_multiplier == 0.0

    def test_cache_save_and_load(self):
        """Events persist to cache and reload correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "cal_cache.json")
            config = EconomicCalendarConfig(cache_file=cache_file, enabled=True)
            f = EconomicCalendarFilter(config)

            events = [
                EconomicEvent(
                    timestamp="2026-03-24T14:30:00+00:00",
                    currency="USD",
                    impact="HIGH",
                    event_name="NFP",
                ),
                EconomicEvent(
                    timestamp="2026-03-24T16:00:00+00:00",
                    currency="EUR",
                    impact="MEDIUM",
                    event_name="PMI",
                ),
            ]
            f.update_events(events)

            # Verify file exists
            assert os.path.exists(cache_file)

            # Load into new instance
            f2 = EconomicCalendarFilter(config)
            assert len(f2.events) == 2
            assert f2.events[0].event_name == "NFP"

    def test_cache_corrupted_graceful(self):
        """Corrupted cache file doesn't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "cal_cache.json")
            with open(cache_file, "w") as f:
                f.write("{not valid json")

            config = EconomicCalendarConfig(cache_file=cache_file)
            f = EconomicCalendarFilter(config)
            assert len(f.events) == 0  # Graceful fallback

    def test_add_events_dedup(self):
        """add_events doesn't duplicate existing events."""
        config = EconomicCalendarConfig(enabled=True)
        f = EconomicCalendarFilter(config)

        evt = EconomicEvent(
            timestamp="2026-03-24T14:30:00+00:00",
            currency="USD", impact="HIGH", event_name="NFP"
        )
        f.add_events([evt])
        f.add_events([evt])  # Same event again
        assert len(f.events) == 1

    def test_get_upcoming_events(self):
        """get_upcoming_events filters by time window."""
        now = datetime(2026, 3, 24, 14, 0, 0, tzinfo=timezone.utc)
        config = EconomicCalendarConfig(enabled=True)
        f = EconomicCalendarFilter(config)
        f._events = [
            _make_event("USD", "HIGH", "NFP", 60, now),   # 1h from now
            _make_event("EUR", "HIGH", "ECB", 300, now),   # 5h from now
            _make_event("GBP", "LOW", "BOE", -60, now),   # 1h ago
        ]

        upcoming = f.get_upcoming_events(hours_ahead=4, check_time=now)
        assert len(upcoming) == 1
        assert upcoming[0].event_name == "NFP"

    def test_is_cache_stale(self):
        """Cache staleness detection works."""
        config = EconomicCalendarConfig(cache_refresh_hours=4)
        f = EconomicCalendarFilter(config)
        f._cache_loaded_at = None
        assert f.is_cache_stale() is True

        f._cache_loaded_at = 0  # Very old
        assert f.is_cache_stale() is True

    def test_config_validation(self):
        """Invalid config values are caught."""
        with pytest.raises(ValueError):
            EconomicCalendarConfig(high_impact_blackout_minutes=-5).validate()
        with pytest.raises(ValueError):
            EconomicCalendarConfig(medium_impact_size_multiplier=1.5).validate()


class TestCalendarFilterFactory:
    """Test factory function."""

    def test_create_default(self):
        f = create_default_calendar_filter()
        assert isinstance(f, EconomicCalendarFilter)
        assert f.config.enabled is True

    def test_create_with_config(self):
        config = EconomicCalendarConfig(high_impact_blackout_minutes=45)
        f = create_default_calendar_filter(config=config)
        assert f.config.high_impact_blackout_minutes == 45
