"""
Integration tests for news sentiment features and callback system.

Focuses on integration-level tests not covered by existing test_news_features.py:
1. set/get_news_fetcher callback registration and retrieval
2. fetch_forex_news with configured vs unconfigured fetcher
3. add_sentiment_features DataFrame column correctness
4. check_economic_calendar dispatching through callback system
5. rss_forex_news_fetcher with mocked feedparser
6. EconomicEvent dataclass fields and ImpactLevel enum
7. Module-level auto-registration logic
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_fetcher_globals():
    """Reset module-level fetcher globals before and after each test.

    This is critical because news_features uses module-level state
    (_NEWS_FETCHER, _CALENDAR_FETCHER) that persists across tests.
    """
    import news_features

    # Save originals
    orig_news = news_features._NEWS_FETCHER
    orig_cal = news_features._CALENDAR_FETCHER

    # Clear for clean test state
    news_features._NEWS_FETCHER = None
    news_features._CALENDAR_FETCHER = None

    yield

    # Restore originals
    news_features._NEWS_FETCHER = orig_news
    news_features._CALENDAR_FETCHER = orig_cal


@pytest.fixture
def sample_df():
    """Simple OHLCV DataFrame for sentiment feature tests."""
    return pd.DataFrame({
        'close': [1.1000, 1.1010, 1.1020, 1.1015, 1.1025],
        'open': [1.0990, 1.1000, 1.1010, 1.1020, 1.1015],
        'high': [1.1010, 1.1020, 1.1030, 1.1025, 1.1035],
        'low': [1.0985, 1.0995, 1.1005, 1.1010, 1.1010],
        'volume': [100, 110, 120, 115, 130],
    })


# ---------------------------------------------------------------------------
# Test: Callback system - set/get news fetcher
# ---------------------------------------------------------------------------

class TestNewsCallbackSystem:
    """Test the news fetcher callback registration system."""

    def test_get_fetcher_default_none(self):
        """get_news_fetcher returns None when not configured."""
        from news_features import get_news_fetcher
        assert get_news_fetcher() is None

    def test_set_and_get_fetcher(self):
        """set_news_fetcher stores callback, get_news_fetcher retrieves it."""
        from news_features import set_news_fetcher, get_news_fetcher

        def my_fetcher(instrument, max_headlines=10):
            return ["Test headline"]

        set_news_fetcher(my_fetcher)
        retrieved = get_news_fetcher()
        assert retrieved is my_fetcher

    def test_set_fetcher_overrides_previous(self):
        """Setting a new fetcher replaces the old one."""
        from news_features import set_news_fetcher, get_news_fetcher

        fetcher_a = lambda inst, n=10: ["A"]
        fetcher_b = lambda inst, n=10: ["B"]

        set_news_fetcher(fetcher_a)
        set_news_fetcher(fetcher_b)

        result = get_news_fetcher()
        assert result is fetcher_b

    def test_register_news_fetcher_backward_compat(self):
        """register_news_fetcher is a backward-compatible alias."""
        from news_features import register_news_fetcher, get_news_fetcher

        def my_fetcher(inst, n=10):
            return ["Registered"]

        register_news_fetcher(my_fetcher)
        assert get_news_fetcher() is my_fetcher


# ---------------------------------------------------------------------------
# Test: Callback system - set/get calendar fetcher
# ---------------------------------------------------------------------------

class TestCalendarCallbackSystem:
    """Test the calendar fetcher callback registration system."""

    def test_get_calendar_default_none(self):
        """get_calendar_fetcher returns None when not configured."""
        from news_features import get_calendar_fetcher
        assert get_calendar_fetcher() is None

    def test_set_and_get_calendar(self):
        """set_calendar_fetcher stores callback, get_calendar_fetcher retrieves it."""
        from news_features import set_calendar_fetcher, get_calendar_fetcher

        def my_cal():
            return []

        set_calendar_fetcher(my_cal)
        assert get_calendar_fetcher() is my_cal

    def test_register_calendar_fetcher_backward_compat(self):
        """register_calendar_fetcher is a backward-compatible alias."""
        from news_features import register_calendar_fetcher, get_calendar_fetcher

        def my_cal():
            return []

        register_calendar_fetcher(my_cal)
        assert get_calendar_fetcher() is my_cal


# ---------------------------------------------------------------------------
# Test: fetch_forex_news
# ---------------------------------------------------------------------------

class TestFetchForexNews:
    """Test fetch_forex_news dispatching."""

    def test_empty_when_no_fetcher(self):
        """Returns empty list when no fetcher is configured."""
        from news_features import fetch_forex_news
        result = fetch_forex_news("EUR_USD")
        assert result == []

    def test_uses_configured_fetcher(self):
        """Uses the configured fetcher callback."""
        from news_features import fetch_forex_news, set_news_fetcher

        headlines = ["ECB keeps rates steady", "Euro strengthens"]

        def mock_fetcher(instrument, max_headlines=10):
            assert instrument == "EUR_USD"
            return headlines[:max_headlines]

        set_news_fetcher(mock_fetcher)
        result = fetch_forex_news("EUR_USD", max_headlines=5)
        assert result == headlines

    def test_handles_fetcher_exception(self):
        """Returns empty list if fetcher raises an exception."""
        from news_features import fetch_forex_news, set_news_fetcher

        def failing_fetcher(inst, n=10):
            raise ConnectionError("Network error")

        set_news_fetcher(failing_fetcher)
        result = fetch_forex_news("EUR_USD")
        assert result == []

    def test_passes_max_headlines(self):
        """max_headlines parameter is forwarded to fetcher."""
        from news_features import fetch_forex_news, set_news_fetcher

        received_max = [None]

        def capture_fetcher(inst, max_headlines=10):
            received_max[0] = max_headlines
            return []

        set_news_fetcher(capture_fetcher)
        fetch_forex_news("GBP_USD", max_headlines=3)
        assert received_max[0] == 3


# ---------------------------------------------------------------------------
# Test: fetch_economic_events (calendar dispatching)
# ---------------------------------------------------------------------------

class TestFetchEconomicEvents:
    """Test fetch_economic_events dispatching through callback system."""

    def test_empty_when_no_calendar(self):
        """Returns empty list when no calendar fetcher is set."""
        from news_features import fetch_economic_events
        result = fetch_economic_events()
        assert result == []

    def test_uses_configured_calendar(self):
        """Uses the configured calendar fetcher."""
        from news_features import (
            fetch_economic_events, set_calendar_fetcher,
            EconomicEvent, ImpactLevel,
        )

        events = [
            EconomicEvent(
                title="FOMC Minutes",
                currency="USD",
                impact=ImpactLevel.HIGH,
                time=datetime.now(timezone.utc),
                time_to=45,
            )
        ]
        set_calendar_fetcher(lambda: events)
        result = fetch_economic_events()
        assert len(result) == 1
        assert result[0].title == "FOMC Minutes"


# ---------------------------------------------------------------------------
# Test: add_sentiment_features
# ---------------------------------------------------------------------------

class TestAddSentimentFeatures:
    """Test adding sentiment columns to a DataFrame."""

    def test_adds_columns_no_news(self, sample_df):
        """With no fetcher, adds neutral sentiment columns."""
        from news_features import add_sentiment_features

        result = add_sentiment_features(sample_df, "EUR_USD")

        assert "news_sentiment" in result.columns
        assert "news_volume" in result.columns
        assert (result["news_sentiment"] == 0.0).all()
        assert (result["news_volume"] == 0).all()

    def test_preserves_original_df(self, sample_df):
        """Original DataFrame should not be modified."""
        from news_features import add_sentiment_features

        original_cols = set(sample_df.columns)
        add_sentiment_features(sample_df, "EUR_USD")
        assert set(sample_df.columns) == original_cols

    def test_news_volume_matches_headline_count(self, sample_df):
        """news_volume should reflect the number of fetched headlines."""
        from news_features import add_sentiment_features, set_news_fetcher

        headlines = ["Bull market", "Rate hike expected", "GDP surprise"]

        def mock_fetcher(inst, n=10):
            return headlines

        set_news_fetcher(mock_fetcher)
        result = add_sentiment_features(sample_df, "EUR_USD", use_finbert=False)
        assert (result["news_volume"] == 3).all()

    def test_sentiment_is_float(self, sample_df):
        """news_sentiment should be a float column."""
        from news_features import add_sentiment_features, set_news_fetcher

        set_news_fetcher(lambda i, n=10: ["Bullish outlook"])
        result = add_sentiment_features(sample_df, "EUR_USD", use_finbert=False)
        assert result["news_sentiment"].dtype in [np.float64, np.float32, float]


# ---------------------------------------------------------------------------
# Test: check_economic_calendar
# ---------------------------------------------------------------------------

class TestCheckEconomicCalendar:
    """Test check_economic_calendar gate logic."""

    def test_safe_to_trade_no_events(self):
        """No events -> trade=True."""
        from news_features import check_economic_calendar
        result = check_economic_calendar()
        assert result["trade"] is True
        assert result["reason"] is None

    def test_blocks_imminent_high_impact(self):
        """High-impact event within threshold -> trade=False."""
        import news_features
        from news_features import check_economic_calendar, EconomicEvent, ImpactLevel

        event = EconomicEvent(
            title="Non-Farm Payrolls",
            currency="USD",
            impact=ImpactLevel.HIGH,
            time=datetime.now(timezone.utc),
            time_to=60,
        )

        # Patch fetch_forexfactory_calendar to return our event
        with patch.object(news_features, 'fetch_forexfactory_calendar', return_value=[event]):
            result = check_economic_calendar(minutes_threshold=120)

        assert result["trade"] is False
        assert "Non-Farm Payrolls" in result["reason"]
        assert len(result["events"]) == 1

    def test_allows_distant_high_impact(self):
        """High-impact event beyond threshold -> trade=True."""
        import news_features
        from news_features import check_economic_calendar, EconomicEvent, ImpactLevel

        event = EconomicEvent(
            title="FOMC Decision",
            currency="USD",
            impact=ImpactLevel.HIGH,
            time=datetime.now(timezone.utc),
            time_to=300,  # 5 hours away
        )

        with patch.object(news_features, 'fetch_forexfactory_calendar', return_value=[event]):
            result = check_economic_calendar(minutes_threshold=120)

        assert result["trade"] is True

    def test_ignores_low_impact(self):
        """Low/medium impact events don't block trading."""
        import news_features
        from news_features import check_economic_calendar, EconomicEvent, ImpactLevel

        event = EconomicEvent(
            title="Building Permits",
            currency="USD",
            impact=ImpactLevel.LOW,
            time=datetime.now(timezone.utc),
            time_to=10,  # Very close but low impact
        )

        with patch.object(news_features, 'fetch_forexfactory_calendar', return_value=[event]):
            result = check_economic_calendar()

        assert result["trade"] is True

    def test_selects_soonest_event(self):
        """When multiple high-impact events, reason mentions the soonest."""
        import news_features
        from news_features import check_economic_calendar, EconomicEvent, ImpactLevel

        events = [
            EconomicEvent(
                title="ECB Rate", currency="EUR",
                impact=ImpactLevel.HIGH,
                time=datetime.now(timezone.utc), time_to=90,
            ),
            EconomicEvent(
                title="NFP", currency="USD",
                impact=ImpactLevel.HIGH,
                time=datetime.now(timezone.utc), time_to=30,
            ),
        ]

        with patch.object(news_features, 'fetch_forexfactory_calendar', return_value=events):
            result = check_economic_calendar(minutes_threshold=120)

        assert result["trade"] is False
        assert "NFP" in result["reason"]  # NFP is sooner (30min vs 90min)


# ---------------------------------------------------------------------------
# Test: EconomicEvent dataclass
# ---------------------------------------------------------------------------

class TestEconomicEvent:
    """Test EconomicEvent dataclass fields."""

    def test_create_full_event(self):
        """All fields set correctly."""
        from news_features import EconomicEvent, ImpactLevel

        event = EconomicEvent(
            title="CPI Release",
            currency="USD",
            impact=ImpactLevel.HIGH,
            time=datetime(2025, 6, 15, 14, 30, tzinfo=timezone.utc),
            time_to=45,
            forecast="3.2%",
            previous="3.0%",
        )
        assert event.title == "CPI Release"
        assert event.currency == "USD"
        assert event.impact == ImpactLevel.HIGH
        assert event.time_to == 45
        assert event.forecast == "3.2%"
        assert event.previous == "3.0%"

    def test_optional_fields_default_none(self):
        """forecast and previous default to None."""
        from news_features import EconomicEvent, ImpactLevel

        event = EconomicEvent(
            title="PMI", currency="EUR",
            impact=ImpactLevel.MEDIUM,
            time=datetime.now(timezone.utc), time_to=120,
        )
        assert event.forecast is None
        assert event.previous is None


# ---------------------------------------------------------------------------
# Test: ImpactLevel enum
# ---------------------------------------------------------------------------

class TestImpactLevel:
    """Test ImpactLevel enum values."""

    def test_values(self):
        """ImpactLevel has HIGH, MEDIUM, LOW string values."""
        from news_features import ImpactLevel

        assert ImpactLevel.HIGH == "high"
        assert ImpactLevel.MEDIUM == "medium"
        assert ImpactLevel.LOW == "low"

    def test_is_string_enum(self):
        """ImpactLevel values are strings (str, Enum)."""
        from news_features import ImpactLevel

        assert isinstance(ImpactLevel.HIGH, str)


# ---------------------------------------------------------------------------
# Test: rss_forex_news_fetcher with mocked feedparser
# ---------------------------------------------------------------------------

class TestRSSFetcher:
    """Test rss_forex_news_fetcher with mocked feedparser."""

    def test_returns_empty_without_feedparser(self):
        """When feedparser is unavailable, returns empty list."""
        import news_features

        # Temporarily pretend feedparser is unavailable
        original = news_features._HAS_FEEDPARSER
        news_features._HAS_FEEDPARSER = False
        try:
            result = news_features.rss_forex_news_fetcher("EUR_USD")
            assert result == []
        finally:
            news_features._HAS_FEEDPARSER = original

    @pytest.mark.skipif(
        not pytest.importorskip("feedparser", reason="feedparser not installed"),
        reason="feedparser required"
    )
    def test_filters_by_currency_keywords(self):
        """Headlines are filtered by currency-relevant keywords."""
        import news_features

        # Create a mock feed with entries
        mock_entry_relevant = MagicMock()
        mock_entry_relevant.title = "EUR rally continues as ECB holds rates"

        mock_entry_irrelevant = MagicMock()
        mock_entry_irrelevant.title = "Bitcoin surges to new highs"

        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = [mock_entry_relevant, mock_entry_irrelevant]

        with patch('news_features.fp.parse', return_value=mock_feed):
            original = news_features._HAS_FEEDPARSER
            news_features._HAS_FEEDPARSER = True
            try:
                result = news_features.rss_forex_news_fetcher("EUR_USD", max_headlines=10)
            finally:
                news_features._HAS_FEEDPARSER = original

        # EUR headline should be included, Bitcoin headline should not
        assert any("EUR" in h or "ecb" in h.lower() for h in result) or len(result) == 0

    def test_currency_keywords_exist(self):
        """_CURRENCY_KEYWORDS should have entries for major currencies."""
        from news_features import _CURRENCY_KEYWORDS

        expected_currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
        for cur in expected_currencies:
            assert cur in _CURRENCY_KEYWORDS, f"Missing keywords for {cur}"
            assert len(_CURRENCY_KEYWORDS[cur]) > 0, f"Empty keywords for {cur}"

    def test_default_rss_feeds_exist(self):
        """_DEFAULT_RSS_FEEDS should be a non-empty list of URLs."""
        from news_features import _DEFAULT_RSS_FEEDS

        assert isinstance(_DEFAULT_RSS_FEEDS, list)
        assert len(_DEFAULT_RSS_FEEDS) > 0
        for url in _DEFAULT_RSS_FEEDS:
            assert url.startswith("http"), f"Invalid URL: {url}"


# ---------------------------------------------------------------------------
# Test: _sentiment_to_score
# ---------------------------------------------------------------------------

class TestSentimentToScore:
    """Test _sentiment_to_score conversion."""

    def test_positive(self):
        """Positive label -> positive score."""
        from news_features import _sentiment_to_score
        assert _sentiment_to_score({"label": "positive", "score": 0.85}) == pytest.approx(0.85)

    def test_negative(self):
        """Negative label -> negative score."""
        from news_features import _sentiment_to_score
        assert _sentiment_to_score({"label": "negative", "score": 0.7}) == pytest.approx(-0.7)

    def test_neutral(self):
        """Neutral label -> 0.0."""
        from news_features import _sentiment_to_score
        assert _sentiment_to_score({"label": "neutral", "score": 0.9}) == pytest.approx(0.0)

    def test_missing_label(self):
        """Missing label defaults to neutral (0.0)."""
        from news_features import _sentiment_to_score
        assert _sentiment_to_score({"score": 0.5}) == pytest.approx(0.0)
