"""Unit tests for news-based features and economic calendar."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import news_features
from news_features import (
    EconomicEvent,
    ImpactLevel,
    PostTradeUpdater,
    TradeFeatures,
    WIN_THRESHOLD,
    add_sentiment_features,
    analyze_sentiment_finbert,
    check_economic_calendar,
    fetch_forex_news,
    fetch_forexfactory_calendar,
    get_post_trade_updater,
    post_trade_update,
    register_calendar_fetcher,
    register_news_fetcher,
    _sentiment_to_score,
)


class TestSentimentAnalysis(unittest.TestCase):
    """Test sentiment analysis functionality."""

    def test_sentiment_to_score_positive(self):
        """Positive sentiment returns positive score."""
        result = {"label": "positive", "score": 0.9}
        self.assertAlmostEqual(_sentiment_to_score(result), 0.9)

    def test_sentiment_to_score_negative(self):
        """Negative sentiment returns negative score."""
        result = {"label": "negative", "score": 0.8}
        self.assertAlmostEqual(_sentiment_to_score(result), -0.8)

    def test_sentiment_to_score_neutral(self):
        """Neutral sentiment returns zero."""
        result = {"label": "neutral", "score": 0.6}
        self.assertAlmostEqual(_sentiment_to_score(result), 0.0)

    def test_sentiment_to_score_empty(self):
        """Empty result returns zero."""
        result = {}
        self.assertAlmostEqual(_sentiment_to_score(result), 0.0)


class TestNewsFetching(unittest.TestCase):
    """Test news fetching functionality."""

    def setUp(self):
        """Save original fetcher state."""
        self._original_news_fetcher = news_features.NEWS_FETCHER
    
    def tearDown(self):
        """Restore original fetcher state."""
        news_features.NEWS_FETCHER = self._original_news_fetcher

    def test_fetch_forex_news_default_empty(self):
        """Default fetcher returns empty list."""
        news_features.NEWS_FETCHER = None
        
        news = fetch_forex_news("EUR_USD")
        self.assertEqual(news, [])

    def test_register_news_fetcher(self):
        """Custom news fetcher can be registered."""
        mock_news = ["EUR rises on ECB decision", "Dollar weakens"]
        
        def mock_fetcher(instrument: str, max_headlines: int):
            return mock_news[:max_headlines]
        
        register_news_fetcher(mock_fetcher)
        
        news = fetch_forex_news("EUR_USD", max_headlines=2)
        self.assertEqual(news, mock_news)


class TestAddSentimentFeatures(unittest.TestCase):
    """Test add_sentiment_features function."""

    def setUp(self):
        """Create test dataframe and save original fetcher state."""
        self._original_news_fetcher = news_features.NEWS_FETCHER
        self.df = pd.DataFrame({
            "close": [1.1000, 1.1010, 1.1020],
            "open": [1.0990, 1.1000, 1.1010],
            "high": [1.1010, 1.1020, 1.1030],
            "low": [1.0985, 1.0995, 1.1005],
            "volume": [100, 110, 120],
        })
    
    def tearDown(self):
        """Restore original fetcher state."""
        news_features.NEWS_FETCHER = self._original_news_fetcher

    def test_add_sentiment_no_news(self):
        """When no news available, adds neutral features."""
        news_features.NEWS_FETCHER = None
        
        result = add_sentiment_features(self.df, "EUR_USD")
        
        self.assertIn("news_sentiment", result.columns)
        self.assertIn("news_volume", result.columns)
        self.assertEqual(result["news_sentiment"].iloc[0], 0.0)
        self.assertEqual(result["news_volume"].iloc[0], 0)

    def test_add_sentiment_with_news(self):
        """With news available, computes sentiment features."""
        # Register mock fetcher with positive news
        def mock_fetcher(instrument: str, max_headlines: int):
            return ["Bull market continues", "Strong growth ahead"]
        
        news_features.NEWS_FETCHER = mock_fetcher
        
        result = add_sentiment_features(self.df, "EUR_USD", use_finbert=False)
        
        self.assertIn("news_sentiment", result.columns)
        self.assertIn("news_volume", result.columns)
        self.assertEqual(result["news_volume"].iloc[0], 2)
        # Simple sentiment should return positive for bullish words
        self.assertGreater(result["news_sentiment"].iloc[0], 0.0)

    def test_add_sentiment_preserves_original(self):
        """Original dataframe is not modified."""
        news_features.NEWS_FETCHER = None
        
        original_cols = list(self.df.columns)
        result = add_sentiment_features(self.df, "EUR_USD")
        
        # Original unchanged
        self.assertEqual(list(self.df.columns), original_cols)
        # Result has new columns
        self.assertIn("news_sentiment", result.columns)


class TestEconomicCalendar(unittest.TestCase):
    """Test economic calendar functionality."""

    def setUp(self):
        """Save original calendar fetcher state."""
        self._original_calendar_fetcher = news_features.CALENDAR_FETCHER
    
    def tearDown(self):
        """Restore original calendar fetcher state."""
        news_features.CALENDAR_FETCHER = self._original_calendar_fetcher

    def test_fetch_calendar_default_empty(self):
        """Default calendar fetcher returns empty list."""
        news_features.CALENDAR_FETCHER = None
        
        events = fetch_forexfactory_calendar()
        self.assertEqual(events, [])

    def test_register_calendar_fetcher(self):
        """Custom calendar fetcher can be registered."""
        mock_events = [
            EconomicEvent(
                title="NFP",
                currency="USD",
                impact=ImpactLevel.HIGH,
                time=datetime.now(timezone.utc),
                time_to=30,
            )
        ]
        
        register_calendar_fetcher(lambda: mock_events)
        
        events = fetch_forexfactory_calendar()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].title, "NFP")

    def test_check_calendar_no_events(self):
        """No events means safe to trade."""
        news_features.CALENDAR_FETCHER = None
        
        result = check_economic_calendar()
        
        self.assertTrue(result["trade"])
        self.assertIsNone(result["reason"])
        self.assertEqual(result["events"], [])

    def test_check_calendar_high_impact_soon(self):
        """High-impact event soon means don't trade."""
        mock_events = [
            EconomicEvent(
                title="NFP",
                currency="USD",
                impact=ImpactLevel.HIGH,
                time=datetime.now(timezone.utc),
                time_to=30,  # 30 minutes away
            )
        ]
        
        news_features.CALENDAR_FETCHER = lambda: mock_events
        
        result = check_economic_calendar(minutes_threshold=120)
        
        self.assertFalse(result["trade"])
        self.assertIn("NFP", result["reason"])
        self.assertIn("30min", result["reason"])
        self.assertEqual(len(result["events"]), 1)

    def test_check_calendar_high_impact_far(self):
        """High-impact event far away means safe to trade."""
        mock_events = [
            EconomicEvent(
                title="NFP",
                currency="USD",
                impact=ImpactLevel.HIGH,
                time=datetime.now(timezone.utc),
                time_to=180,  # 3 hours away
            )
        ]
        
        news_features.CALENDAR_FETCHER = lambda: mock_events
        
        result = check_economic_calendar(minutes_threshold=120)  # 2 hour threshold
        
        self.assertTrue(result["trade"])
        self.assertIsNone(result["reason"])

    def test_check_calendar_low_impact_ignored(self):
        """Low-impact events are ignored."""
        mock_events = [
            EconomicEvent(
                title="Building Permits",
                currency="USD",
                impact=ImpactLevel.LOW,
                time=datetime.now(timezone.utc),
                time_to=10,  # Very close but low impact
            )
        ]
        
        news_features.CALENDAR_FETCHER = lambda: mock_events
        
        result = check_economic_calendar()
        
        self.assertTrue(result["trade"])


class TestPostTradeUpdater(unittest.TestCase):
    """Test PostTradeUpdater class."""

    def setUp(self):
        """Create test updater."""
        self.updater = PostTradeUpdater(
            update_interval=3,  # Train every 3 trades for testing
            incremental_epochs=2,
        )
        
        # Sample feature array
        self.features = np.random.randn(10, 5)  # 10 timesteps, 5 features

    def test_post_trade_update_stores_trade(self):
        """Trade result is stored."""
        result = self.updater.post_trade_update(
            trade_features=self.features,
            actual_outcome=1.0,
            instrument="EUR_USD",
            trade_id="test_001",
        )
        
        self.assertTrue(result["stored"])
        self.assertEqual(result["total_trades"], 1)
        self.assertFalse(result["trained"])

    def test_post_trade_update_triggers_training(self):
        """Training is triggered after interval trades."""
        train_called = [False]
        
        def mock_train(epochs):
            train_called[0] = True
            return {"loss": 0.5}
        
        self.updater.set_train_callback(mock_train)
        
        # First two trades - no training
        for i in range(2):
            result = self.updater.post_trade_update(
                trade_features=self.features,
                actual_outcome=float(i % 2),
                instrument="EUR_USD",
            )
            self.assertFalse(result["trained"])
        
        # Third trade - triggers training (interval=3)
        result = self.updater.post_trade_update(
            trade_features=self.features,
            actual_outcome=1.0,
            instrument="EUR_USD",
        )
        
        self.assertTrue(result["trained"])
        self.assertTrue(train_called[0])

    def test_get_recent_accuracy(self):
        """Recent accuracy is computed correctly."""
        # Add some trades
        self.updater.post_trade_update(self.features, 1.0, "EUR_USD")
        self.updater.post_trade_update(self.features, 1.0, "EUR_USD")
        self.updater.post_trade_update(self.features, 0.0, "EUR_USD")
        self.updater.post_trade_update(self.features, 1.0, "EUR_USD")
        
        # 3 wins out of 4 = 75%
        accuracy = self.updater.get_recent_accuracy(n=4)
        self.assertAlmostEqual(accuracy, 0.75)

    def test_clear_history(self):
        """History can be cleared."""
        self.updater.post_trade_update(self.features, 1.0, "EUR_USD")
        self.updater.post_trade_update(self.features, 0.0, "EUR_USD")
        
        self.updater.clear_history()
        
        # History cleared but total count preserved
        accuracy = self.updater.get_recent_accuracy()
        self.assertEqual(accuracy, 0.0)

    def test_with_replay_buffer(self):
        """Works with replay buffer integration."""
        mock_buffer = MagicMock()
        self.updater.set_replay_buffer(mock_buffer)
        
        self.updater.post_trade_update(
            trade_features=self.features,
            actual_outcome=1.0,
            instrument="EUR_USD",
        )
        
        # Verify replay buffer was called
        mock_buffer.add_samples.assert_called_once()


class TestGlobalPostTradeUpdater(unittest.TestCase):
    """Test global PostTradeUpdater singleton."""

    def setUp(self):
        """Save original updater state."""
        self._original_updater = news_features._post_trade_updater
    
    def tearDown(self):
        """Restore original updater state."""
        news_features._post_trade_updater = self._original_updater

    def test_get_post_trade_updater(self):
        """Global updater is accessible."""
        news_features._post_trade_updater = None  # Reset
        
        updater = get_post_trade_updater()
        self.assertIsInstance(updater, PostTradeUpdater)
        
        # Same instance returned
        updater2 = get_post_trade_updater()
        self.assertIs(updater, updater2)

    def test_post_trade_update_convenience(self):
        """Convenience function works."""
        news_features._post_trade_updater = None  # Reset
        
        features = np.random.randn(10, 5)
        result = post_trade_update(
            trade_features=features,
            actual_outcome=1.0,
            instrument="GBP_USD",
        )
        
        self.assertTrue(result["stored"])


class TestEconomicEventDataclass(unittest.TestCase):
    """Test EconomicEvent dataclass."""

    def test_create_event(self):
        """Can create economic event."""
        event = EconomicEvent(
            title="Federal Reserve Interest Rate Decision",
            currency="USD",
            impact=ImpactLevel.HIGH,
            time=datetime(2025, 1, 15, 19, 0, tzinfo=timezone.utc),
            time_to=60,
            forecast="5.25%",
            previous="5.00%",
        )
        
        self.assertEqual(event.title, "Federal Reserve Interest Rate Decision")
        self.assertEqual(event.currency, "USD")
        self.assertEqual(event.impact, ImpactLevel.HIGH)
        self.assertEqual(event.time_to, 60)
        self.assertEqual(event.forecast, "5.25%")


class TestTradeFeatures(unittest.TestCase):
    """Test TradeFeatures dataclass."""

    def test_create_trade_features(self):
        """Can create trade features."""
        X = np.random.randn(50, 10)
        tf = TradeFeatures(
            X=X,
            y=1.0,
            instrument="EUR_USD",
            timestamp="2025-01-15T10:00:00Z",
            trade_id="T001",
        )
        
        self.assertEqual(tf.y, 1.0)
        self.assertEqual(tf.instrument, "EUR_USD")
        self.assertEqual(tf.trade_id, "T001")
        np.testing.assert_array_equal(tf.X, X)


if __name__ == "__main__":
    unittest.main()
