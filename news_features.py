"""
News-based sentiment features and economic calendar integration.

This module provides:
1. News sentiment analysis using FinBERT (optional transformer dependency)
2. Economic calendar checking to avoid high-impact events
3. Post-trade model updates with incremental training

Dependencies:
- transformers (optional) - for FinBERT sentiment analysis
- requests (optional) - for fetching external data

If transformers is not installed, falls back to simple sentiment from text_features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Impact levels for economic events
class ImpactLevel(str, Enum):
    """Economic event impact levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Threshold for considering a trade as a win (0.5 = anything >= 0.5 is a win)
WIN_THRESHOLD = 0.5

# Default time threshold in minutes for avoiding high-impact events
DEFAULT_EVENT_THRESHOLD_MINUTES = 120


# Try to import transformers for FinBERT sentiment analysis
_HAS_TRANSFORMERS = False
_sentiment_pipeline = None

try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except ImportError:
    logger.debug("transformers not installed, using simple sentiment fallback")


def _get_sentiment_pipeline():
    """Lazy-load the FinBERT sentiment pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None and _HAS_TRANSFORMERS:
        try:
            from transformers import pipeline
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
            )
            logger.info("Loaded FinBERT sentiment model")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}")
    return _sentiment_pipeline


# =============================================================================
# NEWS DATA FETCHING (Placeholder implementations)
# =============================================================================

def fetch_forex_news(instrument: str, max_headlines: int = 10) -> List[str]:
    """
    Fetch recent news headlines for a forex instrument.
    
    This is a placeholder that returns empty list by default.
    Override with actual implementation by setting `NEWS_FETCHER` callback.
    
    Args:
        instrument: Currency pair (e.g., 'EUR_USD')
        max_headlines: Maximum number of headlines to return
        
    Returns:
        List of news headline strings
    
    Note:
        To implement actual news fetching, you can:
        1. Use ForexFactory RSS feeds
        2. Use Reuters API (requires subscription)
        3. Use NewsAPI.org with forex keywords
        4. Use FinnHub API for market news
    """
    if NEWS_FETCHER is not None:
        return NEWS_FETCHER(instrument, max_headlines)
    
    # Default: return empty list (no news data available)
    logger.debug(f"No news fetcher configured for {instrument}")
    return []


# Callback for custom news fetcher implementation
NEWS_FETCHER: Optional[Callable[[str, int], List[str]]] = None


def register_news_fetcher(fetcher: Callable[[str, int], List[str]]) -> None:
    """Register a custom news fetching function."""
    global NEWS_FETCHER
    NEWS_FETCHER = fetcher
    logger.info("Registered custom news fetcher")


# =============================================================================
# SENTIMENT ANALYSIS
# =============================================================================

def analyze_sentiment_finbert(headlines: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze sentiment using FinBERT (financial domain BERT).
    
    Args:
        headlines: List of news headline strings
        
    Returns:
        List of dicts with 'label' ('positive', 'negative', 'neutral') and 'score'
    """
    pipeline_fn = _get_sentiment_pipeline()
    
    if pipeline_fn is None or not headlines:
        return []
    
    try:
        results = pipeline_fn(headlines)
        # Handle both single and batch results
        if isinstance(results, dict):
            results = [results]
        return results
    except Exception as e:
        logger.warning(f"FinBERT sentiment analysis failed: {e}")
        return []


def _sentiment_to_score(result: Dict[str, Any]) -> float:
    """Convert FinBERT result to numeric score in [-1, 1]."""
    label = result.get("label", "neutral").lower()
    score = result.get("score", 0.5)
    
    if label == "positive":
        return score  # 0 to 1
    elif label == "negative":
        return -score  # -1 to 0
    else:  # neutral
        return 0.0


def add_sentiment_features(
    df: pd.DataFrame,
    instrument: str,
    use_finbert: bool = True,
) -> pd.DataFrame:
    """
    Add news sentiment features to a dataframe.
    
    Fetches recent news for the instrument and computes:
    - news_sentiment: Mean sentiment score [-1, 1]
    - news_volume: Number of news items (activity = volatility signal)
    
    Args:
        df: DataFrame to add features to
        instrument: Currency pair (e.g., 'EUR_USD')
        use_finbert: Use FinBERT if available, else simple sentiment
        
    Returns:
        DataFrame with news_sentiment and news_volume columns added
    """
    df = df.copy()
    
    # Fetch news headlines
    news = fetch_forex_news(instrument)
    
    if not news:
        # No news available - use neutral defaults
        df["news_sentiment"] = 0.0
        df["news_volume"] = 0
        return df
    
    # Compute sentiment scores
    if use_finbert and _HAS_TRANSFORMERS:
        results = analyze_sentiment_finbert(news)
        if results:
            scores = [_sentiment_to_score(r) for r in results]
        else:
            # Fallback to simple sentiment
            from src.utils.text_features import simple_sentiment_score
            scores = [simple_sentiment_score(h) for h in news]
    else:
        # Use simple lexicon-based sentiment
        from src.utils.text_features import simple_sentiment_score
        scores = [simple_sentiment_score(h) for h in news]
    
    # Compute aggregate features
    mean_sentiment = float(np.mean(scores)) if scores else 0.0
    news_count = len(news)
    
    # Add as constant columns (same value for all rows)
    # In real-time use, this would be updated per-bar
    df["news_sentiment"] = mean_sentiment
    df["news_volume"] = news_count
    
    logger.info(
        f"Added sentiment features for {instrument}: "
        f"sentiment={mean_sentiment:.3f}, volume={news_count}"
    )
    
    return df


# =============================================================================
# ECONOMIC CALENDAR
# =============================================================================

@dataclass
class EconomicEvent:
    """Represents an economic calendar event."""
    title: str
    currency: str
    impact: str  # 'high', 'medium', 'low' - use ImpactLevel enum values
    time: datetime
    time_to: int  # Minutes until event
    forecast: Optional[str] = None
    previous: Optional[str] = None


def fetch_forexfactory_calendar() -> List[EconomicEvent]:
    """
    Fetch economic calendar events.
    
    This is a placeholder that returns empty list by default.
    Override with actual implementation by setting `CALENDAR_FETCHER` callback.
    
    Returns:
        List of EconomicEvent objects
        
    Note:
        To implement actual calendar fetching:
        1. Scrape ForexFactory calendar (respect robots.txt)
        2. Use Investing.com API
        3. Use FinnHub economic calendar endpoint
        4. Use TradingEconomics API (paid)
    """
    if CALENDAR_FETCHER is not None:
        return CALENDAR_FETCHER()
    
    # Default: return empty list (no calendar data)
    logger.debug("No calendar fetcher configured")
    return []


# Callback for custom calendar fetcher implementation
CALENDAR_FETCHER: Optional[Callable[[], List[EconomicEvent]]] = None


def register_calendar_fetcher(fetcher: Callable[[], List[EconomicEvent]]) -> None:
    """Register a custom economic calendar fetching function."""
    global CALENDAR_FETCHER
    CALENDAR_FETCHER = fetcher
    logger.info("Registered custom calendar fetcher")


def check_economic_calendar(
    minutes_threshold: int = DEFAULT_EVENT_THRESHOLD_MINUTES,
) -> Dict[str, Any]:
    """
    Check economic calendar for imminent high-impact events.
    
    High-impact events (NFP, FOMC, ECB rates) cause extreme volatility
    and unpredictable price action. Better to avoid trading around them.
    
    Args:
        minutes_threshold: Minutes before event to start avoiding (default: 2 hours)
        
    Returns:
        Dict with:
        - 'trade': bool - True if safe to trade, False if should avoid
        - 'reason': str - Explanation if trade=False
        - 'events': List[EconomicEvent] - Upcoming high-impact events
    """
    events = fetch_forexfactory_calendar()
    
    if not events:
        return {
            "trade": True,
            "reason": None,
            "events": [],
        }
    
    # Filter for high-impact events within threshold
    high_impact = [
        e for e in events
        if e.impact == ImpactLevel.HIGH and 0 <= e.time_to < minutes_threshold
    ]
    
    if high_impact:
        # Sort by time_to to get the soonest event
        high_impact.sort(key=lambda e: e.time_to)
        soonest = high_impact[0]
        
        return {
            "trade": False,
            "reason": f"{soonest.title} in {soonest.time_to}min",
            "events": high_impact,
        }
    
    return {
        "trade": True,
        "reason": None,
        "events": [],
    }


# =============================================================================
# POST-TRADE MODEL UPDATES
# =============================================================================

@dataclass
class TradeFeatures:
    """Features associated with a trade for replay buffer."""
    X: np.ndarray  # Input features at trade entry
    y: float  # Actual outcome (1.0 for win, 0.0 for loss)
    instrument: str
    timestamp: str
    trade_id: Optional[str] = None


class PostTradeUpdater:
    """
    Manages post-trade model updates with incremental training.
    
    After each trade closes, stores the trade features and actual outcome.
    Every N trades, triggers incremental training to adapt to recent market.
    
    This implements online learning to keep the model current without
    full retraining.
    """
    
    def __init__(
        self,
        replay_buffer=None,
        update_interval: int = 100,
        incremental_epochs: int = 5,
    ):
        """
        Initialize post-trade updater.
        
        Args:
            replay_buffer: ReplayBuffer instance from modular_trainers
            update_interval: Number of trades between incremental updates
            incremental_epochs: Number of epochs for quick update
        """
        self.replay_buffer = replay_buffer
        self.update_interval = update_interval
        self.incremental_epochs = incremental_epochs
        
        # In-memory storage for trade results
        self._trade_results: List[TradeFeatures] = []
        self._total_trades = 0
        
        # Callback for incremental training
        self._train_callback: Optional[Callable[[int], None]] = None
    
    def set_replay_buffer(self, replay_buffer) -> None:
        """Set or update the replay buffer reference."""
        self.replay_buffer = replay_buffer
    
    def set_train_callback(
        self,
        callback: Callable[[int], None],
    ) -> None:
        """
        Set callback function for incremental training.
        
        Args:
            callback: Function that takes epochs as argument
        """
        self._train_callback = callback
    
    def post_trade_update(
        self,
        trade_features: np.ndarray,
        actual_outcome: float,
        instrument: str,
        trade_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update after a trade closes.
        
        Stores the trade result and triggers incremental training
        every `update_interval` trades.
        
        Args:
            trade_features: Feature array at trade entry time
            actual_outcome: 1.0 for profitable trade, 0.0 for loss
            instrument: Currency pair
            trade_id: Optional trade identifier
            
        Returns:
            Dict with status and any training metrics
        """
        # Store trade result
        trade = TradeFeatures(
            X=trade_features,
            y=actual_outcome,
            instrument=instrument,
            timestamp=datetime.now(timezone.utc).isoformat(),
            trade_id=trade_id,
        )
        self._trade_results.append(trade)
        self._total_trades += 1
        
        result = {
            "stored": True,
            "total_trades": self._total_trades,
            "trained": False,
            "metrics": None,
        }
        
        # Add to replay buffer if available
        if self.replay_buffer is not None:
            try:
                # Convert to numpy arrays for buffer
                X = np.expand_dims(trade_features, axis=0)
                y = np.array([actual_outcome])
                
                self.replay_buffer.add_samples(
                    X=X,
                    y=y,
                    data_id=f"{instrument}_{trade.timestamp}",
                )
            except Exception as e:
                logger.warning(f"Failed to add to replay buffer: {e}")
        
        # Check if we should trigger incremental training
        if self._total_trades % self.update_interval == 0:
            result["trained"] = True
            result["metrics"] = self.incremental_train(self.incremental_epochs)
        
        return result
    
    def incremental_train(self, epochs: int = 5) -> Optional[Dict[str, Any]]:
        """
        Perform quick incremental training on recent trades.
        
        Uses low learning rate and few epochs to avoid catastrophic
        forgetting while adapting to recent market conditions.
        
        Args:
            epochs: Number of training epochs (default: 5)
            
        Returns:
            Training metrics dict or None if no callback set
        """
        logger.info(
            f"Triggering incremental training: {epochs} epochs "
            f"on {len(self._trade_results)} recent trades"
        )
        
        if self._train_callback is not None:
            try:
                return self._train_callback(epochs)
            except Exception as e:
                logger.error(f"Incremental training failed: {e}")
                return {"error": str(e)}
        else:
            logger.warning("No training callback set, skipping incremental train")
            return None
    
    def get_recent_accuracy(self, n: int = 100) -> float:
        """Get win rate of most recent N trades."""
        if not self._trade_results:
            return 0.0
        
        recent = self._trade_results[-n:]
        wins = sum(1 for t in recent if t.y >= WIN_THRESHOLD)
        return wins / len(recent)
    
    def clear_history(self) -> None:
        """Clear stored trade history (keeps total count)."""
        self._trade_results = []
        logger.info("Cleared trade history")


# Singleton instance for global access
_post_trade_updater: Optional[PostTradeUpdater] = None


def get_post_trade_updater() -> PostTradeUpdater:
    """Get or create the global PostTradeUpdater instance."""
    global _post_trade_updater
    if _post_trade_updater is None:
        _post_trade_updater = PostTradeUpdater()
    return _post_trade_updater


def post_trade_update(
    trade_features: np.ndarray,
    actual_outcome: float,
    instrument: str = "EUR_USD",
    trade_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to update after a trade closes.
    
    Uses the global PostTradeUpdater instance.
    """
    updater = get_post_trade_updater()
    return updater.post_trade_update(
        trade_features=trade_features,
        actual_outcome=actual_outcome,
        instrument=instrument,
        trade_id=trade_id,
    )
