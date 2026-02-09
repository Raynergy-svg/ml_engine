"""
News-based sentiment features and economic calendar integration.

This module provides:
1. News sentiment analysis using FinBERT (optional transformer dependency)
2. Economic calendar checking to avoid high-impact events
3. Post-trade model updates with incremental training
4. Configurable callback system for pluggable news/calendar data sources
5. RSS-based news fetcher adapter (requires optional feedparser dependency)

Dependencies:
- transformers (optional) - for FinBERT sentiment analysis
- feedparser (optional) - for RSS-based news fetching
- requests (optional) - for fetching external data

If transformers is not installed, falls back to simple sentiment from text_features.
If feedparser is not installed, RSS fetcher logs a warning and returns empty list.

Callback system:
    # Register a custom news fetcher:
    set_news_fetcher(my_custom_fetcher)

    # Register a custom calendar fetcher:
    set_calendar_fetcher(my_custom_calendar)

    # Or use the built-in RSS adapter (auto-registered when feedparser is available):
    from news_features import rss_forex_news_fetcher
    set_news_fetcher(rss_forex_news_fetcher)
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

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
    import transformers  # noqa: F401
    _HAS_TRANSFORMERS = True
except ImportError:
    logger.debug("transformers not installed, using simple sentiment fallback")

# Try to import feedparser for RSS-based news fetching
_HAS_FEEDPARSER = False
try:
    import feedparser  # noqa: F401
    _HAS_FEEDPARSER = True
except ImportError:
    logger.debug("feedparser not installed, RSS news fetcher will be unavailable")


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
# CONFIGURABLE CALLBACK SYSTEM
# =============================================================================

# Private module-level callback variables (default None = no fetcher configured)
_NEWS_FETCHER: Optional[Callable[[str, int], List[str]]] = None
_CALENDAR_FETCHER: Optional[Callable[[], List['EconomicEvent']]] = None


def set_news_fetcher(callback: Callable[[str, int], List[str]]) -> None:
    """
    Set the news fetcher callback function.

    The callback should accept (instrument: str, max_headlines: int)
    and return a List[str] of headline strings.

    Args:
        callback: Function that fetches news headlines for an instrument.

    Example:
        def my_fetcher(instrument, max_headlines=10):
            return ["USD rallies on strong jobs data"]
        set_news_fetcher(my_fetcher)
    """
    global _NEWS_FETCHER
    _NEWS_FETCHER = callback
    logger.info("News fetcher callback configured")


def get_news_fetcher() -> Optional[Callable[[str, int], List[str]]]:
    """
    Get the currently configured news fetcher callback.

    Returns:
        The news fetcher callback function, or None if not configured.
    """
    return _NEWS_FETCHER


def set_calendar_fetcher(callback: Callable[[], List['EconomicEvent']]) -> None:
    """
    Set the economic calendar fetcher callback function.

    The callback should accept no arguments and return a List[EconomicEvent].

    Args:
        callback: Function that fetches economic calendar events.
    """
    global _CALENDAR_FETCHER
    _CALENDAR_FETCHER = callback
    logger.info("Calendar fetcher callback configured")


def get_calendar_fetcher() -> Optional[Callable[[], List['EconomicEvent']]]:
    """
    Get the currently configured calendar fetcher callback.

    Returns:
        The calendar fetcher callback function, or None if not configured.
    """
    return _CALENDAR_FETCHER


# Backward-compatible public aliases – kept as module-level names so that
# legacy code doing ``news_features.NEWS_FETCHER = X`` still works.  The
# property-like getter/setter pair below keeps the private variable in sync.

def __getattr__(name: str):
    """Module-level __getattr__ for backward-compatible alias access."""
    if name == "NEWS_FETCHER":
        return _NEWS_FETCHER
    if name == "CALENDAR_FETCHER":
        return _CALENDAR_FETCHER
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_news_fetcher(fetcher: Callable[[str, int], List[str]]) -> None:
    """Register a custom news fetching function (backward-compatible alias)."""
    set_news_fetcher(fetcher)


def register_calendar_fetcher(fetcher: Callable[[], List['EconomicEvent']]) -> None:
    """Register a custom calendar fetching function (backward-compatible alias)."""
    set_calendar_fetcher(fetcher)


# =============================================================================
# RSS-BASED NEWS FETCHER ADAPTER
# =============================================================================

# Currency keyword mapping for filtering headlines by instrument relevance
_CURRENCY_KEYWORDS: Dict[str, List[str]] = {
    "USD": ["usd", "dollar", "fed", "fomc", "nfp", "cpi", "powell", "treasury",
            "payrolls", "inflation", "us economy"],
    "EUR": ["eur", "euro", "ecb", "lagarde", "eurozone", "european", "germany"],
    "GBP": ["gbp", "pound", "boe", "sterling", "bailey", "britain", "uk"],
    "JPY": ["jpy", "yen", "boj", "ueda", "japan", "japanese"],
    "CHF": ["chf", "franc", "snb", "swiss", "switzerland"],
    "CAD": ["cad", "loonie", "boc", "canada", "canadian"],
    "AUD": ["aud", "aussie", "rba", "australia", "australian"],
    "NZD": ["nzd", "kiwi", "rbnz", "new zealand"],
}

# Free RSS feeds for forex news (no API key required)
_DEFAULT_RSS_FEEDS: List[str] = [
    "https://www.forexlive.com/feed/news",
    "https://www.fxstreet.com/rss/news",
]


def rss_forex_news_fetcher(instrument: str, max_headlines: int = 10) -> List[str]:
    """
    Fetch forex news headlines from free RSS feeds.

    Uses feedparser library to parse RSS/Atom feeds and filters headlines
    by currency keywords derived from the instrument pair.

    If feedparser is not installed, logs a warning and returns an empty list.

    Args:
        instrument: Currency pair (e.g., 'EUR_USD')
        max_headlines: Maximum number of headlines to return

    Returns:
        List of news headline strings relevant to the instrument.
        Returns empty list if feedparser is unavailable or all feeds fail.
    """
    if not _HAS_FEEDPARSER:
        logger.warning(
            "feedparser not installed, RSS news fetcher returning empty list. "
            "Install with: pip install feedparser"
        )
        return []

    import feedparser as fp

    # Extract currencies from instrument (e.g., 'EUR_USD' -> ['EUR', 'USD'])
    parts = instrument.replace("/", "_").split("_")
    currencies = [p.upper() for p in parts if len(p) == 3]

    # Build keyword list from currency mapping
    keywords: List[str] = []
    for cur in currencies:
        keywords.extend(_CURRENCY_KEYWORDS.get(cur, [cur.lower()]))
    # Add the instrument itself as a keyword
    keywords.append(instrument.lower().replace("_", "/"))
    keywords.append(instrument.lower().replace("_", ""))

    headlines: List[str] = []
    seen: set = set()

    for feed_url in _DEFAULT_RSS_FEEDS:
        if len(headlines) >= max_headlines:
            break

        try:
            feed = fp.parse(feed_url)
            if feed.bozo and not feed.entries:
                logger.debug(f"RSS feed parse error for {feed_url}: {feed.bozo_exception}")
                continue

            for entry in feed.entries:
                title = getattr(entry, "title", "").strip()
                if not title:
                    continue

                title_lc = title.lower()

                # Check if headline matches any relevant keywords
                if any(k in title_lc for k in keywords):
                    if title_lc not in seen:
                        seen.add(title_lc)
                        headlines.append(title)

                if len(headlines) >= max_headlines:
                    break

        except Exception as e:
            logger.debug(f"RSS fetch failed for {feed_url}: {e}")
            continue

    logger.debug(
        f"rss_forex_news_fetcher({instrument}) -> "
        f"{len(headlines)} headlines from {len(_DEFAULT_RSS_FEEDS)} feeds"
    )
    return headlines


def _rss_fallback_fetcher(instrument: str, max_headlines: int = 10) -> List[str]:
    """
    Fallback RSS fetcher using stdlib urllib when feedparser is unavailable.

    Parses RSS XML manually using xml.etree.ElementTree. Less robust than
    feedparser but works with zero extra dependencies.

    Args:
        instrument: Currency pair (e.g., 'EUR_USD')
        max_headlines: Maximum number of headlines to return

    Returns:
        List of news headline strings, or empty list on failure.
    """
    import urllib.request

    parts = instrument.replace("/", "_").split("_")
    currencies = [p.upper() for p in parts if len(p) == 3]

    keywords: List[str] = []
    for cur in currencies:
        keywords.extend(_CURRENCY_KEYWORDS.get(cur, [cur.lower()]))
    keywords.append(instrument.lower().replace("_", "/"))
    keywords.append(instrument.lower().replace("_", ""))

    headlines: List[str] = []
    seen: set = set()

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }

    for feed_url in _DEFAULT_RSS_FEEDS:
        if len(headlines) >= max_headlines:
            break

        try:
            req = urllib.request.Request(feed_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
        except Exception as e:
            logger.debug(f"RSS stdlib fetch failed for {feed_url}: {e}")
            continue

        # Parse RSS 2.0 items
        items = root.findall(".//item")
        if not items:
            # Try Atom format
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        for item in items:
            title = (item.findtext("title") or "").strip()
            if not title:
                title_elem = item.find("{http://www.w3.org/2005/Atom}title")
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""

            if not title:
                continue

            title_lc = title.lower()
            if any(k in title_lc for k in keywords):
                if title_lc not in seen:
                    seen.add(title_lc)
                    headlines.append(title)

            if len(headlines) >= max_headlines:
                break

    logger.debug(
        f"_rss_fallback_fetcher({instrument}) -> "
        f"{len(headlines)} headlines (stdlib)"
    )
    return headlines


# =============================================================================
# NEWS DATA FETCHING
# =============================================================================

def fetch_forex_news(instrument: str, max_headlines: int = 10) -> List[str]:
    """
    Fetch recent news headlines for a forex instrument.

    Checks for a configured news fetcher callback first. If none is set,
    returns an empty list (safe default for Gate 6 sentiment evaluation).

    Args:
        instrument: Currency pair (e.g., 'EUR_USD')
        max_headlines: Maximum number of headlines to return

    Returns:
        List of news headline strings. Empty list when no fetcher is
        configured or when fetcher returns no results.

    Note:
        Configure a fetcher with set_news_fetcher() or register_news_fetcher().
        The built-in rss_forex_news_fetcher is auto-registered when feedparser
        is available.
    """
    fetcher = _NEWS_FETCHER
    if fetcher is not None:
        try:
            return fetcher(instrument, max_headlines)
        except Exception as e:
            logger.warning(f"News fetcher failed for {instrument}: {e}")
            return []

    # Default: return empty list (no news data available)
    logger.debug(f"No news fetcher configured for {instrument}")
    return []


def fetch_economic_events() -> List['EconomicEvent']:
    """
    Fetch upcoming economic calendar events.

    Checks for a configured calendar fetcher callback first. If none is set,
    delegates to fetch_forexfactory_calendar() which also returns an empty list
    by default.

    Returns:
        List of EconomicEvent objects. Empty list when no fetcher is
        configured or when fetcher returns no results.
    """
    fetcher = _CALENDAR_FETCHER
    if fetcher is not None:
        try:
            return fetcher()
        except Exception as e:
            logger.warning(f"Calendar fetcher failed: {e}")
            return []

    # Delegate to the existing forexfactory function for backward compatibility
    return fetch_forexfactory_calendar()


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

    This is a low-level placeholder that returns an empty list by default.
    Prefer using fetch_economic_events() which checks the callback system first.

    Override with actual implementation by calling set_calendar_fetcher().

    Returns:
        List of EconomicEvent objects

    Note:
        To implement actual calendar fetching:
        1. Scrape ForexFactory calendar (respect robots.txt)
        2. Use Investing.com API
        3. Use FinnHub economic calendar endpoint
        4. Use TradingEconomics API (paid)
    """
    # Check callback for backward compatibility (callers that bypass fetch_economic_events)
    fetcher = _CALENDAR_FETCHER
    if fetcher is not None:
        try:
            return fetcher()
        except Exception as e:
            logger.warning(f"Calendar fetcher failed in fetch_forexfactory_calendar: {e}")
            return []

    # Default: return empty list (no calendar data)
    logger.debug("No calendar data source configured")
    return []


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


# =============================================================================
# AUTO-REGISTRATION OF DEFAULT NEWS FETCHER
# =============================================================================
# When feedparser is available, automatically register the RSS-based news fetcher
# as the default. This can be overridden at any time by calling set_news_fetcher()
# with a different callback.

def _auto_register_default_fetcher() -> None:
    """
    Auto-register the best available news fetcher at module load time.

    Priority:
    1. rss_forex_news_fetcher (requires feedparser)
    2. _rss_fallback_fetcher (stdlib only, less robust)
    3. None (no fetcher - fetch_forex_news returns empty list)
    """
    global _NEWS_FETCHER
    if _NEWS_FETCHER is not None:
        # Already configured (e.g., by user code that imported before this runs)
        return

    if _HAS_FEEDPARSER:
        _NEWS_FETCHER = rss_forex_news_fetcher
        logger.debug("Auto-registered rss_forex_news_fetcher (feedparser available)")
    else:
        # Leave as None; fetch_forex_news will return empty list gracefully
        logger.debug(
            "No default news fetcher registered (feedparser not installed). "
            "Install feedparser for RSS news: pip install feedparser"
        )


# Run auto-registration on module import
_auto_register_default_fetcher()
