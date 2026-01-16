"""
Market Intelligence Module - News Sentiment, Economic Calendar, Online Learning

Provides:
1. News sentiment analysis using FinBERT
2. Economic calendar integration (ForexFactory, Investing.com)
3. Online/incremental learning from trade outcomes

Author: ML Engine
"""

from __future__ import annotations

import logging
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# NEWS SENTIMENT ANALYSIS
# =============================================================================

@dataclass
class NewsItem:
    """Single news item with sentiment."""
    headline: str
    source: str
    timestamp: datetime
    sentiment_score: float  # -1 (bearish) to +1 (bullish)
    sentiment_label: str  # 'bullish', 'bearish', 'neutral'
    relevance: float  # 0-1 relevance to instrument
    url: Optional[str] = None


class NewsSentimentAnalyzer:
    """
    Analyze news sentiment for FX instruments using FinBERT.
    
    Uses ProsusAI/finbert for financial sentiment analysis.
    Falls back to VADER if transformers not available.
    """
    
    def __init__(self, cache_dir: str = "trained_data/sentiment_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._use_finbert = False
        self._initialized = False
    
    def _initialize(self):
        """Lazy load sentiment model."""
        if self._initialized:
            return
        
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            
            # Try FinBERT first (best for financial text)
            try:
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device=-1,  # CPU (use 0 for GPU)
                )
                self._use_finbert = True
                logger.info("✓ FinBERT sentiment model loaded")
            except Exception as e:
                # Fallback to distilbert
                logger.warning(f"FinBERT not available ({e}), using distilbert")
                self._pipeline = pipeline("sentiment-analysis", device=-1)
                self._use_finbert = False
            
            self._initialized = True
            
        except ImportError:
            logger.warning("transformers not installed, using VADER fallback")
            try:
                import nltk
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                nltk.download('vader_lexicon', quiet=True)
                self._vader = SentimentIntensityAnalyzer()
                self._use_finbert = False
                self._initialized = True
            except ImportError:
                logger.error("Neither transformers nor nltk available for sentiment")
                self._initialized = True  # Mark as initialized but disabled
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.
        
        Returns:
            {
                'score': float (-1 to 1),
                'label': str ('bullish', 'bearish', 'neutral'),
                'confidence': float (0 to 1)
            }
        """
        self._initialize()
        
        if self._pipeline is not None:
            try:
                result = self._pipeline(text[:512])[0]  # Truncate for model limits
                
                if self._use_finbert:
                    # FinBERT labels: positive, negative, neutral
                    label_map = {'positive': 'bullish', 'negative': 'bearish', 'neutral': 'neutral'}
                    score_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
                    label = label_map.get(result['label'].lower(), 'neutral')
                    base_score = score_map.get(result['label'].lower(), 0.0)
                    score = base_score * result['score']  # Scale by confidence
                else:
                    # Generic sentiment model
                    label = 'bullish' if result['label'] == 'POSITIVE' else 'bearish'
                    score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
                
                return {
                    'score': score,
                    'label': label,
                    'confidence': result['score'],
                }
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        
        # VADER fallback
        if hasattr(self, '_vader'):
            scores = self._vader.polarity_scores(text)
            compound = scores['compound']
            if compound >= 0.05:
                label = 'bullish'
            elif compound <= -0.05:
                label = 'bearish'
            else:
                label = 'neutral'
            return {
                'score': compound,
                'label': label,
                'confidence': abs(compound),
            }
        
        # No model available
        return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple texts efficiently."""
        return [self.analyze(t) for t in texts]
    
    def get_instrument_sentiment(
        self,
        instrument: str,
        headlines: List[str],
        lookback_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get aggregate sentiment for an instrument from news headlines.
        
        Returns:
            {
                'aggregate_score': float (-1 to 1),
                'aggregate_label': str,
                'num_headlines': int,
                'bullish_count': int,
                'bearish_count': int,
                'neutral_count': int,
                'confidence': float,
            }
        """
        if not headlines:
            return {
                'aggregate_score': 0.0,
                'aggregate_label': 'neutral',
                'num_headlines': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'confidence': 0.0,
            }
        
        results = self.analyze_batch(headlines)
        
        scores = [r['score'] for r in results]
        labels = [r['label'] for r in results]
        
        aggregate_score = np.mean(scores)
        bullish = labels.count('bullish')
        bearish = labels.count('bearish')
        neutral = labels.count('neutral')
        
        if aggregate_score >= 0.1:
            aggregate_label = 'bullish'
        elif aggregate_score <= -0.1:
            aggregate_label = 'bearish'
        else:
            aggregate_label = 'neutral'
        
        # Confidence based on agreement
        max_count = max(bullish, bearish, neutral)
        confidence = max_count / len(headlines) if headlines else 0.0
        
        return {
            'aggregate_score': float(aggregate_score),
            'aggregate_label': aggregate_label,
            'num_headlines': len(headlines),
            'bullish_count': bullish,
            'bearish_count': bearish,
            'neutral_count': neutral,
            'confidence': float(confidence),
        }


# =============================================================================
# ECONOMIC CALENDAR
# =============================================================================

@dataclass
class EconomicEvent:
    """Single economic calendar event."""
    name: str
    currency: str  # e.g., 'USD', 'EUR'
    impact: str  # 'high', 'medium', 'low'
    time: datetime
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None
    
    @property
    def is_high_impact(self) -> bool:
        return self.impact.lower() == 'high'
    
    @property
    def minutes_until(self) -> float:
        """Minutes until event (negative if past)."""
        return (self.time - datetime.utcnow()).total_seconds() / 60


class EconomicCalendar:
    """
    Economic calendar for trading decision support.
    
    Fetches high-impact events that may affect FX pairs.
    """
    
    # High-impact events to watch
    HIGH_IMPACT_EVENTS = {
        'USD': ['NFP', 'FOMC', 'CPI', 'GDP', 'Retail Sales', 'ISM'],
        'EUR': ['ECB', 'CPI', 'GDP', 'PMI', 'German'],
        'GBP': ['BOE', 'CPI', 'GDP', 'PMI', 'Employment'],
        'JPY': ['BOJ', 'CPI', 'GDP', 'Tankan'],
        'CHF': ['SNB', 'CPI', 'GDP'],
        'CAD': ['BOC', 'CPI', 'GDP', 'Employment', 'Retail'],
        'AUD': ['RBA', 'CPI', 'GDP', 'Employment'],
        'NZD': ['RBNZ', 'CPI', 'GDP'],
    }
    
    def __init__(self, cache_dir: str = "trained_data/calendar_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._events_cache: Dict[str, List[EconomicEvent]] = {}
        self._cache_time: Optional[datetime] = None
    
    def _get_currencies_for_instrument(self, instrument: str) -> List[str]:
        """Extract currencies from instrument (e.g., 'USD_JPY' -> ['USD', 'JPY'])."""
        parts = instrument.replace('/', '_').split('_')
        return [p.upper() for p in parts if len(p) == 3]
    
    def fetch_events(
        self,
        instrument: str,
        hours_ahead: int = 24,
        force_refresh: bool = False,
    ) -> List[EconomicEvent]:
        """
        Fetch upcoming economic events for an instrument.
        
        Args:
            instrument: FX pair (e.g., 'USD_JPY')
            hours_ahead: Hours to look ahead
            force_refresh: Force API refresh
        
        Returns:
            List of EconomicEvent objects
        """
        currencies = self._get_currencies_for_instrument(instrument)
        
        # Check cache (valid for 1 hour)
        cache_key = f"{instrument}_{hours_ahead}"
        if not force_refresh and self._cache_time:
            cache_age = (datetime.utcnow() - self._cache_time).total_seconds() / 3600
            if cache_age < 1.0 and cache_key in self._events_cache:
                return self._events_cache[cache_key]
        
        events = []
        
        # Try to fetch from ForexFactory API (or cached file)
        try:
            events = self._fetch_forexfactory(currencies, hours_ahead)
        except Exception as e:
            logger.warning(f"ForexFactory fetch failed: {e}")
        
        # If no events fetched, try cached file
        if not events:
            events = self._load_cached_events(currencies, hours_ahead)
        
        # Cache results
        self._events_cache[cache_key] = events
        self._cache_time = datetime.utcnow()
        
        return events
    
    def _fetch_forexfactory(
        self,
        currencies: List[str],
        hours_ahead: int,
    ) -> List[EconomicEvent]:
        """Fetch from ForexFactory (requires web scraping or API)."""
        # ForexFactory doesn't have a public API - would need scraping
        # For now, return empty and rely on cached/manual data
        return []
    
    def _load_cached_events(
        self,
        currencies: List[str],
        hours_ahead: int,
    ) -> List[EconomicEvent]:
        """Load events from local cache file."""
        cache_file = self.cache_dir / "economic_events.json"
        
        if not cache_file.exists():
            return []
        
        try:
            with open(cache_file) as f:
                data = json.load(f)
            
            now = datetime.utcnow()
            cutoff = now + timedelta(hours=hours_ahead)
            
            events = []
            for item in data.get('events', []):
                event_time = datetime.fromisoformat(item['time'].replace('Z', ''))
                if now <= event_time <= cutoff:
                    if item['currency'] in currencies:
                        events.append(EconomicEvent(
                            name=item['name'],
                            currency=item['currency'],
                            impact=item['impact'],
                            time=event_time,
                            actual=item.get('actual'),
                            forecast=item.get('forecast'),
                            previous=item.get('previous'),
                        ))
            
            return sorted(events, key=lambda e: e.time)
            
        except Exception as e:
            logger.warning(f"Failed to load cached events: {e}")
            return []
    
    def check_trade_safety(
        self,
        instrument: str,
        min_minutes_before: int = 30,
        min_minutes_after: int = 15,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if it's safe to trade (no high-impact events imminent).
        
        Args:
            instrument: FX pair
            min_minutes_before: Don't trade N minutes before event
            min_minutes_after: Don't trade N minutes after event
        
        Returns:
            (is_safe, reason) - True if safe to trade, reason if not
        """
        events = self.fetch_events(instrument, hours_ahead=2)
        
        for event in events:
            if not event.is_high_impact:
                continue
            
            minutes = event.minutes_until
            
            # Event is imminent
            if -min_minutes_after <= minutes <= min_minutes_before:
                if minutes > 0:
                    reason = f"High-impact event '{event.name}' ({event.currency}) in {int(minutes)} minutes"
                else:
                    reason = f"High-impact event '{event.name}' ({event.currency}) just occurred {int(-minutes)} minutes ago"
                return False, reason
        
        return True, None
    
    def get_next_high_impact(self, instrument: str) -> Optional[EconomicEvent]:
        """Get the next high-impact event for an instrument."""
        events = self.fetch_events(instrument, hours_ahead=24)
        for event in events:
            if event.is_high_impact and event.minutes_until > 0:
                return event
        return None


# =============================================================================
# ONLINE LEARNING (INCREMENTAL UPDATES)
# =============================================================================

@dataclass
class TradeOutcome:
    """Record of a completed trade for learning."""
    trade_id: str
    instrument: str
    direction: int  # 1=long, 0=short
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl_pips: float
    pnl_percent: float
    features: np.ndarray  # Input features at entry
    prediction: float  # Model's prediction at entry
    confidence: float  # Model's confidence at entry
    actual_outcome: int  # 1=profitable, 0=loss
    
    @property
    def was_correct(self) -> bool:
        """Was the prediction correct?"""
        predicted_dir = 1 if self.prediction > 0.5 else 0
        actual_dir = 1 if self.pnl_pips > 0 else 0
        return predicted_dir == actual_dir


class OnlineLearner:
    """
    Online/incremental learning from trade outcomes.
    
    Accumulates trade results and periodically retrains the model
    on recent experiences to adapt to changing market conditions.
    """
    
    def __init__(
        self,
        buffer_size: int = 500,
        retrain_threshold: int = 50,  # Retrain after N new trades
        storage_dir: str = "trained_data/online_learning",
    ):
        self.buffer_size = buffer_size
        self.retrain_threshold = retrain_threshold
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.trade_buffer: List[TradeOutcome] = []
        self.trades_since_retrain = 0
        self._load_buffer()
    
    def _load_buffer(self):
        """Load trade buffer from disk."""
        buffer_file = self.storage_dir / "trade_buffer.json"
        if buffer_file.exists():
            try:
                with open(buffer_file) as f:
                    data = json.load(f)
                
                self.trades_since_retrain = data.get('trades_since_retrain', 0)
                
                for item in data.get('trades', [])[-self.buffer_size:]:
                    self.trade_buffer.append(TradeOutcome(
                        trade_id=item['trade_id'],
                        instrument=item['instrument'],
                        direction=item['direction'],
                        entry_time=datetime.fromisoformat(item['entry_time']),
                        exit_time=datetime.fromisoformat(item['exit_time']),
                        entry_price=item['entry_price'],
                        exit_price=item['exit_price'],
                        pnl_pips=item['pnl_pips'],
                        pnl_percent=item['pnl_percent'],
                        features=np.array(item['features']),
                        prediction=item['prediction'],
                        confidence=item['confidence'],
                        actual_outcome=item['actual_outcome'],
                    ))
                
                logger.info(f"Loaded {len(self.trade_buffer)} trades from buffer")
            except Exception as e:
                logger.warning(f"Failed to load trade buffer: {e}")
    
    def _save_buffer(self):
        """Save trade buffer to disk."""
        buffer_file = self.storage_dir / "trade_buffer.json"
        
        data = {
            'trades_since_retrain': self.trades_since_retrain,
            'last_updated': datetime.utcnow().isoformat(),
            'trades': [
                {
                    'trade_id': t.trade_id,
                    'instrument': t.instrument,
                    'direction': t.direction,
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat(),
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'pnl_pips': t.pnl_pips,
                    'pnl_percent': t.pnl_percent,
                    'features': t.features.tolist(),
                    'prediction': t.prediction,
                    'confidence': t.confidence,
                    'actual_outcome': t.actual_outcome,
                }
                for t in self.trade_buffer[-self.buffer_size:]
            ],
        }
        
        with open(buffer_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def record_trade(self, trade: TradeOutcome):
        """
        Record a completed trade for learning.
        
        Args:
            trade: TradeOutcome with features and actual result
        """
        self.trade_buffer.append(trade)
        self.trades_since_retrain += 1
        
        # Trim buffer to max size
        if len(self.trade_buffer) > self.buffer_size:
            self.trade_buffer = self.trade_buffer[-self.buffer_size:]
        
        self._save_buffer()
        
        # Log trade outcome
        correct = "✓" if trade.was_correct else "✗"
        logger.info(
            f"Trade recorded [{correct}]: {trade.instrument} "
            f"pred={trade.prediction:.2f} actual={'profit' if trade.actual_outcome else 'loss'} "
            f"({trade.pnl_pips:+.1f} pips)"
        )
    
    def should_retrain(self) -> bool:
        """Check if we have enough new trades to trigger retraining."""
        return self.trades_since_retrain >= self.retrain_threshold
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get features and labels from trade buffer for retraining.
        
        Returns:
            (X, y) - Features and binary outcomes
        """
        if not self.trade_buffer:
            return np.array([]), np.array([])
        
        X = np.stack([t.features for t in self.trade_buffer])
        y = np.array([t.actual_outcome for t in self.trade_buffer])
        
        return X, y
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from recent trades."""
        if not self.trade_buffer:
            return {'total_trades': 0}
        
        correct = sum(1 for t in self.trade_buffer if t.was_correct)
        total = len(self.trade_buffer)
        
        pnl_pips = [t.pnl_pips for t in self.trade_buffer]
        
        return {
            'total_trades': total,
            'accuracy': correct / total if total > 0 else 0,
            'correct_trades': correct,
            'total_pips': sum(pnl_pips),
            'avg_pips': np.mean(pnl_pips) if pnl_pips else 0,
            'win_rate': sum(1 for p in pnl_pips if p > 0) / total if total > 0 else 0,
            'trades_since_retrain': self.trades_since_retrain,
        }
    
    def mark_retrained(self):
        """Mark that model was retrained, reset counter."""
        self.trades_since_retrain = 0
        self._save_buffer()
        logger.info("Online learning: marked as retrained")


# =============================================================================
# INTEGRATED MARKET INTELLIGENCE
# =============================================================================

class MarketIntelligence:
    """
    Unified market intelligence combining sentiment, calendar, and online learning.
    
    Usage:
        intel = MarketIntelligence()
        
        # Before trading
        can_trade, reason = intel.pre_trade_check('USD_JPY')
        
        # After trade closes
        intel.record_trade_outcome(trade_data)
        
        # Check if model needs update
        if intel.should_update_model():
            # Trigger incremental training
    """
    
    def __init__(
        self,
        enable_sentiment: bool = True,
        enable_calendar: bool = True,
        enable_online_learning: bool = True,
    ):
        self.sentiment = NewsSentimentAnalyzer() if enable_sentiment else None
        self.calendar = EconomicCalendar() if enable_calendar else None
        self.online_learner = OnlineLearner() if enable_online_learning else None
        
        logger.info(
            f"MarketIntelligence initialized: "
            f"sentiment={enable_sentiment}, calendar={enable_calendar}, "
            f"online_learning={enable_online_learning}"
        )
    
    def pre_trade_check(
        self,
        instrument: str,
        headlines: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Pre-trade safety and intelligence check.
        
        Args:
            instrument: FX pair
            headlines: Optional news headlines for sentiment
        
        Returns:
            (can_trade, reason, intel_data)
        """
        intel_data = {}
        
        # 1. Check economic calendar
        if self.calendar:
            is_safe, event_reason = self.calendar.check_trade_safety(instrument)
            intel_data['calendar_safe'] = is_safe
            intel_data['calendar_reason'] = event_reason
            
            if not is_safe:
                return False, event_reason, intel_data
            
            next_event = self.calendar.get_next_high_impact(instrument)
            if next_event:
                intel_data['next_high_impact'] = {
                    'name': next_event.name,
                    'currency': next_event.currency,
                    'minutes_until': next_event.minutes_until,
                }
        
        # 2. Analyze sentiment (if headlines provided)
        if self.sentiment and headlines:
            sentiment = self.sentiment.get_instrument_sentiment(instrument, headlines)
            intel_data['sentiment'] = sentiment
            
            # Optional: Block trades against strong sentiment
            # if abs(sentiment['aggregate_score']) > 0.7:
            #     return False, f"Strong {sentiment['aggregate_label']} sentiment", intel_data
        
        return True, None, intel_data
    
    def record_trade_outcome(
        self,
        trade_id: str,
        instrument: str,
        direction: int,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        exit_price: float,
        pnl_pips: float,
        features: np.ndarray,
        prediction: float,
        confidence: float,
    ):
        """Record completed trade for online learning."""
        if self.online_learner is None:
            return
        
        pnl_percent = pnl_pips / 10000  # Approximate
        actual_outcome = 1 if pnl_pips > 0 else 0
        
        trade = TradeOutcome(
            trade_id=trade_id,
            instrument=instrument,
            direction=direction,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pips=pnl_pips,
            pnl_percent=pnl_percent,
            features=features,
            prediction=prediction,
            confidence=confidence,
            actual_outcome=actual_outcome,
        )
        
        self.online_learner.record_trade(trade)
    
    def should_update_model(self) -> bool:
        """Check if model should be updated with recent trades."""
        if self.online_learner is None:
            return False
        return self.online_learner.should_retrain()
    
    def get_online_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get accumulated trade data for incremental training."""
        if self.online_learner is None:
            return np.array([]), np.array([])
        return self.online_learner.get_training_data()
    
    def mark_model_updated(self):
        """Mark that model was updated."""
        if self.online_learner:
            self.online_learner.mark_retrained()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive intelligence stats."""
        stats = {}
        
        if self.online_learner:
            stats['online_learning'] = self.online_learner.get_performance_stats()
        
        return stats


# =============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# =============================================================================

def fetch_forex_news(instrument: str, max_items: int = 10) -> List[str]:
    """
    Fetch recent news headlines for an instrument.
    
    This is a placeholder - in production, would connect to:
    - Reuters API
    - Bloomberg API
    - ForexLive RSS
    - Twitter/X financial accounts
    
    Returns:
        List of headline strings
    """
    # Placeholder - return empty list
    # In production, integrate with news APIs
    logger.debug(f"fetch_forex_news({instrument}) - no API configured")
    return []


def add_sentiment_features(
    df,
    instrument: str,
    sentiment_analyzer: Optional[NewsSentimentAnalyzer] = None,
) -> None:
    """
    Add sentiment features to a dataframe.
    
    Adds columns:
    - news_sentiment: Aggregate sentiment score (-1 to 1)
    - news_volume: Number of recent headlines
    - news_bullish_pct: % bullish headlines
    
    Note: In production, this would fetch live news. For backtesting,
    use historical sentiment data from a provider.
    """
    if sentiment_analyzer is None:
        sentiment_analyzer = NewsSentimentAnalyzer()
    
    # Placeholder values (no news API)
    df['news_sentiment'] = 0.0
    df['news_volume'] = 0
    df['news_bullish_pct'] = 0.5
    
    logger.debug(f"add_sentiment_features({instrument}) - using placeholder values")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize
    intel = MarketIntelligence()
    
    # Pre-trade check
    can_trade, reason, data = intel.pre_trade_check("USD_JPY")
    print(f"Can trade: {can_trade}, Reason: {reason}")
    print(f"Intel data: {data}")
    
    # Test sentiment
    sentiment = NewsSentimentAnalyzer()
    result = sentiment.analyze("Fed signals hawkish stance on inflation")
    print(f"Sentiment: {result}")
