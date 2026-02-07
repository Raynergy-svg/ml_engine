"""
Market Intelligence Module - News Sentiment, Economic Calendar, Online Learning

Provides:
1. News sentiment analysis using FinBERT
2. Economic calendar integration (ForexFactory, Investing.com)
3. Online/incremental learning from trade outcomes
4. Drift detection with automatic model retraining triggers

Author: ML Engine
"""

from __future__ import annotations

import logging
import json
import hashlib
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import deque

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
    Falls back to VADER if transformers not available or fast_mode=True.
    """
    
    def __init__(self, cache_dir: str = "trained_data/sentiment_cache", fast_mode: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._vader = None
        self._use_finbert = False
        self._initialized = False
        self._fast_mode = fast_mode  # Use VADER by default (instant loading)
    
    def _initialize(self):
        """Lazy load sentiment model."""
        if self._initialized:
            return
        
        # Fast mode: use VADER (instant loading, good enough for FX news)
        if self._fast_mode:
            try:
                import nltk
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                try:
                    nltk.data.find('sentiment/vader_lexicon.zip')
                except LookupError:
                    nltk.download('vader_lexicon', quiet=True)
                self._vader = SentimentIntensityAnalyzer()
                self._use_finbert = False
                self._initialized = True
                logger.debug("Using VADER for fast sentiment analysis")
                return
            except ImportError:
                logger.debug("VADER not available, trying transformers")
        
        # Full mode: try FinBERT/DistilBERT
        try:
            from transformers import pipeline
            import warnings
            
            # Suppress verbose transformers warnings during fallback
            warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
            
            # Try FinBERT first (best for financial text)
            try:
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device=-1,  # CPU
                )
                self._use_finbert = True
                logger.info("✓ FinBERT sentiment model loaded")
            except Exception as e:
                error_msg = str(e)
                if "torch" in error_msg.lower() and ("2.6" in error_msg or "upgrade" in error_msg.lower()):
                    logger.info("FinBERT requires torch>=2.6, using DistilBERT")
                else:
                    short_error = error_msg[:80] + "..." if len(error_msg) > 80 else error_msg
                    logger.info(f"FinBERT unavailable ({short_error}), using DistilBERT")
                
                try:
                    self._pipeline = pipeline(
                        "sentiment-analysis",
                        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                        device=-1,
                    )
                except Exception:
                    self._pipeline = pipeline("sentiment-analysis", device=-1)
                self._use_finbert = False
            
            self._initialized = True
            
        except ImportError:
            # Final fallback to VADER
            try:
                import nltk
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                try:
                    nltk.data.find('sentiment/vader_lexicon.zip')
                except LookupError:
                    nltk.download('vader_lexicon', quiet=True)
                self._vader = SentimentIntensityAnalyzer()
                self._use_finbert = False
                self._initialized = True
                logger.debug("Using VADER fallback for sentiment analysis")
            except Exception as e:
                logger.warning(f"Sentiment analysis unavailable: {e}")
                self._initialized = True
    
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
        
        # Pre-process: detect beat/miss patterns for economic data
        text_lower = text.lower()
        econ_sentiment = self._analyze_economic_surprise(text)
        if econ_sentiment is not None:
            return econ_sentiment
        
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
        
        # VADER fallback with financial keyword boosting
        if self._vader is not None:
            # Boost VADER with financial keywords
            boosted_text = self._boost_financial_text(text)
            scores = self._vader.polarity_scores(boosted_text)
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
    
    def _analyze_economic_surprise(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Detect economic data beats/misses from headlines.
        
        Patterns like:
        - "US January Philly Fed +12.6 vs -1.0 expected" -> bullish (beat)
        - "GDP 2.1% vs 2.5% expected" -> bearish (miss)
        """
        import re
        
        # Pattern: actual vs expected
        pattern = r'([+-]?\d+\.?\d*)\s*%?\s*vs\s*([+-]?\d+\.?\d*)\s*%?\s*(?:expected|exp|forecast)'
        match = re.search(pattern, text.lower())
        
        if match:
            try:
                actual = float(match.group(1))
                expected = float(match.group(2))
                diff = actual - expected
                
                # Calculate score based on surprise magnitude
                if abs(diff) < 0.1:  # Within 0.1 = as expected
                    return {'score': 0.0, 'label': 'neutral', 'confidence': 0.3}
                
                # Normalize to -1 to 1 range (cap at ±5 point surprise)
                score = max(-1.0, min(1.0, diff / 5.0))
                label = 'bullish' if score > 0 else 'bearish'
                confidence = min(1.0, abs(diff) / 3.0)  # Higher diff = higher confidence
                
                return {'score': score, 'label': label, 'confidence': confidence}
            except (ValueError, IndexError):
                pass
        
        return None
    
    def _boost_financial_text(self, text: str) -> str:
        """Add sentiment-carrying words to help VADER understand financial text."""
        text_lower = text.lower()
        boosts = []
        
        # Bullish indicators
        bullish_patterns = [
            ('surge', 'excellent great positive'),
            ('rally', 'great positive'),
            ('beat', 'excellent positive'),
            ('strong', 'good positive'),
            ('jump', 'good'),
            ('gain', 'positive'),
            ('rise', 'good'),
            ('hawkish', 'positive strong'),  # For central banks
        ]
        
        # Bearish indicators  
        bearish_patterns = [
            ('crash', 'terrible bad negative'),
            ('plunge', 'bad negative'),
            ('miss', 'disappointing negative'),
            ('weak', 'bad negative'),
            ('fall', 'negative'),
            ('drop', 'bad'),
            ('decline', 'negative'),
            ('dovish', 'negative weak'),  # For central banks
        ]
        
        for pattern, boost in bullish_patterns:
            if pattern in text_lower:
                boosts.append(boost)
        
        for pattern, boost in bearish_patterns:
            if pattern in text_lower:
                boosts.append(boost)
        
        return text + ' ' + ' '.join(boosts) if boosts else text
    
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
        self._feed_cache_file = self.cache_dir / "ff_calendar_thisweek.xml"
        self._last_error: Optional[str] = None
    
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
        
        # Check cache (valid for 6 hours)
        cache_key = f"{instrument}_{hours_ahead}"
        if not force_refresh and self._cache_time:
            cache_age = (datetime.utcnow() - self._cache_time).total_seconds() / 3600
            if cache_age < 6.0 and cache_key in self._events_cache:
                return self._events_cache[cache_key]
        
        self._last_error = None
        events = []
        
        # Try to fetch from ForexFactory API (or cached file)
        try:
            events = self._fetch_forexfactory(currencies, hours_ahead)
        except Exception as e:
            self._last_error = f"ForexFactory fetch failed: {e}"
            logger.warning(self._last_error)
        
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
        """Fetch from ForexFactory calendar XML (public feed)."""
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

        # Use cached feed if it's fresh (6 hours)
        if self._feed_cache_file.exists():
            age_seconds = (datetime.utcnow().timestamp() - self._feed_cache_file.stat().st_mtime)
            if age_seconds < 21600:
                xml_data = self._feed_cache_file.read_bytes()
            else:
                xml_data = None
        else:
            xml_data = None

        if xml_data is None:
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                        "Accept": "application/xml,text/xml,*/*",
                        "Referer": "https://www.forexfactory.com/",
                    },
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    xml_data = resp.read()
                try:
                    self._feed_cache_file.write_bytes(xml_data)
                except Exception:
                    pass
            except Exception as e:
                self._last_error = f"ForexFactory calendar fetch failed: {e}"
                logger.warning(self._last_error)
                if self._feed_cache_file.exists():
                    xml_data = self._feed_cache_file.read_bytes()
                else:
                    return []

        try:
            root = ET.fromstring(xml_data)
        except Exception as e:
            logger.warning(f"Failed to parse ForexFactory XML: {e}")
            return []

        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours_ahead)
        events: List[EconomicEvent] = []

        for item in root.findall(".//event"):
            try:
                currency = (item.findtext("country") or "").strip().upper()
                if currency not in currencies:
                    continue

                title = (item.findtext("title") or "").strip()
                impact = (item.findtext("impact") or "").strip().lower()
                date_str = (item.findtext("date") or "").strip()
                time_str = (item.findtext("time") or "").strip()

                # Skip events without a concrete time
                if not date_str or not time_str:
                    continue
                if time_str.lower() in {"all day", "tentative"}:
                    continue

                # ForexFactory uses formats like "Jan 16, 2026" or "01-16-2026"
                dt_str = f"{date_str} {time_str}".replace(".", "")
                parsed = None
                date_formats = ["%b %d, %Y", "%m-%d-%Y", "%d-%m-%Y"]
                time_formats = ["%I:%M%p", "%H:%M"]
                for df in date_formats:
                    for tf in time_formats:
                        try:
                            parsed = datetime.strptime(dt_str, f"{df} {tf}")
                            break
                        except ValueError:
                            continue
                    if parsed is not None:
                        break
                if parsed is None:
                    continue
                event_time = parsed

                if not (now <= event_time <= cutoff):
                    continue

                events.append(EconomicEvent(
                    name=title,
                    currency=currency,
                    impact=impact,
                    time=event_time,
                    actual=(item.findtext("actual") or None),
                    forecast=(item.findtext("forecast") or None),
                    previous=(item.findtext("previous") or None),
                ))
            except Exception:
                continue

        return sorted(events, key=lambda e: e.time)
    
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
        return self._check_trade_safety_from_events(events, min_minutes_before, min_minutes_after)

    def _check_trade_safety_from_events(
        self,
        events: List[EconomicEvent],
        min_minutes_before: int,
        min_minutes_after: int,
    ) -> Tuple[bool, Optional[str]]:
        """Check safety using a pre-fetched event list."""
        
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

    def _get_next_high_impact_from_events(
        self,
        events: List[EconomicEvent],
    ) -> Optional[EconomicEvent]:
        """Get the next high-impact event from a pre-fetched list."""
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
        
        # Filter trades to have consistent feature sizes
        # Use the most common feature size
        feature_sizes = [len(t.features) for t in self.trade_buffer]
        if not feature_sizes:
            return np.array([]), np.array([])
        
        # Find the most common feature size
        from collections import Counter
        size_counts = Counter(feature_sizes)
        target_size = size_counts.most_common(1)[0][0]
        
        # Filter to trades with the target feature size
        valid_trades = [t for t in self.trade_buffer if len(t.features) == target_size]
        
        if not valid_trades:
            return np.array([]), np.array([])
        
        X = np.stack([t.features for t in valid_trades])
        y = np.array([t.actual_outcome for t in valid_trades])
        
        logger.info(f"Training data: {len(valid_trades)}/{len(self.trade_buffer)} trades with {target_size} features")
        
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
# DRIFT DETECTION & INCREMENTAL RETRAINING MANAGER
# =============================================================================

@dataclass
class DriftConfig:
    """Configuration for drift detection and incremental retraining."""
    # Feature drift thresholds
    feature_drift_threshold: float = 0.10  # 10% shift in feature distribution
    feature_drift_window: int = 100  # Samples for computing drift stats
    
    # Performance drift thresholds
    performance_drift_threshold: float = 0.05  # 5% accuracy drop triggers retrain
    min_trades_for_drift_check: int = 20  # Minimum trades before checking drift
    
    # Incremental retraining settings
    incremental_retrain_epochs: int = 3  # Few epochs for quick adaptation
    incremental_learning_rate_factor: float = 0.1  # LR = base_lr * 0.1 for fine-tuning
    min_samples_for_retrain: int = 50  # Minimum samples needed to retrain
    
    # Replay buffer settings
    replay_buffer_size: int = 500  # Max samples to keep
    replay_mix_ratio: float = 0.20  # Mix 20% old samples during retrain
    
    # Automatic retraining triggers
    auto_retrain_on_drift: bool = True  # Auto-trigger retrain when drift detected
    max_retrains_per_day: int = 3  # Prevent excessive retraining
    cooldown_minutes: int = 60  # Minimum time between retrains


@dataclass 
class DriftResult:
    """Result of drift detection check."""
    drift_detected: bool
    feature_drift: bool = False
    performance_drift: bool = False
    feature_shift_pct: float = 0.0
    accuracy_drop: float = 0.0
    reason: str = ""
    recommendation: str = "continue"  # 'continue', 'monitor', 'retrain', 'full_retrain'
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class DriftDetectionManager:
    """
    Real-time drift detection for inference mode.
    
    Monitors:
    1. Feature distribution shift (input drift)
    2. Model performance degradation (concept drift)
    3. Prediction confidence patterns
    
    When drift exceeds thresholds, triggers incremental retraining.
    
    Usage:
        drift_mgr = DriftDetectionManager(config)
        
        # During inference:
        drift_mgr.record_features(features)
        drift_mgr.record_prediction(prediction, outcome)
        
        # Check if retraining needed:
        result = drift_mgr.check_drift()
        if result.drift_detected and result.recommendation == 'retrain':
            retrain_callback()
    """
    
    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        storage_dir: str = "trained_data/drift_detection",
        retrain_callback: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        self.config = config or DriftConfig()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Callback for incremental retraining
        self._retrain_callback = retrain_callback
        
        # Feature statistics (baseline from training)
        self._baseline_feature_stats: Optional[Dict[str, np.ndarray]] = None
        
        # Running feature statistics (recent inference samples)
        self._recent_features: deque = deque(maxlen=self.config.feature_drift_window)
        
        # Performance tracking
        self._predictions: deque = deque(maxlen=self.config.feature_drift_window)
        self._outcomes: deque = deque(maxlen=self.config.feature_drift_window)
        self._baseline_accuracy: float = 0.0
        
        # Replay buffer for incremental training
        self._replay_buffer_X: deque = deque(maxlen=self.config.replay_buffer_size)
        self._replay_buffer_y: deque = deque(maxlen=self.config.replay_buffer_size)
        
        # Retraining cooldown
        self._last_retrain_time: Optional[datetime] = None
        self._retrains_today: int = 0
        self._last_retrain_date: Optional[str] = None
        
        # Drift history for monitoring
        self._drift_history: List[DriftResult] = []
        
        # Load saved state if exists
        self._load_state()
    
    def set_baseline(
        self,
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
        baseline_accuracy: float = 0.0,
    ) -> None:
        """
        Set baseline feature statistics from training data.
        
        This should be called after model training to establish the 
        reference distribution for drift detection.
        
        Args:
            feature_means: Mean of each feature from training data
            feature_stds: Std of each feature from training data  
            baseline_accuracy: Model accuracy on validation set
        """
        self._baseline_feature_stats = {
            'means': np.array(feature_means),
            'stds': np.array(feature_stds),
            'n_features': len(feature_means),
        }
        self._baseline_accuracy = baseline_accuracy
        self._save_state()
        logger.info(
            f"📊 Drift baseline set: {len(feature_means)} features, "
            f"baseline_acc={baseline_accuracy:.2%}"
        )
    
    def record_features(self, features: np.ndarray) -> None:
        """
        Record features from an inference sample for drift monitoring.
        
        Args:
            features: 1D array of features from current inference
        """
        # Flatten if needed
        if features.ndim > 1:
            features = features.flatten()[-self._baseline_feature_stats['n_features']:]
        self._recent_features.append(features)
    
    def record_prediction(
        self,
        prediction: float,
        outcome: Optional[int] = None,
        features: Optional[np.ndarray] = None,
    ) -> None:
        """
        Record a model prediction and its outcome.
        
        Args:
            prediction: Model's probability output (0-1)
            outcome: Actual outcome (1=correct, 0=incorrect), can be added later
            features: Feature array for replay buffer
        """
        self._predictions.append(prediction)
        if outcome is not None:
            self._outcomes.append(outcome)
        
        # Add to replay buffer if features provided
        if features is not None:
            if features.ndim > 1:
                features = features.flatten()
            self._replay_buffer_X.append(features)
            if outcome is not None:
                self._replay_buffer_y.append(outcome)
    
    def record_trade_outcome(
        self,
        prediction: float,
        outcome: int,
        features: np.ndarray,
    ) -> Optional[DriftResult]:
        """
        Record a completed trade outcome and check for drift.
        
        This is the main entry point during live trading.
        
        Args:
            prediction: Model's prediction at entry
            outcome: 1 if trade was profitable, 0 otherwise
            features: Features at trade entry
            
        Returns:
            DriftResult if drift was detected, None otherwise
        """
        self.record_prediction(prediction, outcome, features)
        self.record_features(features)
        
        # Check drift if we have enough samples
        if len(self._outcomes) >= self.config.min_trades_for_drift_check:
            result = self.check_drift()
            if result.drift_detected:
                self._drift_history.append(result)
                self._save_state()
                
                # Auto-trigger retraining if enabled
                if (self.config.auto_retrain_on_drift and 
                    result.recommendation in ('retrain', 'full_retrain')):
                    self._maybe_trigger_retrain(result)
                
                return result
        
        return None
    
    def check_drift(self) -> DriftResult:
        """
        Check for feature distribution and performance drift.
        
        Returns:
            DriftResult with drift status and recommendations
        """
        result = DriftResult(drift_detected=False)
        
        # === Feature Drift Check ===
        if (self._baseline_feature_stats is not None and 
            len(self._recent_features) >= self.config.min_trades_for_drift_check):
            
            recent_array = np.array(list(self._recent_features))
            
            # Compute current feature statistics
            current_means = np.mean(recent_array, axis=0)
            baseline_means = self._baseline_feature_stats['means']
            baseline_stds = self._baseline_feature_stats['stds']
            
            # Ensure shapes match
            min_len = min(len(current_means), len(baseline_means))
            current_means = current_means[:min_len]
            baseline_means = baseline_means[:min_len]
            baseline_stds = baseline_stds[:min_len]
            
            # Compute normalized shift (z-score)
            shift = np.abs(current_means - baseline_means)
            normalized_shift = shift / (baseline_stds + 1e-8)
            max_shift = float(np.max(normalized_shift))
            mean_shift = float(np.mean(normalized_shift))
            
            # Check if significant drift
            if max_shift > self.config.feature_drift_threshold * 10:  # 10 sigma
                result.feature_drift = True
                result.feature_shift_pct = mean_shift
                result.drift_detected = True
        
        # === Performance Drift Check ===
        if (len(self._outcomes) >= self.config.min_trades_for_drift_check and
            self._baseline_accuracy > 0):
            
            # Compute recent accuracy
            outcomes = list(self._outcomes)
            predictions = list(self._predictions)
            
            # Binary accuracy
            recent_correct = sum(
                1 for pred, out in zip(predictions[-len(outcomes):], outcomes)
                if (pred > 0.5) == (out == 1)
            )
            recent_accuracy = recent_correct / len(outcomes)
            
            # Check accuracy drop
            accuracy_drop = self._baseline_accuracy - recent_accuracy
            if accuracy_drop > self.config.performance_drift_threshold:
                result.performance_drift = True
                result.accuracy_drop = accuracy_drop
                result.drift_detected = True
        
        # === Build Recommendation ===
        if result.drift_detected:
            if result.feature_drift and result.performance_drift:
                result.recommendation = 'full_retrain'
                result.reason = (
                    f"Feature AND performance drift: "
                    f"feature_shift={result.feature_shift_pct:.2f}σ, "
                    f"accuracy_drop={result.accuracy_drop:.2%}"
                )
                logger.warning(f"⚠️ DRIFT DETECTED: {result.reason}")
            elif result.performance_drift:
                result.recommendation = 'retrain'
                result.reason = f"Performance drift: accuracy_drop={result.accuracy_drop:.2%}"
                logger.warning(f"⚠️ Performance drift: {result.reason}")
            elif result.feature_drift:
                result.recommendation = 'monitor'
                result.reason = f"Feature drift: shift={result.feature_shift_pct:.2f}σ"
                logger.info(f"📊 Feature drift (monitoring): {result.reason}")
        else:
            result.recommendation = 'continue'
            result.reason = "No significant drift detected"
        
        return result
    
    def _maybe_trigger_retrain(self, drift_result: DriftResult) -> bool:
        """
        Maybe trigger incremental retraining based on drift and cooldown.
        
        Note: This is called automatically by record_trade_outcome() when drift
        is detected. For manual triggering, use trigger_retraining_if_needed().
        
        Returns:
            True if retrain was triggered
        """
        # Use the new unified check method
        check = self._check_retrain_allowed()
        
        if not check['allowed']:
            logger.info(f"📊 Drift-triggered retrain blocked: {check['reason']}")
            return False
        
        # Check if we have enough samples
        if len(self._replay_buffer_X) < self.config.min_samples_for_retrain:
            logger.info(
                f"📊 Drift detected but insufficient samples "
                f"({len(self._replay_buffer_X)}/{self.config.min_samples_for_retrain})"
            )
            return False
        
        # Trigger retrain
        if self._retrain_callback is not None:
            logger.info(
                f"🔄 AUTO-RETRAIN: Triggering incremental retrain due to drift\n"
                f"   Reason: {drift_result.reason}\n"
                f"   Recommendation: {drift_result.recommendation}\n"
                f"   Samples in buffer: {len(self._replay_buffer_X)}"
            )
            try:
                result = self._retrain_callback()
                self._last_retrain_time = datetime.utcnow()
                self._retrains_today += 1
                self._save_state()
                
                # Log success with details
                models = result.get('models_retrained', [])
                status = result.get('status', 'unknown')
                logger.info(
                    f"✅ AUTO-RETRAIN completed: status={status}, "
                    f"models={models}, retrains_today={self._retrains_today}"
                )
                return True
            except Exception as e:
                logger.error(f"❌ AUTO-RETRAIN failed: {e}")
                return False
        else:
            logger.warning("⚠️ Drift detected but no retrain callback set")
            return False
    
    def set_retrain_callback(
        self,
        callback: Callable[[], Dict[str, Any]],
    ) -> None:
        """Set the callback function for incremental retraining."""
        self._retrain_callback = callback
        logger.info("📊 Drift detection retrain callback registered")
    
    def trigger_retraining_if_needed(
        self,
        force: bool = False,
        queue_if_blocked: bool = True,
    ) -> Dict[str, Any]:
        """
        Check if retraining is needed and trigger if thresholds exceeded.
        
        This is the recommended method to call periodically during inference
        to ensure drift-triggered retraining happens.
        
        Args:
            force: Bypass cooldown and daily limits
            queue_if_blocked: If True, set pending_retrain flag when blocked
            
        Returns:
            Dictionary with:
            - triggered: Whether retrain was triggered
            - status: 'not_needed', 'triggered', 'blocked', 'queued', 'no_callback'
            - reason: Why retraining was/wasn't triggered
            - result: Retrain result if triggered
        """
        response = {
            'triggered': False,
            'status': 'not_needed',
            'reason': 'No drift detected',
            'result': None,
        }
        
        # Check if drift warrants retraining
        drift_result = self.check_drift()
        
        if not drift_result.drift_detected:
            return response
        
        if drift_result.recommendation not in ('retrain', 'full_retrain'):
            response['reason'] = f"Drift detected but recommendation is '{drift_result.recommendation}'"
            return response
        
        # Drift detected and retrain recommended
        logger.info(
            f"🔄 Drift-triggered retraining check: {drift_result.reason} "
            f"→ recommendation={drift_result.recommendation}"
        )
        
        if self._retrain_callback is None:
            response['status'] = 'no_callback'
            response['reason'] = 'Drift detected but no retrain callback configured'
            logger.warning("⚠️ Drift detected but no retrain callback set. "
                          "Run: python main.py retrain-gates")
            return response
        
        # Check if we can retrain (cooldown, daily limit)
        if not force:
            can_retrain = self._check_retrain_allowed()
            if not can_retrain['allowed']:
                response['status'] = 'blocked' if not queue_if_blocked else 'queued'
                response['reason'] = can_retrain['reason']
                logger.info(f"📊 Retrain blocked: {can_retrain['reason']}")
                return response
        
        # Trigger retrain
        try:
            logger.info("🔄 Triggering drift-initiated model retraining...")
            result = self._retrain_callback()
            
            response['triggered'] = True
            response['status'] = 'triggered'
            response['reason'] = drift_result.reason
            response['result'] = result
            
            # Update tracking
            self._last_retrain_time = datetime.utcnow()
            self._retrains_today += 1
            self._save_state()
            
            logger.info(
                f"✅ Drift-triggered retrain completed: "
                f"status={result.get('status', 'unknown')}, "
                f"models={result.get('models_retrained', [])}"
            )
            
        except Exception as e:
            response['status'] = 'error'
            response['reason'] = f"Retrain callback failed: {e}"
            logger.error(f"❌ Drift-triggered retrain failed: {e}")
        
        return response
    
    def _check_retrain_allowed(self) -> Dict[str, Any]:
        """Check if retraining is allowed based on cooldown and limits."""
        now = datetime.utcnow()
        today = now.strftime('%Y-%m-%d')
        
        # Reset daily counter if new day
        if self._last_retrain_date != today:
            self._retrains_today = 0
            self._last_retrain_date = today
        
        # Check daily limit
        if self._retrains_today >= self.config.max_retrains_per_day:
            return {
                'allowed': False,
                'reason': f"Daily limit reached ({self.config.max_retrains_per_day}/day)",
            }
        
        # Check cooldown
        if self._last_retrain_time is not None:
            elapsed = (now - self._last_retrain_time).total_seconds() / 60
            remaining = self.config.cooldown_minutes - elapsed
            if remaining > 0:
                return {
                    'allowed': False,
                    'reason': f"Cooldown active ({remaining:.0f} min remaining)",
                }
        
        return {'allowed': True, 'reason': 'Ready'}

    def get_replay_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get accumulated data for incremental retraining.
        
        Returns:
            (X, y) - Features and labels from replay buffer
        """
        if not self._replay_buffer_X or not self._replay_buffer_y:
            return np.array([]), np.array([])
        
        X = np.array(list(self._replay_buffer_X))
        y = np.array(list(self._replay_buffer_y))
        return X, y
    
    def get_drift_stats(self) -> Dict[str, Any]:
        """Get drift detection statistics for monitoring."""
        return {
            'baseline_set': self._baseline_feature_stats is not None,
            'recent_samples': len(self._recent_features),
            'tracked_outcomes': len(self._outcomes),
            'replay_buffer_size': len(self._replay_buffer_X),
            'drift_events': len(self._drift_history),
            'retrains_today': self._retrains_today,
            'last_retrain': self._last_retrain_time.isoformat() if self._last_retrain_time else None,
            'baseline_accuracy': self._baseline_accuracy,
            'current_accuracy': (
                sum(
                    1 for p, o in zip(list(self._predictions), list(self._outcomes))
                    if (p > 0.5) == (o == 1)
                ) / len(self._outcomes)
                if self._outcomes else None
            ),
        }
    
    def _save_state(self) -> None:
        """Save drift detector state to disk."""
        state = {
            'baseline_feature_stats': (
                {k: v.tolist() if isinstance(v, np.ndarray) else v 
                 for k, v in self._baseline_feature_stats.items()}
                if self._baseline_feature_stats else None
            ),
            'baseline_accuracy': self._baseline_accuracy,
            'last_retrain_time': self._last_retrain_time.isoformat() if self._last_retrain_time else None,
            'last_retrain_date': self._last_retrain_date,
            'retrains_today': self._retrains_today,
            'drift_history': [
                {
                    'drift_detected': d.drift_detected,
                    'feature_drift': d.feature_drift,
                    'performance_drift': d.performance_drift,
                    'reason': d.reason,
                    'recommendation': d.recommendation,
                    'timestamp': d.timestamp,
                }
                for d in self._drift_history[-100:]  # Keep last 100
            ],
            'saved_at': datetime.utcnow().isoformat(),
        }
        
        try:
            with open(self.storage_dir / 'drift_state.json', 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save drift state: {e}")
    
    def _load_state(self) -> None:
        """Load drift detector state from disk."""
        state_path = self.storage_dir / 'drift_state.json'
        if not state_path.exists():
            return
        
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            if state.get('baseline_feature_stats'):
                self._baseline_feature_stats = {
                    k: np.array(v) if isinstance(v, list) else v
                    for k, v in state['baseline_feature_stats'].items()
                }
            
            self._baseline_accuracy = state.get('baseline_accuracy', 0.0)
            
            if state.get('last_retrain_time'):
                self._last_retrain_time = datetime.fromisoformat(state['last_retrain_time'])
            
            self._last_retrain_date = state.get('last_retrain_date')
            self._retrains_today = state.get('retrains_today', 0)
            
            # Restore drift history
            for d in state.get('drift_history', []):
                self._drift_history.append(DriftResult(
                    drift_detected=d['drift_detected'],
                    feature_drift=d.get('feature_drift', False),
                    performance_drift=d.get('performance_drift', False),
                    reason=d.get('reason', ''),
                    recommendation=d.get('recommendation', 'continue'),
                    timestamp=d.get('timestamp', ''),
                ))
            
            logger.info(f"📊 Drift detector state loaded ({len(self._drift_history)} events)")
            
        except Exception as e:
            logger.warning(f"Failed to load drift state: {e}")


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
        enable_drift_detection: bool = True,
        drift_config: Optional[DriftConfig] = None,
        retrain_callback: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        self.sentiment = NewsSentimentAnalyzer() if enable_sentiment else None
        self.calendar = EconomicCalendar() if enable_calendar else None
        self.online_learner = OnlineLearner() if enable_online_learning else None
        
        # Initialize drift detection manager
        self.drift_manager: Optional[DriftDetectionManager] = None
        if enable_drift_detection:
            self.drift_manager = DriftDetectionManager(
                config=drift_config,
                retrain_callback=retrain_callback,
            )
        
        # Track enabled features
        self.enable_drift_detection = enable_drift_detection
        
        logger.info(
            f"MarketIntelligence initialized: "
            f"sentiment={enable_sentiment}, calendar={enable_calendar}, "
            f"online_learning={enable_online_learning}, "
            f"drift_detection={enable_drift_detection}"
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
            events = self.calendar.fetch_events(instrument, hours_ahead=24)
            high_impact_count = sum(1 for e in events if e.is_high_impact)
            intel_data['calendar_events'] = len(events)
            intel_data['calendar_high_impact'] = high_impact_count
            is_safe, event_reason = self.calendar._check_trade_safety_from_events(
                events,
                min_minutes_before=30,
                min_minutes_after=15,
            )
            intel_data['calendar_safe'] = is_safe
            intel_data['calendar_reason'] = event_reason
            if self.calendar._last_error:
                intel_data['calendar_error'] = self.calendar._last_error
            
            if not is_safe:
                return False, event_reason, intel_data
            
            next_event = self.calendar._get_next_high_impact_from_events(events)
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
    ) -> Optional[DriftResult]:
        """
        Record completed trade for online learning and drift detection.
        
        Returns:
            DriftResult if drift was detected, None otherwise
        """
        pnl_percent = pnl_pips / 10000  # Approximate
        actual_outcome = 1 if pnl_pips > 0 else 0
        
        # Record for online learning
        if self.online_learner is not None:
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
        
        # Check drift and maybe trigger retraining
        drift_result = None
        if self.drift_manager is not None:
            drift_result = self.drift_manager.record_trade_outcome(
                prediction=prediction,
                outcome=actual_outcome,
                features=features,
            )
            if drift_result and drift_result.drift_detected:
                logger.warning(
                    f"📊 Drift detected after trade {trade_id}: "
                    f"{drift_result.reason}"
                )
        
        return drift_result
    
    def should_update_model(self) -> Tuple[bool, str]:
        """
        Check if model should be updated with recent trades.
        
        Returns:
            (should_update, reason)
        """
        # Check online learning threshold
        if self.online_learner is not None and self.online_learner.should_retrain():
            return True, "trade_threshold_reached"
        
        # Check drift detection
        if self.drift_manager is not None:
            result = self.drift_manager.check_drift()
            if result.drift_detected and result.recommendation in ('retrain', 'full_retrain'):
                return True, f"drift_detected: {result.reason}"
        
        return False, ""
    
    def set_drift_baseline(
        self,
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
        baseline_accuracy: float = 0.0,
    ) -> None:
        """
        Set baseline feature statistics for drift detection.
        
        Should be called after model training.
        
        Args:
            feature_means: Mean of each feature from training data
            feature_stds: Std of each feature from training data
            baseline_accuracy: Model accuracy on validation set
        """
        if self.drift_manager is not None:
            self.drift_manager.set_baseline(feature_means, feature_stds, baseline_accuracy)
    
    def set_retrain_callback(
        self,
        callback: Callable[[], Dict[str, Any]],
    ) -> None:
        """Set callback function for incremental retraining."""
        if self.drift_manager is not None:
            self.drift_manager.set_retrain_callback(callback)
    
    def check_drift(self) -> Optional[DriftResult]:
        """Explicitly check for drift (manual check)."""
        if self.drift_manager is not None:
            return self.drift_manager.check_drift()
        return None
    
    def trigger_retraining_if_needed(
        self,
        force: bool = False,
        queue_if_blocked: bool = True,
    ) -> Dict[str, Any]:
        """
        Check drift and trigger retraining if thresholds exceeded.
        
        This is the main entry point for drift-triggered retraining.
        Call this periodically during inference or after trade outcomes.
        
        Args:
            force: Bypass cooldown and daily limits
            queue_if_blocked: Queue retrain request if currently blocked
            
        Returns:
            Dictionary with trigger status and results
        """
        if self.drift_manager is None:
            return {
                'triggered': False,
                'status': 'disabled',
                'reason': 'Drift detection not enabled',
            }
        
        return self.drift_manager.trigger_retraining_if_needed(
            force=force,
            queue_if_blocked=queue_if_blocked,
        )

    def record_inference_features(self, features: np.ndarray) -> None:
        """Record features from an inference for drift monitoring."""
        if self.drift_manager is not None:
            self.drift_manager.record_features(features)
    
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
        
        if self.drift_manager:
            stats['drift_detection'] = self.drift_manager.get_drift_stats()
        
        return stats
    
    def get_replay_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get accumulated data for incremental retraining.
        
        Combines data from online learner and drift manager.
        """
        X_parts, y_parts = [], []
        
        # Get from online learner
        if self.online_learner is not None:
            X_online, y_online = self.online_learner.get_training_data()
            if len(X_online) > 0:
                X_parts.append(X_online)
                y_parts.append(y_online)
        
        # Get from drift manager replay buffer
        if self.drift_manager is not None:
            X_drift, y_drift = self.drift_manager.get_replay_data()
            if len(X_drift) > 0:
                X_parts.append(X_drift)
                y_parts.append(y_drift)
        
        if not X_parts:
            return np.array([]), np.array([])
        
        # Combine and deduplicate
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        
        return X, y


# =============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# =============================================================================

def fetch_forex_news(instrument: str, max_items: int = 10) -> List[str]:
    """
    Fetch recent news headlines for an instrument from RSS feeds.
    
    Uses multiple free RSS feeds:
    - ForexLive (primary forex news)
    - DailyFX (market analysis)
    - Investing.com (broad financial news)
    - FXStreet (forex-specific)
    
    Returns:
        List of headline strings relevant to the instrument
    """
    # Multiple RSS feeds for redundancy
    rss_feeds = [
        ("https://www.forexlive.com/feed/news", "ForexLive"),
        ("https://www.dailyfx.com/feeds/market-news", "DailyFX"),
        ("https://www.investing.com/rss/news_285.rss", "Investing.com"),
        ("https://www.fxstreet.com/rss/news", "FXStreet"),
    ]

    def _currencies(pair: str) -> List[str]:
        parts = pair.replace("/", "_").split("_")
        return [p.upper() for p in parts if len(p) == 3]

    # Comprehensive keyword mapping for currency detection
    currency_keywords = {
        "USD": ["usd", "dollar", "fed", "fomc", "nfp", "cpi", "ppi", "powell", "yellen", "treasury", "us economy", "payrolls", "inflation"],
        "EUR": ["eur", "euro", "ecb", "lagarde", "eurozone", "european", "germany", "france", "eu economy"],
        "GBP": ["gbp", "pound", "uk", "boe", "sterling", "bailey", "britain", "british", "brexit"],
        "JPY": ["jpy", "yen", "boj", "ueda", "japan", "japanese", "kuroda", "tokyo"],
        "CHF": ["chf", "franc", "snb", "swiss", "switzerland"],
        "CAD": ["cad", "loonie", "boc", "canada", "canadian", "macklem", "oil price"],
        "AUD": ["aud", "aussie", "rba", "australia", "australian", "bullock", "china trade"],
        "NZD": ["nzd", "kiwi", "rbnz", "new zealand", "orr", "dairy"],
        "CNH": ["cnh", "cny", "yuan", "rmb", "china", "chinese", "pboc"],
        "SGD": ["sgd", "singapore", "mas"],
        "HKD": ["hkd", "hong kong", "hkma"],
        "SEK": ["sek", "krona", "riksbank", "sweden"],
        "NOK": ["nok", "norwegian", "norges bank", "norway", "oil"],
    }

    currencies = _currencies(instrument)
    keywords = []
    for cur in currencies:
        keywords.extend(currency_keywords.get(cur, [cur.lower()]))
    
    # Add the pair itself as a keyword
    keywords.append(instrument.lower().replace("_", "/"))
    keywords.append(instrument.lower().replace("_", ""))

    headlines: List[str] = []
    seen = set()
    
    # Custom headers to avoid 403 errors
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/rss+xml, application/xml, text/xml',
    }

    for feed_url, feed_name in rss_feeds:
        if len(headlines) >= max_items:
            break
            
        try:
            req = urllib.request.Request(feed_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
        except Exception as e:
            logger.debug(f"RSS fetch failed ({feed_name}): {e}")
            continue

        # Try different XML structures (RSS 2.0, Atom)
        items = root.findall(".//item")
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        for item in items:
            # RSS 2.0 format
            title = (item.findtext("title") or "").strip()
            # Atom format fallback
            if not title:
                title_elem = item.find("{http://www.w3.org/2005/Atom}title")
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
            
            if not title:
                continue
                
            title_lc = title.lower()
            
            # Check if headline matches any relevant keywords
            if any(k in title_lc for k in keywords):
                if title_lc not in seen:
                    seen.add(title_lc)
                    headlines.append(title)
                    
            if len(headlines) >= max_items:
                break

    logger.debug(f"fetch_forex_news({instrument}) -> {len(headlines)} headlines from {len(rss_feeds)} feeds")
    return headlines


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
