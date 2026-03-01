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
        text.lower()
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
# PHASE 5: PRE-TRADE RISK ASSESSMENT (LLM-Enhanced)
# =============================================================================

@dataclass
class EventRiskAssessment:
    """Result of LLM-powered event risk assessment."""
    avoid_trade: bool
    volatility: str  # 'low', 'medium', 'high', 'extreme'
    bias: str  # 'bullish', 'bearish', 'none'
    reason: str
    event_name: Optional[str] = None
    minutes_until: Optional[float] = None
    surprise_direction: Optional[str] = None  # For past events: 'beat', 'miss', 'inline'

    @classmethod
    def safe(cls) -> "EventRiskAssessment":
        """Return safe assessment (no events to worry about)."""
        return cls(
            avoid_trade=False,
            volatility="low",
            bias="none",
            reason="No significant events affecting this pair",
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any], event: Optional[EconomicEvent] = None) -> "EventRiskAssessment":
        """Create from parsed JSON."""
        return cls(
            avoid_trade=bool(d.get("avoid_trade", False)),
            volatility=d.get("volatility", "medium"),
            bias=d.get("bias", "none"),
            reason=d.get("reason", ""),
            event_name=event.name if event else None,
            minutes_until=event.minutes_until if event else None,
            surprise_direction=d.get("surprise_direction"),
        )


@dataclass
class MultiFactorRiskScore:
    """Comprehensive risk score combining all factors."""
    score: float  # 0-100 (higher = more risk)
    volatility_risk: float  # 0-100
    event_risk: float  # 0-100
    drawdown_risk: float  # 0-100
    sentiment_risk: float  # 0-100
    time_of_day_risk: float  # 0-100
    recommendation: str  # 'trade', 'reduce_size', 'avoid'
    size_multiplier: float  # 0.25 to 1.0
    factors: List[str]  # List of risk factors

    @classmethod
    def low_risk(cls) -> "MultiFactorRiskScore":
        """Return low risk assessment (safe to trade)."""
        return cls(
            score=20.0,
            volatility_risk=20.0,
            event_risk=10.0,
            drawdown_risk=20.0,
            sentiment_risk=20.0,
            time_of_day_risk=20.0,
            recommendation="trade",
            size_multiplier=1.0,
            factors=[],
        )


def _format_events_for_prompt(events: List[EconomicEvent]) -> str:
    """Format events list for LLM prompt."""
    if not events:
        return "No upcoming events"

    lines = []
    for e in events[:5]:  # Limit to 5 events
        time_str = f"{int(e.minutes_until)}min" if e.minutes_until > 0 else f"{int(-e.minutes_until)}min ago"
        actual_str = f" (Actual: {e.actual})" if e.actual else ""
        forecast_str = f" vs Forecast: {e.forecast}" if e.forecast else ""
        lines.append(f"- [{e.currency}] {e.name} ({e.impact}) in {time_str}{actual_str}{forecast_str}")

    return "\n".join(lines)


LLM_EVENT_RISK_SYSTEM = """You are an FX economic event risk analyst. Assess how upcoming or recent economic events affect trading safety.

Consider:
1. Time until event (avoid 30min before, 15min after high-impact)
2. If event already happened, assess the surprise direction (beat/miss/inline)
3. Expected volatility level
4. Directional bias introduced by the event

Respond with JSON only:
{
  "avoid_trade": true/false,
  "volatility": "low" | "medium" | "high" | "extreme",
  "bias": "bullish" | "bearish" | "none",
  "surprise_direction": "beat" | "miss" | "inline" | null,
  "reason": "Brief explanation"
}"""


def assess_event_risk(
    events: List[EconomicEvent],
    instrument: str,
    llm_call_fn: Optional[Callable] = None,
) -> EventRiskAssessment:
    """
    Use LLM to assess risk from upcoming/recent economic events.

    This is Phase 5.1 from LLM_INTEGRATION_PLAN.md.

    Args:
        events: List of economic events affecting the instrument
        instrument: Currency pair (e.g., "USD_JPY")
        llm_call_fn: Optional LLM call function (defaults to trying buddy_intelligent_mode)

    Returns:
        EventRiskAssessment with avoid_trade, volatility, bias, reason
    """
    # Quick return if no events
    if not events:
        return EventRiskAssessment.safe()

    # Filter to relevant events (next 2 hours or just happened)
    relevant = [e for e in events if -30 <= e.minutes_until <= 120]
    if not relevant:
        return EventRiskAssessment.safe()

    # Check for high-impact events first (quick rule-based check)
    high_impact = [e for e in relevant if e.is_high_impact]
    for e in high_impact:
        if 0 <= e.minutes_until <= 30:
            return EventRiskAssessment(
                avoid_trade=True,
                volatility="extreme",
                bias="none",
                reason=f"High-impact event '{e.name}' in {int(e.minutes_until)} minutes",
                event_name=e.name,
                minutes_until=e.minutes_until,
            )
        if -15 <= e.minutes_until < 0:
            return EventRiskAssessment(
                avoid_trade=True,
                volatility="extreme",
                bias="none",
                reason=f"High-impact event '{e.name}' just occurred {int(-e.minutes_until)} minutes ago",
                event_name=e.name,
                minutes_until=e.minutes_until,
            )

    # For edge cases (30-60 min before, or medium-impact), use LLM if available
    if llm_call_fn is None:
        try:
            from buddy_intelligent_mode import llm_call
            llm_call_fn = llm_call
        except ImportError:
            # No LLM available, use rule-based assessment
            for e in relevant:
                if e.is_high_impact and e.minutes_until <= 60:
                    return EventRiskAssessment(
                        avoid_trade=False,
                        volatility="high",
                        bias="none",
                        reason=f"High-impact event '{e.name}' in {int(e.minutes_until)} minutes - trade with caution",
                        event_name=e.name,
                        minutes_until=e.minutes_until,
                    )
            return EventRiskAssessment.safe()

    # Build LLM prompt
    events_formatted = _format_events_for_prompt(relevant)
    prompt = f"""Assess event risk for {instrument}:

Current time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

Upcoming/Recent Events:
{events_formatted}

Should I avoid trading right now? Respond with JSON:"""

    response = llm_call_fn(
        prompt=prompt,
        system_prompt=LLM_EVENT_RISK_SYSTEM,
        temperature=0.1,
    )

    if response is None:
        # Fallback to rule-based
        return EventRiskAssessment.safe()

    # Parse JSON response
    try:
        import re
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            closest_event = min(relevant, key=lambda e: abs(e.minutes_until))
            return EventRiskAssessment.from_dict(result, closest_event)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM event risk response: {e}")

    return EventRiskAssessment.safe()


LLM_RISK_SCORE_SYSTEM = """You are a comprehensive FX trading risk assessor. Compute an overall risk score (0-100) considering ALL factors.

Factors to consider:
1. Market volatility (ATR, recent range)
2. Economic events (upcoming/recent)
3. Recent drawdown/streak
4. Sentiment alignment/contradiction
5. Time of day (session, liquidity)

Risk score interpretation:
- 0-25: Low risk - trade normally
- 25-50: Moderate risk - consider smaller size
- 50-75: High risk - reduce size significantly or avoid
- 75-100: Extreme risk - do not trade

Respond with JSON only:
{
  "score": 0-100,
  "volatility_risk": 0-100,
  "event_risk": 0-100,
  "drawdown_risk": 0-100,
  "sentiment_risk": 0-100,
  "time_of_day_risk": 0-100,
  "recommendation": "trade" | "reduce_size" | "avoid",
  "size_multiplier": 0.25 to 1.0,
  "factors": ["list", "of", "concerns"]
}"""


def compute_llm_risk_score(
    context: Dict[str, Any],
    llm_call_fn: Optional[Callable] = None,
) -> MultiFactorRiskScore:
    """
    Use LLM to compute comprehensive 0-100 risk score considering ALL factors.

    This is Phase 5.2 from LLM_INTEGRATION_PLAN.md.

    Args:
        context: Dict containing:
            - instrument: Currency pair
            - volatility: ATR or volatility measure
            - events: List of upcoming events or event summary
            - recent_drawdown: Recent drawdown percentage
            - sentiment: Sentiment score (-1 to 1)
            - rsi: Current RSI
            - adx: Current ADX
            - time_of_day: Current hour (UTC)
            - recent_win_rate: Optional recent win rate
        llm_call_fn: Optional LLM call function

    Returns:
        MultiFactorRiskScore with comprehensive risk assessment
    """
    # Quick risk calculation without LLM
    score = 20.0  # Base score
    factors = []

    # Volatility risk
    volatility = context.get('volatility', 0.0)
    atr_pct = context.get('atr_pct', volatility)
    volatility_risk = min(100, atr_pct * 2000)  # 5% ATR = 100 risk
    if atr_pct > 0.02:
        factors.append(f"High volatility ({atr_pct:.1%})")

    # Event risk
    event_risk = 10.0
    events = context.get('events', [])
    event_summary = context.get('event_summary', '')
    if events or 'high-impact' in str(event_summary).lower():
        event_risk = 60.0
        factors.append("High-impact event nearby")

    # Drawdown risk
    drawdown_risk = 20.0
    recent_dd = context.get('recent_drawdown', 0.0)
    if recent_dd > 0.03:
        drawdown_risk = min(100, recent_dd * 2000)
        factors.append(f"Recent drawdown ({recent_dd:.1%})")

    # Sentiment risk (contradiction = high risk)
    sentiment_risk = 20.0
    sentiment = context.get('sentiment', 0.0)
    direction = context.get('proposed_direction', 'long')
    if (direction == 'long' and sentiment < -0.5) or (direction == 'short' and sentiment > 0.5):
        sentiment_risk = 70.0
        factors.append("Sentiment contradicts direction")

    # Time of day risk
    time_risk = 20.0
    hour = context.get('time_of_day', datetime.utcnow().hour)
    if hour >= 21 or hour < 1:  # Late US/early Asia gap
        time_risk = 40.0
        factors.append("Low liquidity session")

    # Compute overall score
    score = (volatility_risk * 0.25 + event_risk * 0.25 + drawdown_risk * 0.20 +
             sentiment_risk * 0.15 + time_risk * 0.15)

    # Try LLM for edge cases (score 40-70 range)
    if llm_call_fn is None and 40 <= score <= 70:
        try:
            from buddy_intelligent_mode import llm_call
            llm_call_fn = llm_call
        except ImportError:
            pass

    if llm_call_fn is not None and 40 <= score <= 70:
        prompt = f"""Compute risk score for {context.get('instrument', 'FX pair')}:

Context:
- Volatility (ATR%): {atr_pct:.2%}
- RSI: {context.get('rsi', 'N/A')}
- ADX: {context.get('adx', 'N/A')}
- Events: {event_summary or 'None imminent'}
- Sentiment: {sentiment:+.2f}
- Proposed Direction: {direction}
- Recent Drawdown: {recent_dd:.1%}
- Time (UTC): {hour}:00

Current rule-based score: {score:.0f}
Identified factors: {factors}

Provide your assessment as JSON:"""

        response = llm_call_fn(
            prompt=prompt,
            system_prompt=LLM_RISK_SCORE_SYSTEM,
            temperature=0.1,
        )

        if response:
            try:
                import re
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return MultiFactorRiskScore(
                        score=float(result.get('score', score)),
                        volatility_risk=float(result.get('volatility_risk', volatility_risk)),
                        event_risk=float(result.get('event_risk', event_risk)),
                        drawdown_risk=float(result.get('drawdown_risk', drawdown_risk)),
                        sentiment_risk=float(result.get('sentiment_risk', sentiment_risk)),
                        time_of_day_risk=float(result.get('time_of_day_risk', time_risk)),
                        recommendation=result.get('recommendation', 'trade'),
                        size_multiplier=max(0.25, min(1.0, float(result.get('size_multiplier', 1.0)))),
                        factors=result.get('factors', factors),
                    )
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"Failed to parse LLM risk score: {e}")

    # Return rule-based score
    recommendation = "trade"
    size_mult = 1.0
    if score > 75:
        recommendation = "avoid"
        size_mult = 0.0
    elif score > 50:
        recommendation = "reduce_size"
        size_mult = 0.5
    elif score > 35:
        size_mult = 0.75

    return MultiFactorRiskScore(
        score=score,
        volatility_risk=volatility_risk,
        event_risk=event_risk,
        drawdown_risk=drawdown_risk,
        sentiment_risk=sentiment_risk,
        time_of_day_risk=time_risk,
        recommendation=recommendation,
        size_multiplier=size_mult,
        factors=factors,
    )


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
    metadata: Optional[Dict[str, Any]] = None

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
        memory_client: Optional[Any] = None,
    ):
        self.buffer_size = buffer_size
        self.retrain_threshold = retrain_threshold
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.trade_buffer: List[TradeOutcome] = []
        self.trades_since_retrain = 0
        self._memory_client = memory_client
        self._init_memory_client()
        self._migrate_legacy_buffer_to_memory()
        self._load_runtime_state()
        self._refresh_trade_buffer_from_memory()

    def _runtime_namespace(self) -> str:
        """Namespace for persisted online learner runtime state."""
        return "online_learning"

    def _init_memory_client(self) -> None:
        """Initialize shared trade ledger if available."""
        if self._memory_client is not None:
            return
        try:
            from memory_client import MLEngineMemory
            self._memory_client = MLEngineMemory()
        except Exception as e:
            logger.debug(f"Shared memory ledger unavailable: {e}")

    def _load_runtime_state(self) -> None:
        """Load online learner counters from shared runtime state."""
        if self._memory_client is None:
            self.trades_since_retrain = 0
            return

        try:
            state = self._memory_client.get_runtime_state(self._runtime_namespace())
            self.trades_since_retrain = int(state.get("trades_since_retrain", 0) or 0)
        except Exception as e:
            logger.debug(f"Failed to load online learner runtime state: {e}")
            self.trades_since_retrain = 0

    def _save_runtime_state(self) -> None:
        """Persist online learner counters into shared runtime state."""
        if self._memory_client is None:
            return

        try:
            self._memory_client.update_runtime_state(
                self._runtime_namespace(),
                {
                    "trades_since_retrain": int(self.trades_since_retrain),
                    "buffer_size": int(self.buffer_size),
                    "retrain_threshold": int(self.retrain_threshold),
                },
            )
        except Exception as e:
            logger.debug(f"Failed to save online learner runtime state: {e}")

    def _serialize_trade(self, trade: TradeOutcome) -> Dict[str, Any]:
        """Convert a TradeOutcome into the shared ledger schema."""
        metadata = dict(trade.metadata or {})
        metadata.update({
            "actual_outcome": int(trade.actual_outcome),
            "direction_code": int(trade.direction),
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat(),
            "feature_count": int(len(trade.features)),
            "features": np.asarray(trade.features, dtype=float).tolist(),
            "pnl_percent": float(trade.pnl_percent),
        })
        return {
            "timestamp": trade.exit_time.isoformat(),
            "trade_id": trade.trade_id,
            "instrument": trade.instrument,
            "direction": "long" if int(trade.direction) == 1 else "short",
            "confidence": float(trade.confidence),
            "prediction": float(trade.prediction),
            "entry": float(trade.entry_price),
            "exit": float(trade.exit_price),
            "pnl": float(trade.pnl_percent),
            "pnl_pips": float(trade.pnl_pips),
            "model": "market_intelligence",
            "metadata": metadata,
        }

    def _trade_from_ledger(self, record: Dict[str, Any]) -> Optional[TradeOutcome]:
        """Reconstruct a TradeOutcome from a shared ledger record."""
        metadata = record.get("metadata") or {}
        features = metadata.get("features")
        if not isinstance(features, list) or not features:
            return None

        entry_time_raw = metadata.get("entry_time") or record.get("timestamp")
        exit_time_raw = metadata.get("exit_time") or record.get("timestamp")
        if not entry_time_raw or not exit_time_raw:
            return None

        try:
            direction_code = metadata.get("direction_code")
            if direction_code is None:
                direction_label = str(record.get("direction", "")).lower()
                direction_code = 1 if direction_label in {"1", "long", "buy"} else 0

            pnl_percent = metadata.get("pnl_percent")
            if pnl_percent is None:
                pnl_percent = record.get("pnl", 0.0)

            actual_outcome = metadata.get("actual_outcome")
            if actual_outcome is None:
                actual_outcome = 1 if float(record.get("pnl_pips") or 0.0) > 0 else 0

            return TradeOutcome(
                trade_id=str(record.get("trade_id") or ""),
                instrument=str(record.get("instrument") or ""),
                direction=int(direction_code),
                entry_time=datetime.fromisoformat(str(entry_time_raw)),
                exit_time=datetime.fromisoformat(str(exit_time_raw)),
                entry_price=float(record.get("entry") or 0.0),
                exit_price=float(record.get("exit") or 0.0),
                pnl_pips=float(record.get("pnl_pips") or 0.0),
                pnl_percent=float(pnl_percent or 0.0),
                features=np.array(features, dtype=float),
                prediction=float(record.get("prediction") or 0.0),
                confidence=float(record.get("confidence") or 0.0),
                actual_outcome=int(actual_outcome),
                metadata=dict(metadata),
            )
        except Exception as e:
            trade_id = record.get("trade_id") or record.get("instrument") or "unknown"
            logger.debug(f"Skipping malformed online-learning trade {trade_id}: {e}")
            return None

    def _refresh_trade_buffer_from_memory(self) -> None:
        """Rebuild the in-memory buffer from the shared ledger."""
        if self._memory_client is None:
            self.trade_buffer = []
            return

        try:
            search_limit = max(self.buffer_size * 4, self.buffer_size)
            recent_trades = self._memory_client.get_recent_trades(limit=search_limit)
            rebuilt: List[TradeOutcome] = []
            for record in recent_trades:
                if record.get("model") != "market_intelligence":
                    continue
                trade = self._trade_from_ledger(record)
                if trade is None:
                    continue
                rebuilt.append(trade)
                if len(rebuilt) >= self.buffer_size:
                    break
            self.trade_buffer = list(reversed(rebuilt))
        except Exception as e:
            logger.warning(f"Failed to refresh online learner buffer from shared memory: {e}")
            self.trade_buffer = []

    def _load_legacy_trade(self, item: Dict[str, Any]) -> Optional[TradeOutcome]:
        """Convert a legacy trade_buffer.json entry into TradeOutcome."""
        try:
            return TradeOutcome(
                trade_id=item["trade_id"],
                instrument=item["instrument"],
                direction=int(item["direction"]),
                entry_time=datetime.fromisoformat(item["entry_time"]),
                exit_time=datetime.fromisoformat(item["exit_time"]),
                entry_price=float(item["entry_price"]),
                exit_price=float(item["exit_price"]),
                pnl_pips=float(item["pnl_pips"]),
                pnl_percent=float(item["pnl_percent"]),
                features=np.array(item["features"], dtype=float),
                prediction=float(item["prediction"]),
                confidence=float(item["confidence"]),
                actual_outcome=int(item["actual_outcome"]),
            )
        except Exception as e:
            trade_id = item.get("trade_id") or item.get("instrument") or "unknown"
            logger.debug(f"Skipping malformed legacy trade buffer entry {trade_id}: {e}")
            return None

    def _migrate_legacy_buffer_to_memory(self) -> None:
        """Import one legacy trade_buffer.json into shared memory."""
        if self._memory_client is None:
            return

        buffer_file = self.storage_dir / "trade_buffer.json"
        if not buffer_file.exists():
            return

        try:
            runtime_state = self._memory_client.get_runtime_state(self._runtime_namespace())
            if runtime_state.get("legacy_trade_buffer_migrated"):
                return

            data = json.loads(buffer_file.read_text())
            migrated = 0
            for item in data.get("trades", []):
                trade = self._load_legacy_trade(item)
                if trade is None:
                    continue
                self._memory_client.log_trade(self._serialize_trade(trade))
                migrated += 1

            state_update = {
                "legacy_trade_buffer_migrated": True,
                "legacy_trade_buffer_path": str(buffer_file),
            }
            if "trades_since_retrain" not in runtime_state:
                state_update["trades_since_retrain"] = int(data.get("trades_since_retrain", 0) or 0)
            self._memory_client.update_runtime_state(self._runtime_namespace(), state_update)

            if migrated:
                logger.info(f"Migrated {migrated} online-learning trades from legacy buffer")
        except Exception as e:
            logger.warning(f"Failed to migrate legacy online-learning buffer: {e}")

    def _persist_trade_to_memory(self, trade: TradeOutcome) -> None:
        """Insert or update a completed trade in shared memory."""
        if self._memory_client is None:
            return
        try:
            self._memory_client.log_trade(self._serialize_trade(trade))
        except Exception as e:
            logger.debug(f"Failed to persist trade to shared memory: {e}")

    def record_trade(self, trade: TradeOutcome):
        """
        Record a completed trade for learning.

        Args:
            trade: TradeOutcome with features and actual result
        """
        already_recorded = False
        if self._memory_client is not None and trade.trade_id:
            already_recorded = self._memory_client.get_trade(trade.trade_id) is not None
        elif trade.trade_id:
            already_recorded = any(t.trade_id == trade.trade_id for t in self.trade_buffer)

        self._persist_trade_to_memory(trade)
        self._refresh_trade_buffer_from_memory()

        if not already_recorded:
            self.trades_since_retrain += 1
            self._save_runtime_state()

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
        self._save_runtime_state()
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
            with open(state_path) as f:
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

        logger.debug(
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
        context_metadata: Optional[Dict[str, Any]] = None,
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
                metadata=context_metadata or {},
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

    def assess_pre_trade_risk(
        self,
        instrument: str,
        market_context: Optional[Dict[str, Any]] = None,
        proposed_direction: str = "long",
        llm_call_fn: Optional[Callable] = None,
    ) -> Tuple[bool, MultiFactorRiskScore, Optional[EventRiskAssessment]]:
        """
        Comprehensive pre-trade risk assessment (Phase 5).

        Combines:
        1. Economic event risk assessment (LLM-enhanced)
        2. Multi-factor risk score (volatility, sentiment, time, etc.)

        Args:
            instrument: Currency pair
            market_context: Dict with volatility, RSI, ADX, sentiment, etc.
            proposed_direction: 'long' or 'short'
            llm_call_fn: Optional LLM call function

        Returns:
            Tuple of (can_trade, risk_score, event_risk)
        """
        market_context = market_context or {}
        market_context['instrument'] = instrument
        market_context['proposed_direction'] = proposed_direction

        # 1. Assess event risk
        event_risk = None
        if self.calendar:
            events = self.calendar.fetch_events(instrument, hours_ahead=4)
            event_risk = assess_event_risk(events, instrument, llm_call_fn)

            # Block trade if event risk says avoid
            if event_risk.avoid_trade:
                return False, MultiFactorRiskScore.low_risk(), event_risk

            # Add event info to context
            market_context['events'] = events
            market_context['event_summary'] = event_risk.reason

        # 2. Compute multi-factor risk score
        risk_score = compute_llm_risk_score(market_context, llm_call_fn)

        # 3. Determine if we should trade
        can_trade = risk_score.recommendation != "avoid"

        return can_trade, risk_score, event_risk


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
