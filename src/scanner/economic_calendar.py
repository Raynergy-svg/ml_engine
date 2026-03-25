"""
Economic Calendar News Filter for Buddy FX Trading Bot.

Integrates economic calendar data to create blackout periods around
high-impact news events. Uses local cache with optional API refresh.

US-301 (Phase 48): Trade Intelligence & Adaptive Filtering.

Research basis:
- HIGH impact events cause 30-100+ pip moves in seconds
- Trading within 30min of NFP/ECB/FOMC loses ~65% of the time
- 5-minute post-event buffer lets initial spike settle
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


@dataclass
class EconomicEvent:
    """A single economic calendar event."""
    timestamp: str  # ISO format UTC
    currency: str   # e.g., "USD", "EUR", "GBP"
    impact: str     # "HIGH", "MEDIUM", "LOW"
    event_name: str
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None

    @property
    def event_time(self) -> datetime:
        """Parse timestamp to datetime."""
        try:
            return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime.now(timezone.utc)


@dataclass
class CalendarCheckResult:
    """Result of an economic calendar filter check."""
    action: str  # "CLEAR", "BLACKOUT", "REDUCE_SIZE"
    reason: str
    nearest_event: Optional[EconomicEvent] = None
    minutes_until_event: Optional[float] = None
    size_multiplier: float = 1.0  # 1.0 = normal, 0.5 = reduced, 0.0 = blocked


@dataclass
class EconomicCalendarConfig:
    """Configuration for economic calendar filtering."""

    # Blackout windows (minutes before event)
    high_impact_blackout_minutes: int = 30
    medium_impact_blackout_minutes: int = 15

    # Post-event buffer (minutes after event)
    high_impact_post_buffer_minutes: int = 5
    medium_impact_post_buffer_minutes: int = 3

    # Size reduction for medium impact
    medium_impact_size_multiplier: float = 0.50

    # Cache settings
    cache_file: str = "trained_data/economic_calendar_cache.json"
    cache_refresh_hours: int = 4

    # API settings (Finnhub)
    api_key: Optional[str] = None  # Set via env var FINNHUB_API_KEY
    api_enabled: bool = True

    # Whether to enable the filter
    enabled: bool = True

    # Currency mapping: which currencies affect which pairs
    # If a USD event is happening, block all USD_* and *_USD pairs
    version: int = 1

    def validate(self) -> None:
        """Validate configuration."""
        if self.high_impact_blackout_minutes < 0:
            raise ValueError("high_impact_blackout_minutes must be >= 0")
        if self.medium_impact_blackout_minutes < 0:
            raise ValueError("medium_impact_blackout_minutes must be >= 0")
        if not 0.0 <= self.medium_impact_size_multiplier <= 1.0:
            raise ValueError("medium_impact_size_multiplier must be 0.0-1.0")


# Mapping of currency to affected pairs
_CURRENCY_TO_PAIRS: Dict[str, List[str]] = {
    "USD": [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD",
        "USD_CAD", "NZD_USD", "USD_SGD",
    ],
    "EUR": [
        "EUR_USD", "EUR_GBP", "EUR_JPY", "EUR_AUD", "EUR_CHF",
        "EUR_NZD", "EUR_CAD",
    ],
    "GBP": [
        "GBP_USD", "EUR_GBP", "GBP_JPY", "GBP_AUD", "GBP_CHF",
        "GBP_NZD", "GBP_CAD",
    ],
    "JPY": [
        "USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY",
        "CAD_JPY", "CHF_JPY",
    ],
    "AUD": ["AUD_USD", "EUR_AUD", "GBP_AUD", "AUD_JPY", "AUD_NZD"],
    "NZD": ["NZD_USD", "EUR_NZD", "GBP_NZD", "AUD_NZD", "NZD_JPY"],
    "CAD": ["USD_CAD", "EUR_CAD", "GBP_CAD", "CAD_JPY"],
    "CHF": ["USD_CHF", "EUR_CHF", "GBP_CHF", "CHF_JPY"],
}


class EconomicCalendarFilter:
    """Filters trade entries based on proximity to economic calendar events.

    Maintains a local cache of upcoming events, optionally refreshed from
    Finnhub API. Checks if a given pair is within a blackout window of
    any relevant high/medium-impact event.

    Usage:
        filter = EconomicCalendarFilter()
        result = filter.check_pair("EUR_USD")
        if result.action == "BLACKOUT":
            logger.info(f"Skipping EUR_USD: {result.reason}")
        elif result.action == "REDUCE_SIZE":
            size *= result.size_multiplier
    """

    def __init__(self, config: Optional[EconomicCalendarConfig] = None):
        self._config = config or EconomicCalendarConfig()
        try:
            self._config.validate()
        except ValueError as e:
            logger.warning(f"EconomicCalendar config invalid: {e}, using defaults")
            self._config = EconomicCalendarConfig()

        self._events: List[EconomicEvent] = []
        self._cache_loaded_at: Optional[float] = None
        self._load_cache()

    @property
    def config(self) -> EconomicCalendarConfig:
        return self._config

    @property
    def events(self) -> List[EconomicEvent]:
        return self._events

    def _load_cache(self) -> None:
        """Load events from local cache file."""
        cache_path = Path(self._config.cache_file)
        if not cache_path.exists():
            logger.debug("No economic calendar cache found")
            return

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.warning("Economic calendar cache is not a list, ignoring")
                return

            self._events = []
            for item in data:
                if isinstance(item, dict):
                    try:
                        self._events.append(EconomicEvent(
                            timestamp=item.get("timestamp", ""),
                            currency=item.get("currency", ""),
                            impact=item.get("impact", "LOW"),
                            event_name=item.get("event_name", "Unknown"),
                            forecast=item.get("forecast"),
                            previous=item.get("previous"),
                            actual=item.get("actual"),
                        ))
                    except Exception as e:
                        logger.debug(f"Skipping malformed event: {e}")

            self._cache_loaded_at = time.time()
            logger.info(f"Loaded {len(self._events)} economic events from cache")

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load economic calendar cache: {e}")

    def save_cache(self) -> None:
        """Save current events to cache (atomic write)."""
        cache_path = Path(self._config.cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for evt in self._events:
            data.append({
                "timestamp": evt.timestamp,
                "currency": evt.currency,
                "impact": evt.impact,
                "event_name": evt.event_name,
                "forecast": evt.forecast,
                "previous": evt.previous,
                "actual": evt.actual,
            })

        tmp_path = str(cache_path) + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp_path, str(cache_path))
            logger.debug(f"Saved {len(data)} events to cache")
        except OSError as e:
            logger.warning(f"Failed to save economic calendar cache: {e}")

    def update_events(self, events: List[EconomicEvent]) -> None:
        """Update the event list (e.g., from API fetch)."""
        self._events = events
        self._cache_loaded_at = time.time()
        self.save_cache()

    def add_events(self, events: List[EconomicEvent]) -> None:
        """Add events without replacing existing ones."""
        existing_keys = {
            (e.timestamp, e.currency, e.event_name) for e in self._events
        }
        for evt in events:
            key = (evt.timestamp, evt.currency, evt.event_name)
            if key not in existing_keys:
                self._events.append(evt)
                existing_keys.add(key)
        self.save_cache()

    def _get_affected_pairs(self, currency: str) -> List[str]:
        """Get pairs affected by events in a given currency."""
        return _CURRENCY_TO_PAIRS.get(currency.upper(), [])

    def _is_pair_affected(self, pair: str, currency: str) -> bool:
        """Check if a pair is affected by an event currency."""
        affected = self._get_affected_pairs(currency)
        return pair in affected

    def check_pair(
        self,
        pair: str,
        check_time: Optional[datetime] = None,
    ) -> CalendarCheckResult:
        """Check if a pair is in a news blackout window.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            check_time: Time to check against (defaults to now UTC)

        Returns:
            CalendarCheckResult with action and details
        """
        if not self._config.enabled:
            return CalendarCheckResult(
                action="CLEAR",
                reason="economic_calendar_disabled",
                size_multiplier=1.0,
            )

        if not self._events:
            return CalendarCheckResult(
                action="CLEAR",
                reason="no_events_loaded",
                size_multiplier=1.0,
            )

        now = check_time or datetime.now(timezone.utc)
        nearest_event: Optional[EconomicEvent] = None
        nearest_minutes: Optional[float] = None
        worst_action = "CLEAR"
        worst_multiplier = 1.0

        for event in self._events:
            # Check if this event's currency affects the pair
            if not self._is_pair_affected(pair, event.currency):
                continue

            event_time = event.event_time
            delta = (event_time - now).total_seconds() / 60.0  # minutes

            # Track nearest relevant event
            if nearest_minutes is None or abs(delta) < abs(nearest_minutes):
                nearest_event = event
                nearest_minutes = delta

            impact = event.impact.upper()

            # PRE-EVENT: check blackout window
            if impact == "HIGH" and 0 < delta <= self._config.high_impact_blackout_minutes:
                worst_action = "BLACKOUT"
                worst_multiplier = 0.0
                logger.info(
                    f"EconomicCalendar BLACKOUT {pair}: HIGH impact "
                    f"'{event.event_name}' in {delta:.0f}min"
                )
            elif impact == "MEDIUM" and 0 < delta <= self._config.medium_impact_blackout_minutes:
                if worst_action != "BLACKOUT":
                    worst_action = "REDUCE_SIZE"
                    worst_multiplier = min(
                        worst_multiplier,
                        self._config.medium_impact_size_multiplier,
                    )

            # POST-EVENT: check buffer window
            if impact == "HIGH" and -self._config.high_impact_post_buffer_minutes <= delta <= 0:
                worst_action = "BLACKOUT"
                worst_multiplier = 0.0
                logger.info(
                    f"EconomicCalendar BLACKOUT {pair}: HIGH impact "
                    f"'{event.event_name}' released {abs(delta):.0f}min ago"
                )
            elif impact == "MEDIUM" and -self._config.medium_impact_post_buffer_minutes <= delta <= 0:
                if worst_action != "BLACKOUT":
                    worst_action = "REDUCE_SIZE"
                    worst_multiplier = min(
                        worst_multiplier,
                        self._config.medium_impact_size_multiplier,
                    )

        reason = "clear_no_nearby_events"
        if worst_action == "BLACKOUT":
            reason = (
                f"news_blackout: {nearest_event.event_name} "
                f"({nearest_event.impact}) "
                f"{'in' if nearest_minutes > 0 else ''} "
                f"{abs(nearest_minutes):.0f}min "
                f"{'before' if nearest_minutes > 0 else 'after'}"
            )
        elif worst_action == "REDUCE_SIZE":
            reason = (
                f"news_reduce_size: {nearest_event.event_name} "
                f"({nearest_event.impact}) nearby"
            )

        return CalendarCheckResult(
            action=worst_action,
            reason=reason,
            nearest_event=nearest_event,
            minutes_until_event=nearest_minutes,
            size_multiplier=worst_multiplier,
        )

    def get_upcoming_events(
        self,
        hours_ahead: int = 4,
        check_time: Optional[datetime] = None,
    ) -> List[EconomicEvent]:
        """Get events in the next N hours."""
        now = check_time or datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        upcoming = []
        for event in self._events:
            evt_time = event.event_time
            if now <= evt_time <= cutoff:
                upcoming.append(event)
        return sorted(upcoming, key=lambda e: e.event_time)

    def is_cache_stale(self) -> bool:
        """Check if the cache needs refreshing."""
        if self._cache_loaded_at is None:
            return True
        age_hours = (time.time() - self._cache_loaded_at) / 3600.0
        return age_hours > self._config.cache_refresh_hours


def create_default_calendar_filter(
    config: Optional[EconomicCalendarConfig] = None,
) -> EconomicCalendarFilter:
    """Factory function for EconomicCalendarFilter."""
    if config is None:
        config = EconomicCalendarConfig()
        # Try to load API key from environment
        api_key = os.environ.get("FINNHUB_API_KEY")
        if api_key:
            config.api_key = api_key
    return EconomicCalendarFilter(config=config)
