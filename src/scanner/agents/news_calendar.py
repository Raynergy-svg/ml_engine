"""News Calendar Agent — Economic calendar awareness for pre-trade risk assessment.

Fetches a weekly economic calendar, classifies events by impact, and applies
time-based confidence penalties to suppress trading before high-impact releases.

Impact classification:
  HIGH:   NFP, FOMC, ECB/BOE/BOJ rate decisions, CPI — confidence -20% (30min) / -10% (2h)
  MEDIUM: PMI, Retail Sales, PPI, Trade Balance — confidence -8% (30min) / -4% (2h)
  LOW:    Minor indicators — no penalty

Cache: refreshed daily (written to .cache/economic_calendar.json). Never blocks scan.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

# ── Currency → list of currency codes that affect it ─────────────────────────
PAIR_CURRENCIES: Dict[str, Tuple[str, str]] = {
    "EUR_USD": ("EUR", "USD"),
    "GBP_USD": ("GBP", "USD"),
    "USD_JPY": ("USD", "JPY"),
    "AUD_USD": ("AUD", "USD"),
    "NZD_USD": ("NZD", "USD"),
    "USD_CAD": ("USD", "CAD"),
    "USD_CHF": ("USD", "CHF"),
}

# ── HIGH-impact event keywords ────────────────────────────────────────────────
HIGH_IMPACT_KEYWORDS = frozenset([
    "nfp", "non-farm", "payroll", "fomc", "federal reserve", "fed rate",
    "ecb rate", "ecb decision", "boe rate", "bank of england rate",
    "boj rate", "bank of japan rate", "interest rate decision",
    "cpi", "consumer price index", "inflation",
    "gdp", "gross domestic product",
    "employment change",
])

MEDIUM_IMPACT_KEYWORDS = frozenset([
    "pmi", "purchasing managers", "retail sales", "trade balance",
    "ppi", "producer price", "ism", "industrial production",
    "unemployment claims", "initial claims", "building permits",
    "housing starts", "consumer confidence", "consumer sentiment",
    "durable goods", "factory orders",
])

# ── Penalty schedule ─────────────────────────────────────────────────────────
# (minutes_before_event, confidence_penalty_fraction)
HIGH_PENALTIES = [(30, -0.20), (120, -0.10)]
MEDIUM_PENALTIES = [(30, -0.08), (120, -0.04)]


class EconomicEvent:
    """Represents a single economic calendar event."""

    __slots__ = ("timestamp", "currency", "title", "impact")

    def __init__(self, timestamp: datetime, currency: str, title: str, impact: str):
        self.timestamp = timestamp
        self.currency = currency.upper()
        self.title = title.lower()
        self.impact = impact  # "HIGH", "MEDIUM", "LOW"

    def __repr__(self) -> str:
        return f"<Event {self.impact} {self.currency} '{self.title}' @{self.timestamp.isoformat()}>"


class NewsCalendarAgent:
    """Fetches economic calendar and computes time-based confidence penalties.

    Usage:
        agent = NewsCalendarAgent()
        penalty = agent.get_confidence_penalty("EUR_USD")
        # Returns float penalty to add to confidence (e.g., -0.10)
    """

    def __init__(
        self,
        cache_path: str = ".cache/economic_calendar.json",
        calendar_url: Optional[str] = None,
        refresh_hours: int = 24,
    ):
        self.cache_path = Path(cache_path)
        # Falls back to Investing.com-compatible static seed if API unavailable
        self.calendar_url = calendar_url
        self.refresh_hours = refresh_hours
        self._events: List[EconomicEvent] = []
        self._load_or_refresh()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_confidence_penalty(self, pair: str, now: Optional[datetime] = None) -> float:
        """Compute confidence penalty for a pair based on upcoming events.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            now: Override current time (for testing)

        Returns:
            Negative float (penalty) or 0.0 if no events in window.
            The worst (most negative) penalty across both currencies is returned.
        """
        if not self._events:
            return 0.0

        if now is None:
            now = datetime.now(timezone.utc)

        currencies = PAIR_CURRENCIES.get(pair.upper(), ())
        if not currencies:
            # Parse ad-hoc pair like "EUR/USD"
            parts = pair.upper().replace("/", "_").split("_")
            currencies = tuple(parts) if len(parts) == 2 else ()

        worst_penalty = 0.0

        for event in self._events:
            if event.currency not in currencies:
                continue

            # Only care about future events within 2 hours
            minutes_until = (event.timestamp - now).total_seconds() / 60.0
            if minutes_until < 0 or minutes_until > 120:
                continue

            penalty_schedule = (
                HIGH_PENALTIES if event.impact == "HIGH"
                else MEDIUM_PENALTIES if event.impact == "MEDIUM"
                else []
            )

            for threshold_minutes, penalty in penalty_schedule:
                if minutes_until <= threshold_minutes:
                    worst_penalty = min(worst_penalty, penalty)  # min = more negative
                    break

        return worst_penalty

    def get_upcoming_events(
        self,
        pair: str,
        hours: int = 2,
        now: Optional[datetime] = None,
    ) -> List[Dict]:
        """Get upcoming events for a pair's currencies within the next N hours."""
        if now is None:
            now = datetime.now(timezone.utc)

        currencies = PAIR_CURRENCIES.get(pair.upper(), ())
        cutoff = now + timedelta(hours=hours)

        events = []
        for ev in self._events:
            if ev.currency not in currencies:
                continue
            if now <= ev.timestamp <= cutoff:
                events.append({
                    "timestamp": ev.timestamp.isoformat(),
                    "currency": ev.currency,
                    "title": ev.title,
                    "impact": ev.impact,
                    "minutes_until": round((ev.timestamp - now).total_seconds() / 60),
                })

        return sorted(events, key=lambda e: e["minutes_until"])

    def refresh(self) -> bool:
        """Force-refresh calendar from source. Returns True on success."""
        return self._fetch_and_cache()

    # ── Calendar loading ──────────────────────────────────────────────────────

    def _load_or_refresh(self) -> None:
        """Load from cache if fresh, otherwise fetch from source."""
        if self._is_cache_fresh():
            if self._load_from_cache():
                return
        self._fetch_and_cache()

    def _is_cache_fresh(self) -> bool:
        """Check if the cache file is within the refresh window."""
        if not self.cache_path.exists():
            return False
        try:
            mtime = datetime.fromtimestamp(
                self.cache_path.stat().st_mtime, tz=timezone.utc
            )
            return (datetime.now(timezone.utc) - mtime).total_seconds() < self.refresh_hours * 3600
        except OSError:
            return False

    def _load_from_cache(self) -> bool:
        """Load events from the cache JSON file."""
        try:
            raw = json.loads(self.cache_path.read_text())
            self._events = []
            for entry in raw.get("events", []):
                ts = datetime.fromisoformat(entry["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                ev = EconomicEvent(
                    timestamp=ts,
                    currency=entry["currency"],
                    title=entry["title"],
                    impact=entry["impact"],
                )
                self._events.append(ev)
            logger.debug(f"Loaded {len(self._events)} calendar events from cache")
            return True
        except Exception as e:
            logger.debug(f"Cache load failed: {e}")
            return False

    def _fetch_and_cache(self) -> bool:
        """Fetch calendar from API or generate seed events. Never raises."""
        try:
            events = self._fetch_from_api() or self._generate_seed_events()
            self._events = events
            self._write_cache(events)
            logger.info(f"NewsCalendarAgent: loaded {len(events)} events")
            return True
        except Exception as e:
            logger.warning(f"NewsCalendarAgent: calendar fetch failed ({e}), running without calendar")
            if not self._events:
                self._events = []
            return False

    def _fetch_from_api(self) -> Optional[List[EconomicEvent]]:
        """Fetch from configured calendar URL. Returns None on failure."""
        if not self.calendar_url:
            return None
        try:
            with urlopen(self.calendar_url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            events = []
            for entry in data:
                ev = self._parse_api_entry(entry)
                if ev:
                    events.append(ev)
            return events if events else None
        except (URLError, json.JSONDecodeError, Exception) as e:
            logger.debug(f"Calendar API fetch failed: {e}")
            return None

    def _parse_api_entry(self, entry: dict) -> Optional[EconomicEvent]:
        """Parse a single API entry into an EconomicEvent."""
        try:
            ts_str = entry.get("date") or entry.get("time") or entry.get("datetime")
            if not ts_str:
                return None
            ts = datetime.fromisoformat(str(ts_str))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            currency = str(entry.get("currency") or entry.get("country") or "").upper()
            title = str(entry.get("title") or entry.get("event") or entry.get("name") or "")
            impact_raw = str(entry.get("impact") or entry.get("importance") or "low").lower()
            impact = self._classify_impact(title, impact_raw)

            if not currency or not title:
                return None
            return EconomicEvent(timestamp=ts, currency=currency, title=title, impact=impact)
        except Exception:
            return None

    def _classify_impact(self, title: str, impact_raw: str) -> str:
        """Classify event impact from title keywords and raw impact field."""
        title_lower = title.lower()

        # Keyword-based override (more reliable than API fields)
        for keyword in HIGH_IMPACT_KEYWORDS:
            if keyword in title_lower:
                return "HIGH"
        for keyword in MEDIUM_IMPACT_KEYWORDS:
            if keyword in title_lower:
                return "MEDIUM"

        # Fallback to API-provided impact
        if any(h in impact_raw for h in ("high", "3", "red", "major")):
            return "HIGH"
        if any(m in impact_raw for m in ("medium", "2", "orange", "moderate")):
            return "MEDIUM"
        return "LOW"

    def _generate_seed_events(self) -> List[EconomicEvent]:
        """Generate a minimal seed of recurring weekly events for the current week.

        This covers the common major events even without API access.
        Times are approximate (actual times vary by week).
        """
        now = datetime.now(timezone.utc)
        # Start of this week (Monday)
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        # (day_offset, hour_utc, currency, title, impact)
        recurring = [
            (2, 13, "USD", "ADP Employment Change", "MEDIUM"),       # Wed
            (2, 19, "USD", "FOMC Meeting Minutes", "HIGH"),           # Wed (bi-weekly, include always)
            (3, 12, "GBP", "Bank of England rate decision", "HIGH"),  # Thu
            (3, 12, "EUR", "ECB Interest Rate Decision", "HIGH"),     # Thu
            (3, 13, "USD", "Initial Jobless Claims", "MEDIUM"),       # Thu
            (3, 15, "USD", "ISM Manufacturing PMI", "MEDIUM"),        # Thu
            (4, 13, "USD", "Non-Farm Payrolls", "HIGH"),              # Fri (monthly, include always)
            (4, 13, "USD", "Unemployment Rate", "HIGH"),              # Fri
            (4, 13, "CAD", "Employment Change", "HIGH"),              # Fri
            (1, 10, "GBP", "UK CPI", "HIGH"),                        # Tue
            (1, 13, "USD", "US CPI", "HIGH"),                        # Tue
            (1, 14, "USD", "Retail Sales", "MEDIUM"),                 # Tue
        ]

        events = []
        for day_offset, hour, currency, title, impact in recurring:
            ts = week_start + timedelta(days=day_offset, hours=hour)
            # Only include events in the next 7 days
            if now - timedelta(hours=1) <= ts <= now + timedelta(days=7):
                events.append(EconomicEvent(
                    timestamp=ts,
                    currency=currency,
                    title=title,
                    impact=impact,
                ))

        logger.debug(f"Generated {len(events)} seed calendar events for the week")
        return events

    def _write_cache(self, events: List[EconomicEvent]) -> None:
        """Write events to cache file."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "events": [
                    {
                        "timestamp": ev.timestamp.isoformat(),
                        "currency": ev.currency,
                        "title": ev.title,
                        "impact": ev.impact,
                    }
                    for ev in events
                ],
            }
            self.cache_path.write_text(json.dumps(payload, indent=2))
        except Exception as e:
            logger.debug(f"Calendar cache write failed: {e}")
