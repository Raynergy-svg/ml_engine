"""News data sources (Phase 2 implementation for ForexFactory).

Defines the abstract ``NewsSource`` contract and the ``NewsEvent`` payload that
flows through the pipeline:

    NewsSource.fetch_events(pair, since, until) -> List[NewsEvent]
                                                  |
                                                  v
    NewsEmbedder.embed(events) -> np.ndarray (n_events, embedding_dim)
                                                  |
                                                  v
    align_news_to_bars(events, bar_timestamps, lookback_window_hours)
                                                  |
                                                  v
    pd.DataFrame   (per-bar fused features) -> compute_normalized_features

See ``docs/superpowers/plans/2026-05-08-news-macro-signal-design.md`` §1 + §4
for design rationale (data-source comparison + time-alignment algorithm).

Phase 2 (this commit): ``ForexFactoryNewsSource.fetch_events`` is a real
implementation backed by the public FF JSON feed
(``https://nfs.faireconomy.media/ff_calendar_thisweek.json``) plus a parquet
cache (``trained_data/news/{pair}_ff_events.parquet``) for idempotent
backfill across weeks.
"""

from __future__ import annotations

import abc
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NewsEvent:
    """A single news / macro event at a known release time.

    Attributes:
        timestamp: UTC datetime of the event's *release* (not scrape time). The
            time-alignment guard in ``align_news_to_bars`` enforces a strict
            ``event.timestamp < bar.timestamp`` open-interval check to prevent
            lookahead bias on the current bar.
        text: Headline / event title / statement text. For calendar events
            (e.g. NFP), this is typically "Non-Farm Payrolls" or similar; for
            RSS headlines, the raw headline text. Whitespace normalized,
            HTML entities decoded; embedder will tokenize.
        source: Provenance string. One of {"forex_factory", "rss", "newsapi",
            "alpaca", "manual"}. Lets the pipeline trace which feeds contributed
            which signal.
        category: Coarse classification. One of {"NFP", "CPI", "GDP", "FOMC",
            "ECB", "BoE", "BoJ", "BoC", "RBA", "RBNZ", "SNB", "OTHER"}. Used
            for the ``event_class_count`` one-hot in ``align_news_to_bars``.
        relevance_score: float in [0.0, 1.0]; how relevant this event is to
            the queried ``pair``. ForexFactory provides per-currency impact
            (low/medium/high) which maps to {0.33, 0.66, 1.0}. RSS scoring
            is left to the source implementation. Used as a multiplier on the
            time-decay weight in alignment.
        pair: Optional currency pair (e.g. "EUR_USD") for which this event was
            fetched. NewsAPI/RSS sources may emit per-pair queries; ForexFactory
            typically emits per-currency events that we expand to all pairs
            sharing the currency.
        impact: Optional impact label from the source ("low", "medium", "high",
            or None). Mirrors ForexFactory's classification.
    """

    timestamp: datetime
    text: str
    source: str
    category: str
    relevance_score: float
    pair: Optional[str] = None
    impact: Optional[str] = None

    def __post_init__(self) -> None:
        # Defensive validation — surface bad data at construction, not at
        # embed/align time. Honors .claude/rules/improvement.md
        # "Silent Exception Prevention".
        if not isinstance(self.timestamp, datetime):
            raise TypeError(
                f"NewsEvent.timestamp must be datetime, got {type(self.timestamp).__name__}"
            )
        if self.timestamp.tzinfo is None:
            raise ValueError(
                "NewsEvent.timestamp must be timezone-aware (UTC) to avoid "
                "lookahead-bias from naive-vs-aware comparisons during alignment."
            )
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("NewsEvent.text must be a non-empty string")
        if not (0.0 <= float(self.relevance_score) <= 1.0):
            raise ValueError(
                f"NewsEvent.relevance_score must be in [0, 1], got {self.relevance_score}"
            )


class NewsSource(abc.ABC):
    """Abstract base for any news/event provider.

    Concrete implementations must:
      - Be deterministic for a given (pair, since, until) tuple — backfill
        runs are repeatable; tests are reproducible.
      - Return ``NewsEvent`` instances with timezone-aware UTC timestamps.
      - Raise (not silently return empty) on transport failure. Empty list
        is reserved for "queried successfully, no events in window".
      - Tolerate ``until > now()`` by returning only events <= now().
    """

    @abc.abstractmethod
    def fetch_events(
        self,
        pair: str,
        since: datetime,
        until: datetime,
    ) -> List[NewsEvent]:
        """Fetch events relevant to ``pair`` between ``since`` and ``until``.

        Args:
            pair: Currency pair, e.g. "EUR_USD" (OANDA convention; underscore-
                separated). Implementations expand to per-currency queries
                (EUR + USD events both relevant to EUR_USD).
            since: UTC datetime, inclusive lower bound on event timestamps.
            until: UTC datetime, exclusive upper bound on event timestamps.

        Returns:
            List of ``NewsEvent`` instances, sorted ascending by timestamp.
            Empty list if no events fell in the window.

        Raises:
            ValueError: if since >= until or pair is malformed.
            RuntimeError: if the upstream feed errors. Caller decides whether
                to retry, fall back, or fail loudly.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ForexFactory implementation
# ---------------------------------------------------------------------------


# Locked category set per NewsEvent contract docstring above. Order matters
# for the heuristic (longer / more-specific patterns first so "FOMC Statement"
# doesn't get caught by a generic "Statement" rule before matching FOMC).
_VALID_CATEGORIES = {
    "NFP", "CPI", "GDP", "FOMC", "ECB", "BoE", "BoJ",
    "BoC", "RBA", "RBNZ", "SNB", "OTHER",
}

# Currency -> central-bank category. Lets us map "Monetary Policy Statement"
# to BoJ when country=JPY etc. without a per-bank string match.
_CB_BY_CURRENCY = {
    "USD": "FOMC",
    "EUR": "ECB",
    "GBP": "BoE",
    "JPY": "BoJ",
    "CAD": "BoC",
    "AUD": "RBA",
    "NZD": "RBNZ",
    "CHF": "SNB",
}

# FF impact label -> relevance_score in [0,1]. Holiday / "non-economic" rows
# get 0.0 so they're effectively ignored downstream while still recorded for
# auditability.
_IMPACT_RELEVANCE = {
    "low": 0.33,
    "medium": 0.66,
    "high": 1.0,
    "holiday": 0.0,
    "non-economic": 0.0,
    "": 0.33,  # FF sometimes emits empty impact for borderline events
}

_FF_JSON_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_FF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Buddy-FX/1.0",
    "Accept": "application/json",
    "Referer": "https://www.forexfactory.com/",
}

# HTML scraper config (Stage 4-A.ii). FF blocks bot UAs at the Cloudflare layer
# but accepts a vanilla Firefox UA after a homepage warm-up that sets the
# ``__cf_bm`` cookie. We session-cache cookies across week fetches to avoid
# tripping rate-limits on every request. Verified empirically 2026-05-08.
_FF_HTML_BASE = "https://www.forexfactory.com"
_FF_HTML_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7; rv:124.0) "
                  "Gecko/20100101 Firefox/124.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
# FF rate-limits aggressively; observed 429 with <2s gaps. 2.5s gives margin.
_FF_HTML_MIN_INTERVAL_SEC = 2.5
# Month abbreviation order matches FF's URL convention (lowercase, e.g. "mar11.2024").
_FF_MONTH_ABBR = ["jan", "feb", "mar", "apr", "may", "jun",
                  "jul", "aug", "sep", "oct", "nov", "dec"]


def _split_pair(pair: str) -> List[str]:
    """EUR_USD -> ['EUR', 'USD']; raises ValueError if malformed."""
    if not isinstance(pair, str):
        raise ValueError(f"pair must be string, got {type(pair).__name__}")
    parts = [p.strip().upper() for p in pair.replace("/", "_").split("_") if p.strip()]
    if len(parts) != 2 or not all(len(p) == 3 for p in parts):
        raise ValueError(
            f"pair must be 'CCY_CCY' (e.g. 'EUR_USD'); got {pair!r}"
        )
    return parts


def _categorize(title: str, currency: str) -> str:
    """Heuristic title -> category mapping. Longer/more-specific patterns first.

    Returns one of _VALID_CATEGORIES. Defaults to OTHER for unmatched events.
    """
    t = (title or "").lower()
    cb = _CB_BY_CURRENCY.get(currency, "")

    # Central-bank policy events — match on policy-action keywords AND map by
    # currency, so "Monetary Policy Statement" with country=USD -> FOMC,
    # country=JPY -> BoJ, etc. This avoids per-bank string lists.
    cb_keywords = ("rate decision", "rate statement", "policy statement",
                   "monetary policy", "press conference", "minutes",
                   "interest rate", "main refinancing", "deposit facility",
                   "marginal lending", "official bank rate", "cash rate",
                   "overnight rate", "policy rate")
    if cb and any(k in t for k in cb_keywords):
        return cb

    # Explicit central-bank name mentions (covers cases where currency is
    # mis-tagged or the event title leads with the bank name).
    bank_aliases = {
        "fomc": "FOMC", "federal funds": "FOMC", "fed chair": "FOMC",
        "ecb": "ECB", "draghi": "ECB", "lagarde": "ECB",
        "boe": "BoE", "bank of england": "BoE", "mpc": "BoE",
        "boj": "BoJ", "bank of japan": "BoJ",
        "boc": "BoC", "bank of canada": "BoC",
        "rba": "RBA", "reserve bank of australia": "RBA",
        "rbnz": "RBNZ", "reserve bank of new zealand": "RBNZ",
        "snb": "SNB", "swiss national bank": "SNB",
    }
    for alias, cat in bank_aliases.items():
        if alias in t:
            return cat

    # Macro indicators
    if "non-farm" in t or "nonfarm" in t or "nfp" in t or "non farm" in t:
        return "NFP"
    if "cpi" in t or "consumer price" in t or "inflation" in t and "rate" not in t:
        # "inflation rate" caught here too; "interest rate" was caught above
        return "CPI"
    if "gdp" in t or "gross domestic" in t:
        return "GDP"

    return "OTHER"


def _impact_to_relevance(impact: str) -> float:
    """Map FF impact label to relevance_score; unknowns -> 0.33 (low)."""
    return _IMPACT_RELEVANCE.get((impact or "").strip().lower(), 0.33)


def _parse_ff_timestamp(date_str: str) -> Optional[datetime]:
    """Parse FF JSON ISO-8601 timestamp -> UTC datetime.

    FF emits timestamps like "2026-05-03T19:00:00-04:00" (NY local time +
    offset). We normalize to UTC. Returns None on parse failure (caller skips).
    """
    if not date_str:
        return None
    try:
        # Python 3.11+ fromisoformat handles "+HH:MM" offsets natively
        dt = datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        # FF should always include offset; treat naive as ET fallback would be
        # risky (DST ambiguity). Drop these events instead of guessing.
        return None
    return dt.astimezone(timezone.utc)


class ForexFactoryNewsSource(NewsSource):
    """ForexFactory economic-calendar source.

    Implementation strategy (Phase 2):
      - Live source: ``ff_calendar_thisweek.json`` mirror — public, no auth,
        JSON schema {title, country, date (ISO-8601 with TZ offset), impact,
        forecast, previous}. Returns the rolling current week (Sun-Sat ET).
      - Historical backfill: parquet cache at
        ``{cache_dir}/{pair}_ff_events.parquet``. Each fetch (a) reads cache
        for the requested window, (b) merges in any live thisweek events that
        overlap, (c) dedupes on (timestamp, currency, title), (d) writes back.
        Net effect: weekly cron of ``fetch_events(pair, last_7d, now)``
        accumulates a complete history without re-scraping.
      - Pair expansion: EUR_USD events = events tagged country=EUR OR USD.
      - Time semantics: ``since`` inclusive, ``until`` exclusive, both UTC.
      - Tolerates ``until > now()`` per NewsSource contract by clipping to now.

    FF-specific quirks (open questions in chat reply):
      - The free mirror only exposes the current week. Historical backfill
        before the cache existed is impossible without the paid HTML scrape.
      - "Holiday" / "Non-economic" rows get relevance_score=0 so they don't
        contaminate the embedding signal but are still cached for auditability.
      - Some events emit empty <impact> tags; we treat as "low" (0.33).
    """

    SOURCE_LABEL = "forex_factory"

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """Initialize the source.

        Args:
            cache_dir: Path to a parquet cache directory. Defaults to
                ``trained_data/news`` per the design doc. Created on demand.
        """
        if cache_dir is None:
            cache_dir = "trained_data/news"
        self.cache_dir = cache_dir
        self._cache_path = Path(cache_dir)
        # Lazy-built ``requests.Session`` for HTML scraping (warmed once,
        # then reused so Cloudflare cookies (``__cf_bm``) survive across
        # week-page requests). Initialized in ``_ensure_html_session``.
        self._html_sess = None
        # Throttle bookkeeping for HTML path. Wall-clock seconds since epoch
        # of the most-recent HTML fetch. Used to enforce the 2.5s polite gap.
        self._last_html_fetch_at: float = 0.0

    # -- Public contract ---------------------------------------------------

    def fetch_events(
        self,
        pair: str,
        since: datetime,
        until: datetime,
    ) -> List[NewsEvent]:
        """See ``NewsSource.fetch_events``.

        Auto-routes between two implementations:
          - **JSON path (fast)**: window fully inside the rolling current
            week (Sun-Sat ET, mirrored to ``ff_calendar_thisweek.json``).
            Single HTTP call, no rate-limit concerns.
          - **HTML path (historical)**: window touches any prior week.
            Walks weekly URLs from FF's calendar archive, parses event
            JSON embedded in each page, dedupe-merges into the same parquet
            cache. Polite 2.5s sleep between fetches.

        Either path returns ``List[NewsEvent]`` with identical schema.

        Behavior (JSON path):
          1. Validate args (UTC tz-aware, since < until, pair format).
          2. Load any existing parquet cache rows in [since, until).
          3. Fetch live FF thisweek JSON; merge any rows in [since, until).
          4. Dedupe on (timestamp, currency, title); write back to parquet.
          5. Return ascending-sorted ``NewsEvent`` list filtered to currencies
             of ``pair``.

        Behavior (HTML path): see ``fetch_events_historical``.
        """
        currencies = _split_pair(pair)  # raises ValueError on bad pair
        since_utc = self._coerce_utc(since, "since")
        until_utc = self._coerce_utc(until, "until")
        if since_utc >= until_utc:
            raise ValueError(
                f"since ({since_utc}) must be < until ({until_utc})"
            )

        # Auto-route. The JSON mirror covers a Sun-Sat ET week; if either
        # boundary falls before the start of the current ET week, we MUST
        # walk the HTML archive. Otherwise the JSON path is sufficient.
        if self._needs_historical(since_utc, until_utc):
            logger.info(
                "ForexFactory: window crosses prior weeks; routing to HTML "
                "historical path (pair=%s, since=%s, until=%s)",
                pair, since_utc.isoformat(), until_utc.isoformat(),
            )
            return self.fetch_events_historical(pair, since_utc, until_utc)

        # Clip until to now() per NewsSource contract — backfill cannot return
        # events from the future regardless of caller's window.
        now = datetime.now(timezone.utc)
        effective_until = min(until_utc, now)

        # 1. Read cache. Empty if first run.
        cache_rows = self._read_cache(pair)

        # 2. Fetch live thisweek JSON. Soft-fail if offline AND cache exists;
        # hard-fail if neither cache nor live can produce data.
        try:
            live_rows = self._fetch_live_json()
        except RuntimeError:
            if not cache_rows:
                # No fallback available — surface to caller per contract.
                raise
            logger.warning(
                "ForexFactory live fetch failed; serving from cache only "
                "(cache_rows=%d, pair=%s)", len(cache_rows), pair
            )
            live_rows = []

        # 3. Merge + dedupe on (timestamp_iso, currency, title). Cache wins
        # ties on the (rare) chance FF revises a past event title.
        merged = self._merge_dedupe(cache_rows, live_rows)

        # 4. Persist updated cache (only if we got live data — don't no-op
        # rewrite the same parquet on a pure cache read).
        if live_rows:
            self._write_cache(pair, merged)

        # 5. Filter to window + relevant currencies + build NewsEvent list.
        return self._rows_to_events(
            merged, pair, currencies, since_utc, effective_until
        )

    # -- HTML historical path (Stage 4-A.ii) -------------------------------

    def fetch_events_historical(
        self,
        pair: str,
        since: datetime,
        until: datetime,
    ) -> List[NewsEvent]:
        """Fetch events for arbitrary past windows by scraping FF's HTML
        calendar archive.

        FF organizes the calendar by week with URLs of the form
        ``/calendar?week=mmmDD.YYYY`` (lowercase 3-letter month + day-of-month
        + year, no separators between month-and-day; verified 2026-05-08 with
        ``mar11.2024``). The page accepts ANY date in the displayed week and
        returns 7 consecutive days starting from that date — to avoid
        overlap and gaps we anchor each fetch to the Monday of the ISO week
        containing each requested day.

        Caches dedupe-merge into ``{cache_dir}/{pair}_ff_events.parquet`` —
        the same parquet the JSON path uses, so a forward-cron + a one-time
        backfill compose cleanly into a single artifact.

        Args:
            pair: Currency pair, e.g. ``"EUR_USD"``.
            since: UTC tz-aware lower bound on event timestamps (inclusive).
            until: UTC tz-aware upper bound on event timestamps (exclusive).

        Returns:
            Ascending-sorted ``List[NewsEvent]`` filtered to currencies of
            ``pair``.

        Raises:
            ValueError: malformed pair or naive timestamps.
            RuntimeError: irrecoverable transport failure AND empty cache
                (per NewsSource contract; partial cache is served if any
                weeks already on disk).
        """
        currencies = _split_pair(pair)
        since_utc = self._coerce_utc(since, "since")
        until_utc = self._coerce_utc(until, "until")
        if since_utc >= until_utc:
            raise ValueError(
                f"since ({since_utc}) must be < until ({until_utc})"
            )
        now = datetime.now(timezone.utc)
        effective_until = min(until_utc, now)

        # Existing cache rows — used for idempotency (skip already-cached
        # weeks) AND fallback (serve cache if all live fetches fail).
        cache_rows = self._read_cache(pair)
        cached_week_keys = self._weeks_in_rows(cache_rows)

        weeks_needed = self._iter_weeks(since_utc, effective_until)
        new_rows: List[dict] = []
        weeks_attempted = 0
        weeks_fetched = 0
        weeks_skipped_cached = 0
        weeks_failed = 0
        for monday in weeks_needed:
            weeks_attempted += 1
            week_key = monday.strftime("%Y-%m-%d")
            if week_key in cached_week_keys:
                # Idempotent: skip weeks already represented in cache. We
                # consider a week "covered" if ANY event exists for that
                # ISO-Monday in cache. FF event volume is consistent enough
                # that a week with zero events is impossible in practice
                # (50-100 events/week typical).
                weeks_skipped_cached += 1
                continue
            try:
                week_rows = self._fetch_week_html(monday)
            except RuntimeError as e:
                weeks_failed += 1
                logger.warning(
                    "FF HTML fetch failed for week %s: %s", week_key, e
                )
                continue
            new_rows.extend(week_rows)
            weeks_fetched += 1
            # Progress log every 4 weeks (~10s of polite traffic).
            if weeks_fetched % 4 == 0:
                logger.info(
                    "FF HTML backfill progress: %d weeks fetched "
                    "(%d skipped cached, %d failed) for %s",
                    weeks_fetched, weeks_skipped_cached, weeks_failed, pair,
                )

        # If everything failed AND cache is empty, surface a hard error per
        # NewsSource contract. Otherwise serve whatever combination of cached
        # rows + freshly-fetched rows we have.
        if not cache_rows and not new_rows:
            raise RuntimeError(
                f"FF HTML backfill: 0 weeks fetched for pair={pair}, window="
                f"{since_utc.isoformat()}..{effective_until.isoformat()}; "
                f"weeks_attempted={weeks_attempted}, failed={weeks_failed}. "
                "All fetches failed and cache is empty."
            )

        merged = self._merge_dedupe(cache_rows, new_rows)
        if new_rows:
            # Only rewrite parquet when we actually added rows. Idempotent
            # re-runs (all weeks cached) are a no-op on disk.
            self._write_cache(pair, merged)

        logger.info(
            "FF HTML backfill complete: pair=%s weeks_attempted=%d "
            "fetched=%d skipped_cached=%d failed=%d new_rows=%d total=%d",
            pair, weeks_attempted, weeks_fetched, weeks_skipped_cached,
            weeks_failed, len(new_rows), len(merged),
        )
        return self._rows_to_events(
            merged, pair, currencies, since_utc, effective_until
        )

    # -- Row -> NewsEvent shared helper ------------------------------------

    def _rows_to_events(
        self,
        rows: List[dict],
        pair: str,
        currencies: List[str],
        since_utc: datetime,
        effective_until: datetime,
    ) -> List[NewsEvent]:
        """Filter cached rows to (window, currencies) + materialize
        ``NewsEvent`` objects. Skips and logs rows that fail invariants
        rather than crashing the backfill."""
        events: List[NewsEvent] = []
        for row in rows:
            ts: datetime = row["timestamp"]
            if ts < since_utc or ts >= effective_until:
                continue
            if row["currency"] not in currencies:
                continue
            try:
                events.append(NewsEvent(
                    timestamp=ts,
                    text=row["title"],
                    source=self.SOURCE_LABEL,
                    category=row["category"],
                    relevance_score=float(row["relevance_score"]),
                    pair=pair,
                    impact=row.get("impact"),
                ))
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Skipping malformed FF cache row: %s; row=%s", e, row
                )
        events.sort(key=lambda e: e.timestamp)
        return events

    # -- Internals ---------------------------------------------------------

    @staticmethod
    def _coerce_utc(dt: datetime, label: str) -> datetime:
        if not isinstance(dt, datetime):
            raise ValueError(f"{label} must be datetime, got {type(dt).__name__}")
        if dt.tzinfo is None:
            raise ValueError(
                f"{label} must be timezone-aware; got naive {dt!r}. "
                "Pass datetime(..., tzinfo=timezone.utc)."
            )
        return dt.astimezone(timezone.utc)

    def _fetch_live_json(self) -> List[dict]:
        """Pull the FF thisweek JSON and parse to list of internal-row dicts.

        Internal row schema:
            timestamp: tz-aware UTC datetime
            currency: 3-char uppercase
            title: cleaned headline string
            impact: lowercased FF impact label
            relevance_score: float in [0,1]
            category: one of _VALID_CATEGORIES

        Raises:
            RuntimeError on transport failure (per NewsSource contract).
        """
        # Lazy import — keeps `requests` out of import-time hot paths.
        try:
            import requests
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(f"requests not available: {e}") from e

        try:
            resp = requests.get(_FF_JSON_URL, headers=_FF_HEADERS, timeout=10)
        except requests.RequestException as e:
            raise RuntimeError(f"ForexFactory transport error: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"ForexFactory HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            payload = resp.json()
        except ValueError as e:
            raise RuntimeError(f"ForexFactory invalid JSON: {e}") from e

        if not isinstance(payload, list):
            raise RuntimeError(
                f"ForexFactory expected JSON array, got {type(payload).__name__}"
            )

        rows: List[dict] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            ts = _parse_ff_timestamp(item.get("date", ""))
            if ts is None:
                continue
            currency = (item.get("country") or "").strip().upper()
            if len(currency) != 3:
                continue
            title = (item.get("title") or "").strip()
            if not title:
                continue
            impact = (item.get("impact") or "").strip().lower()
            rows.append({
                "timestamp": ts,
                "currency": currency,
                "title": title,
                "impact": impact,
                "relevance_score": _impact_to_relevance(impact),
                "category": _categorize(title, currency),
            })
        return rows

    def _cache_file(self, pair: str) -> Path:
        return self._cache_path / f"{pair}_ff_events.parquet"

    def _read_cache(self, pair: str) -> List[dict]:
        """Read parquet cache; return [] if missing or unreadable."""
        path = self._cache_file(pair)
        if not path.exists():
            return []
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not available; skipping FF cache read")
            return []
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            # Per .claude/rules/improvement.md JSON Safety Gates pattern —
            # graceful fallback on corrupt cache, never crash the caller.
            logger.warning(
                "FF cache unreadable at %s: %s; treating as empty", path, e
            )
            return []
        if df.empty:
            return []
        rows: List[dict] = []
        for _, r in df.iterrows():
            ts = r["timestamp"]
            # Parquet round-trips pandas Timestamp; coerce back to tz-aware.
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            rows.append({
                "timestamp": ts,
                "currency": str(r["currency"]),
                "title": str(r["title"]),
                "impact": str(r.get("impact", "") or ""),
                "relevance_score": float(r["relevance_score"]),
                "category": str(r["category"]),
            })
        return rows

    def _write_cache(self, pair: str, rows: Iterable[dict]) -> None:
        """Atomically write merged rows to parquet (tmp + rename)."""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not available; skipping FF cache write")
            return

        rows_list = list(rows)
        if not rows_list:
            return

        self._cache_path.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows_list)
        # Keep tz info round-tripped via pandas Timestamp w/ UTC.
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        path = self._cache_file(pair)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            df.to_parquet(tmp_path, index=False)
            tmp_path.replace(path)  # atomic on POSIX
        except Exception as e:
            logger.warning("FF cache write failed at %s: %s", path, e)
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

    @staticmethod
    def _merge_dedupe(
        cache_rows: List[dict],
        live_rows: List[dict],
    ) -> List[dict]:
        """Dedupe on (timestamp_iso_min, currency, title). Cache wins ties."""
        out: dict = {}
        # Live first, then cache overwrites — preserves any operator-edited
        # cache entries against re-scraped FF rows.
        for row in live_rows:
            key = (
                row["timestamp"].astimezone(timezone.utc).replace(second=0, microsecond=0).isoformat(),
                row["currency"],
                row["title"],
            )
            out[key] = row
        for row in cache_rows:
            key = (
                row["timestamp"].astimezone(timezone.utc).replace(second=0, microsecond=0).isoformat(),
                row["currency"],
                row["title"],
            )
            out[key] = row
        merged = list(out.values())
        merged.sort(key=lambda r: r["timestamp"])
        return merged

    # -- HTML scraper internals --------------------------------------------

    @staticmethod
    def _needs_historical(since_utc: datetime, until_utc: datetime) -> bool:
        """Return True if the window touches any week before the current ET
        Sun-Sat. The JSON mirror only covers the rolling current week.

        Heuristic: if ``since_utc`` is older than 6 days ago we route to HTML
        unconditionally. (FF's calendar week resets Sun 00:00 ET; computing
        the exact ET week boundary for a UTC `since` requires a tz database
        and adds little — 6-day-floor is safe and conservative.)
        """
        now = datetime.now(timezone.utc)
        return since_utc < now - timedelta(days=6)

    @staticmethod
    def _iter_weeks(since_utc: datetime, until_utc: datetime) -> List[datetime]:
        """Yield Monday 00:00 UTC anchors covering [since_utc, until_utc].

        FF's ``?week=mmmDD.YYYY`` URL accepts any date and returns 7 days
        starting from that date. To get non-overlapping coverage we anchor
        each fetch to the Monday of the ISO week containing ``since_utc``,
        then step 7 days at a time until past ``until_utc``.
        """
        # ISO weekday: Mon=0 .. Sun=6 (datetime.weekday())
        first_day = since_utc.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        days_back = first_day.weekday()
        monday = first_day - timedelta(days=days_back)
        weeks: List[datetime] = []
        cursor = monday
        # Iterate while any part of the week could overlap [since, until).
        # The week starting at `cursor` covers [cursor, cursor + 7d).
        while cursor < until_utc:
            weeks.append(cursor)
            cursor = cursor + timedelta(days=7)
        return weeks

    @staticmethod
    def _format_ff_week_param(monday: datetime) -> str:
        """Monday datetime -> FF URL fragment (e.g. ``mar11.2024``)."""
        m = _FF_MONTH_ABBR[monday.month - 1]
        return f"{m}{monday.day}.{monday.year}"

    @staticmethod
    def _weeks_in_rows(rows: List[dict]) -> set:
        """Return {YYYY-MM-DD Monday} for every event timestamp in ``rows``.

        Used to skip already-cached weeks during backfill (idempotency).
        """
        weeks: set = set()
        for r in rows:
            ts = r.get("timestamp")
            if ts is None:
                continue
            ts_utc = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            monday = ts_utc - timedelta(days=ts_utc.weekday())
            weeks.add(monday.strftime("%Y-%m-%d"))
        return weeks

    def _ensure_html_session(self):
        """Build (lazily) a ``requests.Session`` warmed against the FF
        homepage so Cloudflare's ``__cf_bm`` cookie is set. Reused across
        weekly fetches so we don't re-warm per request (saves rate-limit
        budget and avoids redundant network hops).

        Raises:
            RuntimeError on transport failure of the warm-up request.
        """
        if self._html_sess is not None:
            return self._html_sess
        try:
            import requests
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(f"requests not available: {e}") from e
        sess = requests.Session()
        sess.headers.update(_FF_HTML_HEADERS)
        try:
            r = sess.get(_FF_HTML_BASE + "/", timeout=15)
        except requests.RequestException as e:
            raise RuntimeError(f"FF HTML warm-up transport error: {e}") from e
        if r.status_code != 200:
            raise RuntimeError(
                f"FF HTML warm-up HTTP {r.status_code}: {r.text[:200]}"
            )
        # Throttle bookkeeping starts here.
        self._last_html_fetch_at = time.monotonic()
        self._html_sess = sess
        return sess

    def _polite_sleep(self) -> None:
        """Sleep just enough to keep ``_FF_HTML_MIN_INTERVAL_SEC`` between
        consecutive HTML fetches. Uses ``time.monotonic`` so it is robust
        to wall-clock jumps."""
        elapsed = time.monotonic() - self._last_html_fetch_at
        wait = _FF_HTML_MIN_INTERVAL_SEC - elapsed
        if wait > 0:
            time.sleep(wait)

    def _fetch_week_html(self, monday: datetime) -> List[dict]:
        """Fetch one week of FF calendar HTML and parse to internal-row dicts.

        Args:
            monday: Monday 00:00 UTC anchor for the week to fetch.

        Returns:
            List of internal-row dicts (same schema as ``_fetch_live_json``).
            Empty list if the page parses cleanly but contains zero events
            (rare; usually means the page format changed).

        Raises:
            RuntimeError on transport failure (HTTP non-200, JSON-blob not
            found, decode error). The caller decides whether to retry, fall
            back, or surface to the user.
        """
        try:
            import requests
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(f"requests not available: {e}") from e

        sess = self._ensure_html_session()
        self._polite_sleep()

        week_param = self._format_ff_week_param(monday)
        url = f"{_FF_HTML_BASE}/calendar?week={week_param}"
        try:
            resp = sess.get(url, timeout=20)
        except requests.RequestException as e:
            self._last_html_fetch_at = time.monotonic()
            raise RuntimeError(f"FF HTML transport error: {e}") from e
        finally:
            self._last_html_fetch_at = time.monotonic()

        if resp.status_code == 429:
            raise RuntimeError(
                f"FF HTML 429 rate-limit on {url}; back off and retry later"
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"FF HTML HTTP {resp.status_code} on {url}: "
                f"{resp.text[:200]}"
            )
        return self._parse_ff_html_events(resp.text)

    @staticmethod
    def _parse_ff_html_events(html: str) -> List[dict]:
        """Parse internal-row dicts from a FF calendar HTML page.

        FF embeds the events as a JS-object literal under
        ``window.calendarComponentStates[1] = { days: [{events: [...]}, ...] }``.
        The outer object uses unquoted JS keys (so ``json.loads`` chokes on
        it) but each event object inside ``events`` is well-formed JSON.
        We bracket-match each ``{"id":<digits>,...}`` event and parse it
        individually. Robust to FF's occasional whitespace tweaks.

        Returns:
            List of internal-row dicts (timestamp, currency, title, impact,
            relevance_score, category). Empty if the JS blob is missing
            (e.g., page format change) — caller surfaces this as a fetch
            failure if the empty result is unexpected.
        """
        # 1. Locate the JS blob.
        m = re.search(
            r"window\.calendarComponentStates\[1\]\s*=\s*(\{.*?\});\s*</script>",
            html,
            re.DOTALL,
        )
        if not m:
            # Page format may have changed, or page returned an error/login
            # screen with no calendar payload.
            logger.warning(
                "FF HTML parse: calendarComponentStates[1] blob not found "
                "(html_len=%d); FF page format may have changed", len(html)
            )
            return []
        blob = m.group(1)

        # 2. Extract each event JSON object via bracket-counting from the
        # ``{"id":<digits>`` anchor. We avoid a full JS parser because
        # ``demjson3`` etc. add a heavy optional dep for one parsing job.
        events_raw: List[str] = []
        for anchor in re.finditer(r'\{"id":\d+,', blob):
            start = anchor.start()
            depth = 0
            in_str = False
            esc = False
            end = -1
            for i in range(start, len(blob)):
                c = blob[i]
                if esc:
                    esc = False
                    continue
                if c == "\\":
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > 0:
                events_raw.append(blob[start:end])

        rows: List[dict] = []
        for raw in events_raw:
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.debug(
                    "FF HTML parse: event JSON decode failed: %s; raw=%r",
                    e, raw[:120]
                )
                continue

            # Schema fields per empirically-verified FF HTML payload
            # (2026-05-08, week=mar11.2024):
            #   dateline: int epoch seconds (UTC)
            #   currency: 3-char ISO ("USD", "EUR", "JPY", ...)
            #   name: event title (post-prefix; "prefixedName" has CCY prefix)
            #   impactName: "low" | "medium" | "high" | "holiday" | ""
            ts_epoch = ev.get("dateline")
            if not isinstance(ts_epoch, (int, float)) or ts_epoch <= 0:
                continue
            try:
                ts = datetime.fromtimestamp(int(ts_epoch), tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                continue

            currency = (ev.get("currency") or "").strip().upper()
            if len(currency) != 3:
                continue

            title = (ev.get("name") or "").strip()
            if not title:
                # Some FF "all-day" rows (e.g., bank holidays) have no name;
                # fall back to the prefixed name.
                title = (ev.get("prefixedName") or "").strip()
            if not title:
                continue

            impact = (ev.get("impactName") or "").strip().lower()
            rows.append({
                "timestamp": ts,
                "currency": currency,
                "title": title,
                "impact": impact,
                "relevance_score": _impact_to_relevance(impact),
                "category": _categorize(title, currency),
            })
        return rows
