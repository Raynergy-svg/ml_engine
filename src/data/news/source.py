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
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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

    # -- Public contract ---------------------------------------------------

    def fetch_events(
        self,
        pair: str,
        since: datetime,
        until: datetime,
    ) -> List[NewsEvent]:
        """See ``NewsSource.fetch_events``.

        Behavior:
          1. Validate args (UTC tz-aware, since < until, pair format).
          2. Load any existing parquet cache rows in [since, until).
          3. Fetch live FF thisweek JSON; merge any rows in [since, until).
          4. Dedupe on (timestamp, currency, title); write back to parquet.
          5. Return ascending-sorted ``NewsEvent`` list filtered to currencies
             of ``pair``.
        """
        currencies = _split_pair(pair)  # raises ValueError on bad pair
        since_utc = self._coerce_utc(since, "since")
        until_utc = self._coerce_utc(until, "until")
        if since_utc >= until_utc:
            raise ValueError(
                f"since ({since_utc}) must be < until ({until_utc})"
            )

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
        events: List[NewsEvent] = []
        for row in merged:
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
                # Bad row in cache — log and skip rather than crash backfill.
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
