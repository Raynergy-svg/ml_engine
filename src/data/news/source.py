"""News data sources (Phase 1 stubs).

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

Phase 1: this module ships the abstract base + dataclass + a Phase-2 stub for
ForexFactory. Calling ``fetch_events`` raises ``NotImplementedError`` so any
caller that drifts off the phase plan fails loudly rather than silently
producing empty data.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


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


class ForexFactoryNewsSource(NewsSource):
    """ForexFactory economic-calendar source (Phase 2 implementation).

    Design (Phase 2 work):
        - Reuse ``market_intelligence.EconomicCalendar`` for the runtime path
          (already wired in ``src/scanner/agents/_team.py:_evaluate_news_risk``).
        - Add a historical backfill method that scrapes the FF calendar
          archive page-per-week for arbitrary date ranges.
        - Cache to ``trained_data/news/{pair}_ff_events.parquet`` to make
          backfill idempotent across retrains.
        - Map FF impact to ``relevance_score``: low=0.33, medium=0.66, high=1.0.

    Phase 1: stub only. ``fetch_events`` raises NotImplementedError so any
    accidental Phase-3 wiring fails loudly.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """Initialize the source.

        Args:
            cache_dir: Optional path to a parquet cache directory. Phase 2 will
                use this for backfill caching. Phase 1: stored but unused.
        """
        self.cache_dir = cache_dir
        # Phase 1: instance can be constructed (so tests can verify the type)
        # but cannot fetch.

    def fetch_events(
        self,
        pair: str,
        since: datetime,
        until: datetime,
    ) -> List[NewsEvent]:
        """Phase 2 implementation pending."""
        raise NotImplementedError(
            "ForexFactoryNewsSource.fetch_events is a Phase 2 implementation. "
            "See docs/superpowers/plans/2026-05-08-news-macro-signal-design.md "
            "§6 (Sequencing) for phase boundaries."
        )
