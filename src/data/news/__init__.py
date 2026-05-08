"""News/macro signal pipeline (Phase 1 scaffolding).

This package fuses event/headline data with price features for FX direction
prediction. It is the P1 lever in the May 2026 modernization roadmap (see
``docs/superpowers/plans/2026-05-08-news-macro-signal-design.md``).

Phase status (2026-05-08):
    - Phase 1 (this commit): design + stubs only. NO data fetching, NO embedding,
      NO trainer integration.
    - Phase 2 (next session): implement ``ForexFactoryNewsSource.fetch_events``
      and ``FinBERTEmbedder.embed``; produce sample artifact for EUR_USD.
    - Phase 3: implement ``align_news_to_bars``; thread into
      ``compute_normalized_features``; retrain M15 EUR_USD with news features.
    - Phase 4: expand to remaining majors if Phase 3 holdout lift >= 3pp.

Public surface (intentionally minimal until Phase 2):
    NewsEvent           — dataclass: timestamp, text, source, category,
                          relevance_score, optional pair, optional impact.
    NewsSource          — abstract base; ``fetch_events(pair, since, until)``.
    NewsEmbedder        — abstract base; ``embed(events)``.
    ForexFactoryNewsSource — stub; raises NotImplementedError until Phase 2.
    FinBERTEmbedder     — stub; raises NotImplementedError until Phase 2.
    align_news_to_bars  — stub; raises NotImplementedError until Phase 3.

Lazy imports keep this package from pulling heavy ML deps (``transformers``,
``torch``) until something actually instantiates a concrete class. Aligns with
project rule: "ALWAYS use lazy imports in package __init__.py when submodules
have heavy dependencies" (.claude/rules/improvement.md).
"""

from src.data.news.source import (
    NewsEvent,
    NewsSource,
    ForexFactoryNewsSource,
)
from src.data.news.embedder import (
    NewsEmbedder,
    FinBERTEmbedder,
)
from src.data.news.feature_alignment import align_news_to_bars

__all__ = [
    "NewsEvent",
    "NewsSource",
    "ForexFactoryNewsSource",
    "NewsEmbedder",
    "FinBERTEmbedder",
    "align_news_to_bars",
]
