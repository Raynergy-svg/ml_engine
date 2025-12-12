"""Lightweight text-derived features for market models.

This module is dependency-free by design (no NLTK/TextBlob).
It provides a small, deterministic sentiment scorer suitable for
turning news/headlines/notes into numeric features.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


# Small lexicon tuned for finance-ish headlines.
_POSITIVE = {
    "beat",
    "beats",
    "bull",
    "bullish",
    "boom",
    "growth",
    "up",
    "upgrade",
    "upgrades",
    "outperform",
    "strong",
    "stronger",
    "record",
    "surge",
    "surges",
    "rally",
    "rallies",
    "profit",
    "profits",
    "profitable",
    "optimistic",
    "raise",
    "raises",
    "raised",
    "guidance",
    "buy",
}

_NEGATIVE = {
    "bear",
    "bearish",
    "miss",
    "misses",
    "down",
    "downgrade",
    "downgrades",
    "underperform",
    "weak",
    "weaker",
    "drop",
    "drops",
    "plunge",
    "plunges",
    "sell",
    "loss",
    "losses",
    "lawsuit",
    "probe",
    "investigation",
    "fraud",
    "cut",
    "cuts",
    "cutting",
    "lower",
    "warning",
    "warns",
    "recession",
}

_NEGATIONS = {"not", "no", "never", "without"}


@dataclass(frozen=True)
class TextFeatureResult:
    sentiment: float
    token_count: int
    char_count: int


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _token_polarity(token: str) -> int:
    if token in _POSITIVE:
        return 1
    if token in _NEGATIVE:
        return -1
    return 0


def simple_sentiment_score(text: Optional[str]) -> float:
    """Return a small sentiment score in [-1, 1].

    This is intentionally simple/cheap:
    - tokenizes alphabetic words
    - uses a tiny pos/neg lexicon
    - handles negation by flipping the next token's polarity

    If text is empty/None, returns 0.
    """
    if not text:
        return 0.0

    tokens = _tokenize(text)
    if not tokens:
        return 0.0

    score = 0
    polarity_hits = 0
    negate_next = False

    for tok in tokens:
        if tok in _NEGATIONS:
            negate_next = True
            continue

        polarity = _token_polarity(tok)
        if polarity:
            score += (-polarity if negate_next else polarity)
            polarity_hits += 1

        negate_next = False

    if polarity_hits == 0:
        return 0.0

    # Normalize to [-1, 1]
    score = float(score) / float(polarity_hits)
    if score > 1.0:
        return 1.0
    if score < -1.0:
        return -1.0
    return float(score)


def text_feature_summary(texts: Iterable[str]) -> TextFeatureResult:
    """Aggregate a collection of texts into numeric features."""
    texts_list = list(texts)
    if not texts_list:
        return TextFeatureResult(sentiment=0.0, token_count=0, char_count=0)

    sentiments = [simple_sentiment_score(t) for t in texts_list]
    joined = " ".join([t for t in texts_list if t])
    tokens = _tokenize(joined)

    # Mean sentiment across rows (already bounded)
    sentiment = sum(sentiments) / float(len(sentiments)) if sentiments else 0.0

    return TextFeatureResult(
        sentiment=float(sentiment),
        token_count=int(len(tokens)),
        char_count=int(len(joined)),
    )
