"""Lightweight text-derived features for market models.

This module is dependency-free by design (no NLTK/TextBlob).
It provides a small, deterministic sentiment scorer suitable for
turning news/headlines/notes into numeric features.
"""

from __future__ import annotations

import re
import math
import hashlib
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
    sentiment_std: float
    sentiment_min: float
    sentiment_max: float
    sentiment_abs_mean: float
    sentiment_nonzero_frac: float
    token_count: int
    char_count: int


def _stable_hash_to_bucket(text: str, dim: int) -> int:
    digest = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return int(value % dim)


def _hashed_ngrams(
    tokens: list[str],
    dim: int,
    min_n: int,
    max_n: int,
) -> list[float]:
    if dim <= 0:
        raise ValueError("dim must be > 0")
    if min_n <= 0 or max_n <= 0 or min_n > max_n:
        raise ValueError("invalid ngram range")

    vec = [0.0] * dim
    token_count = len(tokens)
    if token_count == 0:
        return vec

    total = 0
    for n in range(min_n, max_n + 1):
        if token_count < n:
            continue
        for i in range(0, token_count - n + 1):
            gram = "_".join(tokens[i: i + n])
            bucket = _stable_hash_to_bucket(gram, dim)
            vec[bucket] += 1.0
            total += 1

    if total <= 0:
        return vec

    inv_total = 1.0 / float(total)
    return [v * inv_total for v in vec]


def _mean(values: list[float]) -> float:
    return float(sum(values)) / float(len(values)) if values else 0.0


def _std(values: list[float], mean: float) -> float:
    if not values:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / float(len(values))
    return float(math.sqrt(variance))


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
        return TextFeatureResult(
            sentiment=0.0,
            sentiment_std=0.0,
            sentiment_min=0.0,
            sentiment_max=0.0,
            sentiment_abs_mean=0.0,
            sentiment_nonzero_frac=0.0,
            token_count=0,
            char_count=0,
        )

    sentiments = [simple_sentiment_score(t) for t in texts_list]
    joined = " ".join([t for t in texts_list if t])
    tokens = _tokenize(joined)

    # Aggregate sentiment distribution (already bounded in [-1, 1])
    sentiment = _mean(sentiments)
    sentiment_min = float(min(sentiments)) if sentiments else 0.0
    sentiment_max = float(max(sentiments)) if sentiments else 0.0
    sentiment_std = _std(sentiments, sentiment)
    sentiment_abs_mean = _mean([abs(s) for s in sentiments])
    sentiment_nonzero = sum(1 for s in sentiments if abs(s) > 1e-12)
    sentiment_nonzero_frac = (
        float(sentiment_nonzero) / float(len(sentiments))
        if sentiments
        else 0.0
    )

    return TextFeatureResult(
        sentiment=float(sentiment),
        sentiment_std=float(sentiment_std),
        sentiment_min=float(sentiment_min),
        sentiment_max=float(sentiment_max),
        sentiment_abs_mean=float(sentiment_abs_mean),
        sentiment_nonzero_frac=float(sentiment_nonzero_frac),
        token_count=int(len(tokens)),
        char_count=int(len(joined)),
    )


def hashed_ngram_features(
    texts: Iterable[str],
    dim: int = 64,
    min_n: int = 1,
    max_n: int = 2,
) -> list[float]:
    """Return a normalized hashed n-gram vector (bag-of-ngrams).

    Dependency-free alternative to TF-IDF. Output is length `dim` with
    normalized counts in [0, 1].
    """
    joined = " ".join([t for t in texts if t])
    tokens = _tokenize(joined)
    return _hashed_ngrams(tokens=tokens, dim=dim, min_n=min_n, max_n=max_n)
# — Raynergy-svg —
