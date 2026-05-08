"""Tests for the Phase 1 news/macro pipeline scaffolding.

Verifies:
  - The package imports cleanly (no import-time side effects).
  - The abstract base classes cannot be instantiated directly.
  - The concrete stubs CAN be instantiated (so Phase 2 has a target to fill).
  - The stub methods raise REAL ``NotImplementedError`` (not mocked, not
    silently skipped) — drift off the phase plan must fail loudly.
  - The ``NewsEvent`` dataclass enforces its construction-time invariants
    (timezone-aware timestamps, valid relevance_score range, non-empty text).

NO MOCKS. Per .claude/rules/improvement.md "No-Mock Rule" (promoted 2026-05-01):
real classes, real disk, real exceptions. The whole point of this test file is
to verify Phase 1 ships exception-raising stubs — mocking the exception would
defeat the test.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------


def test_package_imports() -> None:
    """The package can be imported and re-exports its public surface."""
    import src.data.news as news_pkg

    expected_names = {
        "NewsEvent",
        "NewsSource",
        "ForexFactoryNewsSource",
        "NewsEmbedder",
        "FinBERTEmbedder",
        "align_news_to_bars",
    }
    assert expected_names.issubset(set(news_pkg.__all__))
    for name in expected_names:
        assert hasattr(news_pkg, name), f"Missing public export: {name}"


# ---------------------------------------------------------------------------
# NewsEvent dataclass
# ---------------------------------------------------------------------------


def _make_event(**overrides) -> "object":
    from src.data.news import NewsEvent

    defaults = dict(
        timestamp=datetime(2026, 5, 8, 12, 30, tzinfo=timezone.utc),
        text="Non-Farm Payrolls released at 270k vs 200k expected.",
        source="forex_factory",
        category="NFP",
        relevance_score=1.0,
        pair="EUR_USD",
        impact="high",
    )
    defaults.update(overrides)
    return NewsEvent(**defaults)


def test_news_event_constructs_valid() -> None:
    """A well-formed NewsEvent constructs cleanly."""
    ev = _make_event()
    assert ev.text.startswith("Non-Farm")
    assert ev.relevance_score == 1.0
    assert ev.pair == "EUR_USD"
    assert ev.timestamp.tzinfo is not None


def test_news_event_rejects_naive_timestamp() -> None:
    """Naive (non-tz-aware) timestamps must be rejected — would leak lookahead bias."""
    naive_ts = datetime(2026, 5, 8, 12, 30)  # no tzinfo
    with pytest.raises(ValueError, match="timezone-aware"):
        _make_event(timestamp=naive_ts)


def test_news_event_rejects_non_datetime_timestamp() -> None:
    """Non-datetime timestamp surfaces a clear TypeError."""
    with pytest.raises(TypeError, match="datetime"):
        _make_event(timestamp="2026-05-08T12:30:00Z")


def test_news_event_rejects_empty_text() -> None:
    """Empty/whitespace-only text would feed garbage to the embedder."""
    with pytest.raises(ValueError, match="non-empty"):
        _make_event(text="   ")


def test_news_event_rejects_out_of_range_relevance() -> None:
    """relevance_score must live in [0, 1] for downstream weight math."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _make_event(relevance_score=1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _make_event(relevance_score=-0.1)


# ---------------------------------------------------------------------------
# NewsSource abstract + ForexFactory stub
# ---------------------------------------------------------------------------


def test_news_source_abc_cannot_instantiate() -> None:
    """The abstract base class refuses instantiation."""
    from src.data.news import NewsSource

    with pytest.raises(TypeError, match="abstract"):
        NewsSource()


def test_forex_factory_source_constructs() -> None:
    """The concrete stub can be instantiated (so Phase 2 has a target)."""
    from src.data.news import ForexFactoryNewsSource

    src = ForexFactoryNewsSource()
    assert src.cache_dir is None

    src_with_cache = ForexFactoryNewsSource(cache_dir="/tmp/news_cache")
    assert src_with_cache.cache_dir == "/tmp/news_cache"


def test_forex_factory_fetch_raises_not_implemented() -> None:
    """fetch_events must raise NotImplementedError, not silently return [].

    Silent stubs were the failure mode behind the meta-pipeline _adjuster=None
    incident (see .claude/rules/improvement.md No-Mock Rule).
    """
    from src.data.news import ForexFactoryNewsSource

    src = ForexFactoryNewsSource()
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 8, tzinfo=timezone.utc)
    with pytest.raises(NotImplementedError, match="Phase 2"):
        src.fetch_events("EUR_USD", since, until)


# ---------------------------------------------------------------------------
# NewsEmbedder abstract + FinBERT stub
# ---------------------------------------------------------------------------


def test_news_embedder_abc_cannot_instantiate() -> None:
    """The abstract base class refuses instantiation."""
    from src.data.news import NewsEmbedder

    with pytest.raises(TypeError, match="abstract"):
        NewsEmbedder()


def test_finbert_embedder_constructs_with_defaults() -> None:
    """Default-constructed FinBERTEmbedder exposes the contracted dim."""
    from src.data.news import FinBERTEmbedder

    emb = FinBERTEmbedder()
    assert emb.model_name == "ProsusAI/finbert"
    assert emb.device == "cpu"
    assert emb.batch_size == 32
    assert emb.max_length == 128
    assert emb.embedding_dim == 768  # FinBERT base contract — locked


def test_finbert_embedder_constructs_with_overrides() -> None:
    """Constructor honors override args (Phase 2 will use these)."""
    from src.data.news import FinBERTEmbedder

    emb = FinBERTEmbedder(
        model_name="custom/finbert-variant",
        device="mps",
        batch_size=64,
        max_length=256,
    )
    assert emb.model_name == "custom/finbert-variant"
    assert emb.device == "mps"
    assert emb.batch_size == 64
    assert emb.max_length == 256


def test_finbert_embed_raises_not_implemented() -> None:
    """embed() raises real NotImplementedError; no silent empty-array return."""
    from src.data.news import FinBERTEmbedder

    emb = FinBERTEmbedder()
    ev = _make_event()
    with pytest.raises(NotImplementedError, match="Phase 2"):
        emb.embed([ev])


def test_finbert_embed_raises_not_implemented_on_empty_input() -> None:
    """Even with empty events, the stub fails loudly. Phase 2 will short-circuit."""
    from src.data.news import FinBERTEmbedder

    emb = FinBERTEmbedder()
    with pytest.raises(NotImplementedError, match="Phase 2"):
        emb.embed([])


# ---------------------------------------------------------------------------
# align_news_to_bars stub
# ---------------------------------------------------------------------------


def test_align_news_to_bars_raises_not_implemented() -> None:
    """The Phase 3 alignment function must fail loudly until implemented."""
    from src.data.news import align_news_to_bars

    ev = _make_event()
    base = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)
    bars = [base + timedelta(minutes=15 * i) for i in range(4)]
    with pytest.raises(NotImplementedError, match="Phase 3"):
        align_news_to_bars(events=[ev], bar_timestamps=bars, lookback_window_hours=24)


def test_align_news_to_bars_signature_locked() -> None:
    """Lock the signature so Phase 3 fills the body, not the contract.

    Verifies the function accepts (events, bar_timestamps, lookback_window_hours)
    by attempting the call; the NotImplementedError confirms it accepted args.
    """
    import inspect

    from src.data.news import align_news_to_bars

    sig = inspect.signature(align_news_to_bars)
    params = list(sig.parameters.keys())
    assert params == ["events", "bar_timestamps", "lookback_window_hours"]
    # Default for lookback_window_hours is 24 per design doc §4
    assert sig.parameters["lookback_window_hours"].default == 24


# ---------------------------------------------------------------------------
# Numpy contract sanity (no actual embed call)
# ---------------------------------------------------------------------------


def test_finbert_embedding_dim_constant_for_phase2_planning() -> None:
    """embedding_dim is a class attribute — Phase 3 PCA + alignment depend on it.

    Locking this here means a Phase-2 mistake (e.g., swapping to a 1024-dim
    variant) without updating the contract will fail this test.
    """
    from src.data.news import FinBERTEmbedder

    assert FinBERTEmbedder.embedding_dim == 768
    # numpy import sanity — we use np.float32 dtype downstream
    assert np.dtype(np.float32).itemsize == 4
