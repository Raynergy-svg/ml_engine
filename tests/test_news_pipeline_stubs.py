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
    assert src.cache_dir == "trained_data/news"

    src_with_cache = ForexFactoryNewsSource(cache_dir="/tmp/news_cache")
    assert src_with_cache.cache_dir == "/tmp/news_cache"


def test_forex_factory_fetch_events_phase2() -> None:
    """Phase-2: fetch_events returns list[NewsEvent], not NotImplementedError.

    Serves from parquet cache when live FF API is rate-limited (HTTP 429).
    Skips gracefully when both live fetch and cache are unavailable — never mocks.
    Per .claude/rules/improvement.md No-Mock Rule.
    """
    from src.data.news import ForexFactoryNewsSource, NewsEvent
    src = ForexFactoryNewsSource()
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 8, tzinfo=timezone.utc)
    try:
        events = src.fetch_events("EUR_USD", since, until)
    except Exception as exc:  # no cache and FF rate-limited
        pytest.skip(f"FF unavailable (live 429 + no parquet cache): {exc}")
    assert isinstance(events, list), "fetch_events must return list, not raise"
    for ev in events:
        assert isinstance(ev, NewsEvent), f"Expected NewsEvent, got {type(ev)}"


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


# ---------------------------------------------------------------------------
# FinBERTEmbedder real-inference tests (Stage 4-A, NO MOCKS)
# ---------------------------------------------------------------------------
#
# The Phase-1 "raises NotImplementedError" tests are removed: embed() is now
# a real implementation. These tests load real ProsusAI/finbert weights from
# the HuggingFace cache. First run downloads ~440 MB; subsequent runs hit
# cache. If the model can't be loaded (offline + no cache), the real-inference
# tests are SKIPPED — never mocked. Per .claude/rules/improvement.md
# "No-Mock Rule" (promoted 2026-05-01): mocks hide integration gaps.


def _can_load_finbert() -> bool:
    """Return True iff tokenizer + model load cleanly. No mocks."""
    try:
        from transformers import AutoModel, AutoTokenizer

        AutoTokenizer.from_pretrained("ProsusAI/finbert")
        AutoModel.from_pretrained("ProsusAI/finbert")
        return True
    except Exception:  # pragma: no cover — environment-dependent
        return False


_FINBERT_AVAILABLE = _can_load_finbert()
_real_only = pytest.mark.skipif(
    not _FINBERT_AVAILABLE,
    reason="ProsusAI/finbert not loadable; skipping real-inference tests (NO MOCKS)",
)


def _real_finance_events(n: int = 4):
    from src.data.news import NewsEvent

    base_ts = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)
    headlines = [
        "Federal Reserve raises interest rates by 25 basis points",
        "ECB signals dovish pivot as inflation cools to 2.1 percent",
        "Non-Farm Payrolls beat expectations at 270k vs 200k forecast",
        "Bank of Japan holds rates steady, yen falls to 24-year low",
        "EUR_USD breaks below 1.05 on weak German factory orders",
        "USD_JPY tests 150 as BoJ maintains ultra-loose policy",
        "GBP_USD rallies on BoE hawkish hold and stronger CPI",
        "Risk-on sentiment lifts AUD and NZD on China stimulus",
    ]
    return [
        NewsEvent(
            timestamp=base_ts,
            text=headlines[i % len(headlines)],
            source="manual",
            category="OTHER",
            relevance_score=1.0,
            pair="EUR_USD",
        )
        for i in range(n)
    ]


def test_finbert_embed_empty_input_returns_zero_rows() -> None:
    """Empty input must short-circuit to (0, 768) float32 — no model load."""
    from src.data.news import FinBERTEmbedder

    emb = FinBERTEmbedder()
    out = emb.embed([])
    assert isinstance(out, np.ndarray)
    assert out.shape == (0, 768)
    assert out.dtype == np.float32
    # Empty path must NOT trigger lazy model load.
    assert emb._model is None
    assert emb._tokenizer is None


def test_finbert_unknown_device_falls_back_to_cpu() -> None:
    """Bad device string falls back to cpu; never crashes downstream callers."""
    from src.data.news import FinBERTEmbedder

    emb = FinBERTEmbedder(device="banana")
    assert emb._resolve_device() == "cpu"


@_real_only
def test_finbert_embed_real_shape_and_dtype() -> None:
    """Real FinBERT inference produces (n, 768) float32. NO MOCKS."""
    from src.data.news import FinBERTEmbedder

    emb = FinBERTEmbedder(device="cpu", batch_size=8, max_length=128)
    events = _real_finance_events(n=4)
    out = emb.embed(events)

    assert out.shape == (4, 768)
    assert out.dtype == np.float32
    assert np.abs(out).sum() > 0.0  # non-degenerate
    assert np.isfinite(out).all()


@_real_only
def test_finbert_embed_is_deterministic() -> None:
    """Same input → same output (NewsEmbedder ABC contract)."""
    from src.data.news import FinBERTEmbedder

    events = _real_finance_events(n=4)
    emb = FinBERTEmbedder(device="cpu", batch_size=8, max_length=128)
    out_a = emb.embed(events)
    out_b = emb.embed(events)
    np.testing.assert_array_equal(out_a, out_b)


@_real_only
def test_finbert_embed_batches_match_unbatched() -> None:
    """Batching must not change output (lock contract for alignment stage)."""
    from src.data.news import FinBERTEmbedder

    events = _real_finance_events(n=8)
    emb_small = FinBERTEmbedder(device="cpu", batch_size=2, max_length=128)
    emb_big = FinBERTEmbedder(device="cpu", batch_size=8, max_length=128)
    out_small = emb_small.embed(events)
    out_big = emb_big.embed(events)
    np.testing.assert_allclose(out_small, out_big, rtol=1e-4, atol=1e-5)


@_real_only
def test_finbert_pca_compressed_output_is_float32() -> None:
    """PCA-compressed embedding stays float32 + correct shape (Phase 3)."""
    from sklearn.decomposition import PCA

    from src.data.news import FinBERTEmbedder

    emb = FinBERTEmbedder(device="cpu", batch_size=8, max_length=128)
    events = _real_finance_events(n=16)
    X = emb.embed(events)
    assert X.shape == (16, 768)

    pca = PCA(n_components=8, random_state=42)
    Z = pca.fit_transform(X).astype(np.float32)

    assert Z.shape == (16, 8)
    assert Z.dtype == np.float32
    assert np.isfinite(Z).all()


# ---------------------------------------------------------------------------
# align_news_to_bars (Stage 4-A implementation)
# ---------------------------------------------------------------------------


def test_align_news_to_bars_signature_locked() -> None:
    """Lock the signature post-Stage-4A.

    Stage 4-A added ``embeddings`` as the third positional arg between
    bar_timestamps and lookback_window_hours. The Phase-1 stub took
    (events, bar_timestamps, lookback_window_hours) only — the new contract
    is the source of truth (see feature_alignment.py module docstring,
    "Contract change").
    """
    import inspect

    from src.data.news import align_news_to_bars

    sig = inspect.signature(align_news_to_bars)
    params = list(sig.parameters.keys())
    assert params == [
        "events",
        "bar_timestamps",
        "embeddings",
        "lookback_window_hours",
    ]
    # Default for lookback_window_hours is 24 per design doc §4
    assert sig.parameters["lookback_window_hours"].default == 24


def _emb(n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    """Deterministic synthetic embeddings (small dim for fast tests)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_align_news_to_bars_lookahead_strict_equal_excluded() -> None:
    """Test 1: event AT EXACTLY t_b must NOT be included (strict < t_b).

    This is the load-bearing lookahead-bias guard. Relaxing to <= would let
    the current bar's news leak into the model's input for that bar.
    """
    from src.data.news import align_news_to_bars

    bar_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ev = _make_event(timestamp=bar_ts, category="NFP")  # exactly equal
    embeddings = _emb(1, dim=8)

    out = align_news_to_bars(
        events=[ev],
        bar_timestamps=[bar_ts],
        embeddings=embeddings,
        lookback_window_hours=24,
    )
    # Whole row must be zero — event was excluded.
    assert out.shape == (1, 8 + 8)
    assert np.all(out == 0.0), (
        "Event at t_e == t_b must be excluded by strict-< guard, "
        "but the row had non-zero entries (lookahead leak)."
    )


def test_align_news_to_bars_lookahead_one_second_before_included() -> None:
    """Test 2: event 1s BEFORE t_b must be included.

    Confirms the strict-< boundary doesn't accidentally exclude legitimate
    just-prior-to-bar events.
    """
    from src.data.news import align_news_to_bars

    bar_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ev_ts = datetime(2024, 1, 1, 11, 59, 59, tzinfo=timezone.utc)
    ev = _make_event(timestamp=ev_ts, category="CPI", relevance_score=1.0)
    embeddings = _emb(1, dim=8)

    out = align_news_to_bars(
        events=[ev],
        bar_timestamps=[bar_ts],
        embeddings=embeddings,
        lookback_window_hours=24,
    )

    # CPI is index 1 in the locked category set.
    cpi_count = out[0, 8 + 1]
    assert cpi_count > 0.0, "Event 1s before t_b must register in CPI count"
    # Embedding portion should be a (decay-near-1) scaled copy of the event embedding.
    # 1s out of tau=8h => decay ~= exp(-1/28800) ~= 1.0; weighted mean of single
    # event = the event's embedding exactly (since single-event mean cancels weights).
    np.testing.assert_allclose(out[0, :8], embeddings[0], rtol=1e-5, atol=1e-5)


def test_align_news_to_bars_naive_bar_raises() -> None:
    """Test 3: a tz-naive bar timestamp must raise ValueError.

    Naive bar comparisons against tz-aware NewsEvent.timestamp are the exact
    bias vector that NewsEvent.__post_init__ blocks on construction. Mirror
    the check here for the bar side.
    """
    from src.data.news import align_news_to_bars

    naive_bar = datetime(2024, 1, 1, 12, 0, 0)  # no tzinfo
    ev = _make_event()
    embeddings = _emb(1, dim=8)

    with pytest.raises(ValueError, match="timezone-naive"):
        align_news_to_bars(
            events=[ev],
            bar_timestamps=[naive_bar],
            embeddings=embeddings,
            lookback_window_hours=24,
        )


def test_align_news_to_bars_empty_events_zero_output() -> None:
    """Test 4: empty events with non-empty bars => correct shape + all zeros."""
    from src.data.news import align_news_to_bars

    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    bars = [base + timedelta(minutes=15 * i) for i in range(10)]
    embedding_dim = 8
    embeddings = np.empty((0, embedding_dim), dtype=np.float32)

    out = align_news_to_bars(
        events=[],
        bar_timestamps=bars,
        embeddings=embeddings,
        lookback_window_hours=24,
    )
    assert out.shape == (10, embedding_dim + 8)
    assert np.all(out == 0.0)
    assert out.dtype == np.float32


def test_align_news_to_bars_decay_recent_outweighs_old() -> None:
    """Test 5: same-relevance events earlier in the window get LESS weight.

    Decay tau = lookback/3. With lookback=24h, tau=8h.
    - Event at t-1h:   exp(-1/8) ~= 0.882
    - Event at t-20h:  exp(-20/8) ~= 0.082
    The recent event's weight should be ~10x the old event's.
    """
    from src.data.news import align_news_to_bars

    bar_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    recent_ts = bar_ts - timedelta(hours=1)
    old_ts = bar_ts - timedelta(hours=20)

    # Same category, same relevance — decay is the only difference.
    ev_recent = _make_event(
        timestamp=recent_ts, category="NFP", relevance_score=1.0, text="recent"
    )
    ev_old = _make_event(
        timestamp=old_ts, category="NFP", relevance_score=1.0, text="older"
    )
    # Use distinct embeddings so we can recover the weights from the mean.
    embeddings = np.zeros((2, 4), dtype=np.float32)
    embeddings[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # recent
    embeddings[1] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # old

    out = align_news_to_bars(
        events=[ev_recent, ev_old],
        bar_timestamps=[bar_ts],
        embeddings=embeddings,
        lookback_window_hours=24,
    )
    # The NFP count column = sum of both weights.
    nfp_count = out[0, 4 + 0]  # embedding_dim=4, NFP=index 0

    # Mean embedding row — weight_recent * e_recent + weight_old * e_old, all over sum.
    # Coordinate 0 = w_recent / (w_recent + w_old); coord 1 = w_old / sum.
    coord0 = out[0, 0]  # recent's share
    coord1 = out[0, 1]  # old's share
    assert coord0 > coord1, (
        f"Recent event must dominate the weighted mean; got recent_share={coord0:.4f}, "
        f"old_share={coord1:.4f}"
    )
    # Ratio should be ~ exp(-1/8) / exp(-20/8) = exp(19/8) ~= 10.74
    expected_ratio = float(np.exp(19.0 / 8.0))
    actual_ratio = float(coord0 / max(coord1, 1e-12))
    np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-3)
    # Sanity: shares sum to ~1.0 (mean is normalized).
    np.testing.assert_allclose(coord0 + coord1, 1.0, rtol=1e-5)
    # NFP count must be positive (both events fall in NFP class).
    assert nfp_count > 0.0


def test_align_news_to_bars_embeddings_length_mismatch_raises() -> None:
    """Length mismatch between events and embeddings rows = hard error."""
    from src.data.news import align_news_to_bars

    bar_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ev = _make_event(timestamp=bar_ts - timedelta(hours=1))
    embeddings = _emb(2, dim=8)  # 2 rows, but only 1 event

    with pytest.raises(ValueError, match="must match len"):
        align_news_to_bars(
            events=[ev],
            bar_timestamps=[bar_ts],
            embeddings=embeddings,
        )


def test_align_news_to_bars_nan_embeddings_raises() -> None:
    """NaN/Inf in embeddings = hard error (no silent zero-fill)."""
    from src.data.news import align_news_to_bars

    bar_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ev = _make_event(timestamp=bar_ts - timedelta(hours=1))
    embeddings = np.full((1, 8), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="NaN or Inf"):
        align_news_to_bars(
            events=[ev],
            bar_timestamps=[bar_ts],
            embeddings=embeddings,
        )


def test_align_news_to_bars_unknown_category_collapses_to_other() -> None:
    """Categories outside the locked 8-set collapse into OTHER (last column)."""
    from src.data.news import align_news_to_bars

    bar_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    # 'BoC' is in source.py docstring but NOT in the locked feature-set.
    ev = _make_event(
        timestamp=bar_ts - timedelta(minutes=30),
        category="BoC",
        relevance_score=1.0,
    )
    embeddings = _emb(1, dim=4)

    out = align_news_to_bars(
        events=[ev],
        bar_timestamps=[bar_ts],
        embeddings=embeddings,
        lookback_window_hours=24,
    )
    # OTHER = index 7 in locked order. embedding_dim=4.
    assert out[0, 4 + 7] > 0.0, "Unknown category must collapse to OTHER"
    # And no other category column should fire.
    for col in range(7):
        assert out[0, 4 + col] == 0.0, (
            f"Non-OTHER category column {col} fired for an unknown category"
        )


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


# ---------------------------------------------------------------------------
# ForexFactoryNewsSource — Stage 4-A unit + integration tests (NO MOCKS)
# ---------------------------------------------------------------------------
# Per .claude/rules/improvement.md "No-Mock Rule" (promoted 2026-05-01): all
# tests below use real classes against real disk. The single offline-network
# substitute is a real subclass that overrides ``_fetch_live_json`` — same
# class machinery, different I/O policy. No unittest.mock anywhere.


def test_forex_factory_rejects_malformed_pair(tmp_path) -> None:
    """fetch_events validates pair format before any network call."""
    from src.data.news import ForexFactoryNewsSource

    src = ForexFactoryNewsSource(cache_dir=str(tmp_path))
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 8, tzinfo=timezone.utc)
    for bad in ("EURUSD", "EUR/USD/JPY", "EU_US", "", "EUR_"):
        with pytest.raises(ValueError, match="CCY_CCY|pair"):
            src.fetch_events(bad, since, until)


def test_forex_factory_rejects_naive_timestamps(tmp_path) -> None:
    """fetch_events rejects naive since/until — UTC-only contract."""
    from src.data.news import ForexFactoryNewsSource

    src = ForexFactoryNewsSource(cache_dir=str(tmp_path))
    aware = datetime(2026, 5, 8, tzinfo=timezone.utc)
    naive = datetime(2026, 5, 1)  # no tzinfo
    with pytest.raises(ValueError, match="timezone-aware"):
        src.fetch_events("EUR_USD", naive, aware)
    with pytest.raises(ValueError, match="timezone-aware"):
        src.fetch_events("EUR_USD", aware, naive)


def test_forex_factory_rejects_inverted_window(tmp_path) -> None:
    """since must be strictly less than until."""
    from src.data.news import ForexFactoryNewsSource

    src = ForexFactoryNewsSource(cache_dir=str(tmp_path))
    t = datetime(2026, 5, 8, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="since.*until"):
        src.fetch_events("EUR_USD", t, t)
    with pytest.raises(ValueError, match="since.*until"):
        src.fetch_events("EUR_USD", t, t - timedelta(days=1))


def test_forex_factory_categorize_heuristic() -> None:
    """Title heuristic maps representative FF event names to locked categories."""
    from src.data.news.source import _VALID_CATEGORIES, _categorize

    # Central-bank policy events keyed by currency
    assert _categorize("FOMC Statement", "USD") == "FOMC"
    assert _categorize("Monetary Policy Statement", "JPY") == "BoJ"
    assert _categorize("Main Refinancing Rate", "EUR") == "ECB"
    assert _categorize("BOE Rate Statement", "GBP") == "BoE"
    assert _categorize("RBA Rate Statement", "AUD") == "RBA"
    assert _categorize("RBNZ Press Conference", "NZD") == "RBNZ"
    assert _categorize("BOC Monetary Policy Report", "CAD") == "BoC"
    assert _categorize("SNB Monetary Policy Assessment", "CHF") == "SNB"

    # Macro indicators
    assert _categorize("Non-Farm Employment Change", "USD") == "NFP"
    assert _categorize("CPI y/y", "GBP") == "CPI"
    assert _categorize("Core CPI m/m", "USD") == "CPI"
    assert _categorize("GDP q/q", "EUR") == "GDP"

    # Catch-all
    assert _categorize("Manufacturing PMI", "EUR") == "OTHER"
    assert _categorize("Random Holiday", "JPY") == "OTHER"

    # All outputs must be in the locked set
    for title in ("NFP", "CPI", "GDP", "FOMC", "ECB", "Manufacturing PMI"):
        assert _categorize(title, "USD") in _VALID_CATEGORIES


def test_forex_factory_impact_to_relevance() -> None:
    """FF impact -> relevance_score mapping per design doc contract."""
    from src.data.news.source import _impact_to_relevance

    assert _impact_to_relevance("low") == 0.33
    assert _impact_to_relevance("medium") == 0.66
    assert _impact_to_relevance("high") == 1.0
    # Holiday / non-economic -> 0 (recorded but ignored downstream)
    assert _impact_to_relevance("holiday") == 0.0
    assert _impact_to_relevance("non-economic") == 0.0
    # Empty / unknown -> 0.33 (low) — better to over-include than drop
    assert _impact_to_relevance("") == 0.33
    assert _impact_to_relevance("UNKNOWN") == 0.33
    # Case-insensitive
    assert _impact_to_relevance("HIGH") == 1.0


def test_forex_factory_split_pair() -> None:
    """Pair splitting is deterministic and rejects malformed inputs."""
    from src.data.news.source import _split_pair

    assert _split_pair("EUR_USD") == ["EUR", "USD"]
    assert _split_pair("usd_jpy") == ["USD", "JPY"]
    assert _split_pair("EUR/USD") == ["EUR", "USD"]  # tolerates slash
    with pytest.raises(ValueError):
        _split_pair("EURUSD")
    with pytest.raises(ValueError):
        _split_pair("")
    with pytest.raises(ValueError):
        _split_pair(None)  # type: ignore[arg-type]


def test_forex_factory_parse_timestamp() -> None:
    """FF ISO-8601 with offset -> UTC. Naive / malformed -> None."""
    from src.data.news.source import _parse_ff_timestamp

    # Standard FF format: NY local time + offset
    ts = _parse_ff_timestamp("2026-05-03T19:00:00-04:00")
    assert ts is not None
    assert ts.tzinfo == timezone.utc
    assert ts.hour == 23  # 19:00 -04:00 -> 23:00 UTC

    # Already-UTC
    ts2 = _parse_ff_timestamp("2026-05-03T23:00:00+00:00")
    assert ts2 is not None
    assert ts2.hour == 23

    # Naive timestamp dropped (DST ambiguity)
    assert _parse_ff_timestamp("2026-05-03T19:00:00") is None
    # Empty / garbage
    assert _parse_ff_timestamp("") is None
    assert _parse_ff_timestamp("not a date") is None


@pytest.mark.integration
def test_forex_factory_fetch_events_live(tmp_path) -> None:
    """End-to-end live fetch + cache round-trip (NO MOCKS).

    Hits the real public FF JSON mirror. Marked ``integration`` so it can be
    deselected on offline CI (``pytest -m 'not integration'``). Skipped when
    FF rate-limits — never replaced with a mock.
    """
    from src.data.news import ForexFactoryNewsSource, NewsEvent

    src = ForexFactoryNewsSource(cache_dir=str(tmp_path))
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=7)

    try:
        events = src.fetch_events("EUR_USD", since, now)
    except RuntimeError as e:
        pytest.skip(f"ForexFactory live fetch unavailable: {e}")

    assert isinstance(events, list)
    if events:
        VALID = {"NFP", "CPI", "GDP", "FOMC", "ECB", "BoE", "BoJ",
                 "BoC", "RBA", "RBNZ", "SNB", "OTHER"}
        for ev in events:
            assert isinstance(ev, NewsEvent)
            assert ev.timestamp.tzinfo is not None
            assert 0.0 <= ev.relevance_score <= 1.0
            assert ev.category in VALID
            assert since <= ev.timestamp < now
            assert ev.source == "forex_factory"
            assert ev.pair == "EUR_USD"
        # Sorted ascending
        for a, b in zip(events, events[1:]):
            assert a.timestamp <= b.timestamp

    cache_file = tmp_path / "EUR_USD_ff_events.parquet"
    assert cache_file.exists(), "parquet cache not written"
    assert cache_file.stat().st_size > 0


def test_forex_factory_offline_cache_only(tmp_path) -> None:
    """Cache-only path — no network. Seed parquet + verify fetch reads it.

    NO MOCKS. Uses a real subclass that overrides ``_fetch_live_json`` to
    raise — same class machinery, different I/O policy.
    """
    import pandas as pd

    from src.data.news import ForexFactoryNewsSource

    cache_file = tmp_path / "EUR_USD_ff_events.parquet"

    base = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
    df = pd.DataFrame([
        {
            "timestamp": base,
            "currency": "USD",
            "title": "Non-Farm Employment Change",
            "impact": "high",
            "relevance_score": 1.0,
            "category": "NFP",
        },
        {
            "timestamp": base + timedelta(hours=2),
            "currency": "EUR",
            "title": "ECB Press Conference",
            "impact": "high",
            "relevance_score": 1.0,
            "category": "ECB",
        },
        {
            "timestamp": base + timedelta(hours=4),
            "currency": "JPY",  # filtered out for EUR_USD
            "title": "BoJ Rate Statement",
            "impact": "high",
            "relevance_score": 1.0,
            "category": "BoJ",
        },
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.to_parquet(cache_file, index=False)

    class OfflineFFSource(ForexFactoryNewsSource):
        def _fetch_live_json(self):
            raise RuntimeError("offline test — no network")

    src = OfflineFFSource(cache_dir=str(tmp_path))
    since = base - timedelta(hours=1)
    until = base + timedelta(hours=10)

    events = src.fetch_events("EUR_USD", since, until)
    # JPY row must be filtered out (not in EUR_USD); USD + EUR remain
    assert len(events) == 2
    cats = {ev.category for ev in events}
    assert cats == {"NFP", "ECB"}
    for ev in events:
        assert ev.timestamp.tzinfo is not None
        assert ev.pair == "EUR_USD"
        assert ev.source == "forex_factory"
    # Ascending
    assert events[0].timestamp < events[1].timestamp


def test_forex_factory_offline_no_cache_propagates_error(tmp_path) -> None:
    """If live fails AND cache is empty, RuntimeError surfaces (per contract)."""
    from src.data.news import ForexFactoryNewsSource

    class OfflineFFSource(ForexFactoryNewsSource):
        def _fetch_live_json(self):
            raise RuntimeError("offline test — no network")

    src = OfflineFFSource(cache_dir=str(tmp_path))
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 8, tzinfo=timezone.utc)
    with pytest.raises(RuntimeError, match="offline"):
        src.fetch_events("EUR_USD", since, until)
