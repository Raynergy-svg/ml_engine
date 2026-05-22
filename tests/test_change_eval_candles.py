"""Tests for `ChangeEvalHarness`'s default candle loader.

These tests are deliberately no-mock per `.claude/rules/improvement.md`
(No-Mock Rule, 2026-05-01). They write real CSV files to `tmp_path`,
invoke the real loader / real harness with the real `ReplayValidator`,
and assert against real disk + real ReplayResult output.

Background: the prior default candle loader imported a non-existent
module (`src.scanner.market_data`); every production scorecard for
months produced `{"skipped": true, "reason": "no_candles"}`, which
collapsed `trading_quality` to zero and rejected every self-heal
proposal.  Fix: load from the on-disk training cache that the bot's
training pipelines already populate.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.scanner.automation.change_eval import (
    _DEFAULT_CANDLES_ROOT,
    ChangeEvalHarness,
    _default_candle_loader,
    _normalize_pair,
    _read_candles_csv,
    _resolve_cached_candle_path,
)
from src.scanner.automation.meta_types import (
    ChangeKind,
    ChangePackage,
    Proposal,
)
from src.scanner.automation.replay_validator import ReplayValidator


# --------------------------------------------------------------------------- #
# Helpers — synthetic OHLCV that matches the production CSV schema
# --------------------------------------------------------------------------- #


def _write_csv(path: Path, rows: int = 500, base: float = 1.10) -> Path:
    """Write a deterministic OHLCV CSV in the same shape the training
    pipelines emit (header: `time,open,high,low,close,volume`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        price = base
        # 15-minute candles, march time forward
        for i in range(rows):
            # Small alternating walk so ATR is non-zero and signal logic fires
            delta = 0.0008 if (i % 2 == 0) else -0.0005
            open_p = price
            close_p = price + delta
            high = max(open_p, close_p) + 0.0003
            low = min(open_p, close_p) - 0.0003
            w.writerow(
                [
                    f"2024-01-01 {i // 4:02d}:{(i % 4) * 15:02d}:00+00:00",
                    f"{open_p:.5f}",
                    f"{high:.5f}",
                    f"{low:.5f}",
                    f"{close_p:.5f}",
                    str(100 + i),
                ]
            )
            price = close_p
    return path


def _pkg_with_delta() -> ChangePackage:
    pkg = ChangePackage(kind=ChangeKind.CONFIG)
    pkg.proposal = Proposal(
        config_delta={"confidence_threshold": {"old": 0.55, "new": 0.50}}
    )
    return pkg


# --------------------------------------------------------------------------- #
# T1: harness loads non-zero candles for EUR_USD when real candle files exist
# --------------------------------------------------------------------------- #


def test_default_loader_returns_candles_from_disk(tmp_path, monkeypatch):
    """T1 — given a real CSV at the canonical cache path, the default loader
    returns a non-empty list of dict-shaped candles with the keys
    `ReplayValidator` consumes."""
    cache_dir = tmp_path / "trained_data" / "cache" / "training_data"
    csv_path = cache_dir / "EUR_USD_M15_65000.csv"
    _write_csv(csv_path, rows=200)

    # Point the loader at our temp cache (one monkeypatch — the rest is real)
    monkeypatch.setattr(
        "src.scanner.automation.change_eval._DEFAULT_CANDLES_ROOT",
        cache_dir,
    )

    candles = _default_candle_loader("EUR_USD", days=30)
    assert len(candles) > 0, "loader returned [] despite real CSV on disk"
    # Schema: every candle has the keys ReplayValidator._compute_atr touches
    for c in candles:
        assert "open" in c and "high" in c and "low" in c and "close" in c
        assert isinstance(c["close"], float)
        assert c["high"] >= c["low"]


# --------------------------------------------------------------------------- #
# T2: `no_candles` ONLY when the cache is genuinely empty, NOT on path bugs
# --------------------------------------------------------------------------- #


def test_loader_returns_empty_only_when_cache_truly_missing(tmp_path, monkeypatch):
    """T2 — when the cache directory exists but has no file for the pair,
    the loader returns []. When the cache directory itself is missing, the
    loader also returns []. Both paths are intentional. But when a real
    file exists, it MUST return a non-empty list (the pre-fix bug).
    """
    # Sub-case A: cache directory exists but no file for EUR_USD
    cache_dir = tmp_path / "cache_a"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "src.scanner.automation.change_eval._DEFAULT_CANDLES_ROOT", cache_dir
    )
    assert _default_candle_loader("EUR_USD", days=30) == []

    # Sub-case B: cache directory itself missing
    cache_dir_missing = tmp_path / "does_not_exist"
    monkeypatch.setattr(
        "src.scanner.automation.change_eval._DEFAULT_CANDLES_ROOT", cache_dir_missing
    )
    assert _default_candle_loader("EUR_USD", days=30) == []

    # Sub-case C: file present → MUST NOT return []  (the pre-fix bug)
    cache_dir_c = tmp_path / "cache_c"
    _write_csv(cache_dir_c / "EUR_USD_M15_65000.csv", rows=200)
    monkeypatch.setattr(
        "src.scanner.automation.change_eval._DEFAULT_CANDLES_ROOT", cache_dir_c
    )
    candles = _default_candle_loader("EUR_USD", days=30)
    assert len(candles) >= 200, (
        f"loader returned {len(candles)} candles despite a real 200-row CSV on disk; "
        "this is the pre-fix bug regression sentinel"
    )


# --------------------------------------------------------------------------- #
# T3: symbol normalization round-trips against the on-disk convention
# --------------------------------------------------------------------------- #


def test_symbol_normalization_roundtrips(tmp_path, monkeypatch):
    """T3 — `EUR_USD`, `EUR/USD`, `EURUSD`, `eur_usd`, `  EUR_USD ` must all
    resolve to the same on-disk CSV. The cache file lives at
    `EUR_USD_M15_*.csv` (the production naming convention).
    """
    cache_dir = tmp_path / "trained_data" / "cache" / "training_data"
    _write_csv(cache_dir / "EUR_USD_M15_65000.csv", rows=200)
    monkeypatch.setattr(
        "src.scanner.automation.change_eval._DEFAULT_CANDLES_ROOT", cache_dir
    )

    assert _normalize_pair("EUR_USD") == "EUR_USD"
    assert _normalize_pair("EUR/USD") == "EUR_USD"
    assert _normalize_pair("EURUSD") == "EUR_USD"
    assert _normalize_pair("eur_usd") == "EUR_USD"
    assert _normalize_pair("  EUR_USD  ") == "EUR_USD"
    assert _normalize_pair("eur-usd") == "EUR_USD"

    # All variants resolve to the same path
    canonical = _resolve_cached_candle_path("EUR_USD")
    assert canonical is not None
    for variant in ("EUR/USD", "EURUSD", "eur_usd", "  EUR_USD  "):
        assert _resolve_cached_candle_path(variant) == canonical, (
            f"variant {variant!r} did not resolve to the same path as EUR_USD"
        )


# --------------------------------------------------------------------------- #
# T4: harness output is structurally identical to what production consumes
# --------------------------------------------------------------------------- #


def test_candle_shape_matches_replay_validator_contract(tmp_path, monkeypatch):
    """T4 — `ReplayValidator._compute_atr` and `_simulate_signal` read
    `high`, `low`, `close`, `open` directly off candle dicts. The harness's
    candle output must populate those keys as floats. Run the real
    ReplayValidator over the loaded candles and assert it produces a
    `ReplayResult` (i.e. didn't bail early on `MIN_CANDLES` or KeyError).
    """
    cache_dir = tmp_path / "trained_data" / "cache" / "training_data"
    _write_csv(cache_dir / "EUR_USD_M15_65000.csv", rows=500)
    monkeypatch.setattr(
        "src.scanner.automation.change_eval._DEFAULT_CANDLES_ROOT", cache_dir
    )

    candles = _default_candle_loader("EUR_USD", days=30)
    # Production replay needs at least MIN_CANDLES (100) bars to proceed.
    assert len(candles) >= 100, (
        f"loader returned {len(candles)} candles, below MIN_CANDLES (100) — "
        "replay would bail without ever testing the config delta"
    )

    # Required keys are floats (not strings, not nested OANDA mid.h/c/l blobs)
    sample = candles[50]
    for k in ("open", "high", "low", "close"):
        assert isinstance(sample[k], float), (
            f"{k} was {type(sample[k]).__name__}, expected float — "
            "ReplayValidator._compute_atr expects numeric OHLC"
        )

    validator = ReplayValidator()
    result = validator.replay_scan(
        "EUR_USD", candles, {"confidence_threshold": 0.5}, "test_label"
    )
    # ReplayResult is dataclass-like — its to_dict() must contain the keys the
    # scorecard aggregator pulls (`total_pnl`, `sharpe`, `win_rate`, `trades`).
    metrics = result.to_dict()
    for required_key in ("total_pnl", "sharpe", "win_rate", "trades"):
        assert required_key in metrics, (
            f"ReplayResult.to_dict() missing {required_key!r} — "
            "harness aggregation would crash or silently drop the value"
        )


# --------------------------------------------------------------------------- #
# T5: full harness loop produces a non-zero trading_quality scorecard
# --------------------------------------------------------------------------- #


def test_harness_trading_quality_no_longer_skipped(tmp_path, monkeypatch):
    """T5 — end-to-end: with the fixed candle loader and a real
    ReplayValidator, the trading_quality SubScore must NOT carry
    `{"skipped": True, "reason": "no_candles"}`. The score may pass OR
    fail on its merits — the bug we're fixing is the *skipped* sentinel,
    not the threshold semantics.
    """
    cache_dir = tmp_path / "trained_data" / "cache" / "training_data"
    _write_csv(cache_dir / "EUR_USD_M15_65000.csv", rows=400)
    monkeypatch.setattr(
        "src.scanner.automation.change_eval._DEFAULT_CANDLES_ROOT", cache_dir
    )

    harness = ChangeEvalHarness(
        replay_validator=ReplayValidator(),
        qa_pipeline=None,           # governance is out of scope here
        pytest_runner=lambda _t: {  # avoid spawning real pytest in this test
            "passed": True, "exit_code": 0, "failed": [], "stdout_tail": "ok",
        },
        regime_windows_path=tmp_path / "no_regime.json",  # skip regime layer
        scorecard_log_path=tmp_path / "scorecards.jsonl",
        baseline_config={"confidence_threshold": 0.55},
        pairs=["EUR_USD"],
        history_days=10,
    )
    card = harness.score(_pkg_with_delta())

    assert card.trading_quality is not None
    details = card.trading_quality.details or {}
    per_pair = details.get("per_pair") or []
    assert len(per_pair) == 1, "expected exactly one per-pair entry"
    entry = per_pair[0]
    assert entry.get("pair") == "EUR_USD"
    assert entry.get("skipped") is not True, (
        f"trading_quality skipped with reason={entry.get('reason')!r} despite "
        f"real candles on disk — the regression we're fixing"
    )
    # The aggregate keys must exist and be finite (may be zero if the
    # synthetic data produced no qualifying trades; that is acceptable —
    # we are testing the wiring, not the alpha)
    for k in ("pnl_delta_r", "sharpe_delta", "win_rate", "sharpe"):
        assert k in details
        assert math.isfinite(details[k]), f"{k} is not finite: {details[k]!r}"


# --------------------------------------------------------------------------- #
# T6: real on-disk EUR_USD cache is readable (live-data sanity check)
# --------------------------------------------------------------------------- #


def test_real_production_cache_is_readable():
    """T6 — sanity: the actual production cache at
    `trained_data/cache/training_data/EUR_USD_M15_*.csv` is present and
    loadable. If this fails on CI, the operator needs to know — but in
    the dev environment where this committed, the cache is present.
    """
    if not _DEFAULT_CANDLES_ROOT.is_dir():
        pytest.skip("no production candle cache on this machine")
    p = _resolve_cached_candle_path("EUR_USD")
    if p is None:
        pytest.skip("no EUR_USD cached candles on this machine")
    candles = _read_candles_csv(p, limit=500)
    assert len(candles) > 0
    # Schema check on the real file
    c = candles[-1]
    assert isinstance(c["close"], float)
    assert c["high"] >= c["low"]
