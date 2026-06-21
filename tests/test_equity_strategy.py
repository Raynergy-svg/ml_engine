"""US-004 — Tests for the strategy/engine interface contract.

Tests use real classes (no ``unittest.mock``) and real disk via ``tmp_path``.
The point is to assert the seam US-006 (live harvester) and US-017 (control
loop) agree on, and to prove the loop can call a stub strategy end-to-end
and find the validated target weights on disk afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.equity.strategy import (
    GROSS_CAP,
    PER_NAME_ABS_CAP,
    EquityStrategy,
    StrategyError,
    StubEqualWeightStrategy,
    rebalance_loop_step,
    validate_weights,
    weights_from_factor_panel,
)


# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


def test_stub_satisfies_equity_strategy_protocol():
    """Runtime-checkable Protocol: the stub IS an ``EquityStrategy``."""
    stub = StubEqualWeightStrategy(tickers=("SPY", "QQQ"))
    assert isinstance(stub, EquityStrategy)


def test_arbitrary_object_does_not_satisfy_protocol():
    """A class without ``compute_target_weights`` fails the isinstance check."""

    class NotAStrategy:
        pass

    assert not isinstance(NotAStrategy(), EquityStrategy)


# ---------------------------------------------------------------------------
# Stub strategy behavior
# ---------------------------------------------------------------------------


def test_stub_equal_weight_split():
    stub = StubEqualWeightStrategy(
        tickers=("SPY", "QQQ", "IWM"), target_gross=0.9
    )
    asof = pd.Timestamp("2026-06-19")
    weights = stub.compute_target_weights(asof)

    assert set(weights) == {"SPY", "QQQ", "IWM"}
    assert all(abs(w - 0.30) < 1e-12 for w in weights.values())
    assert abs(sum(weights.values()) - 0.9) < 1e-12


def test_stub_empty_universe_returns_flat_book():
    stub = StubEqualWeightStrategy(tickers=())
    assert stub.compute_target_weights(pd.Timestamp("2026-06-19")) == {}


def test_stub_zero_gross_returns_flat_book():
    stub = StubEqualWeightStrategy(tickers=("SPY",), target_gross=0.0)
    assert stub.compute_target_weights(pd.Timestamp("2026-06-19")) == {}


def test_stub_is_deterministic():
    stub = StubEqualWeightStrategy(tickers=("SPY", "QQQ"), target_gross=1.0)
    a = stub.compute_target_weights(pd.Timestamp("2026-06-19"))
    b = stub.compute_target_weights(pd.Timestamp("2026-06-19"))
    assert a == b


# ---------------------------------------------------------------------------
# validate_weights
# ---------------------------------------------------------------------------


def test_validate_weights_drops_zero_entries():
    out = validate_weights({"SPY": 0.5, "QQQ": 0.0})
    assert out == {"SPY": 0.5}


def test_validate_weights_rejects_nonfinite():
    with pytest.raises(StrategyError, match="non-finite"):
        validate_weights({"SPY": float("nan")})


def test_validate_weights_rejects_short_when_long_only():
    with pytest.raises(StrategyError, match="long-only"):
        validate_weights({"SPY": -0.05}, long_only=True)


def test_validate_weights_allows_short_when_not_long_only():
    out = validate_weights({"SPY": -0.05}, long_only=False)
    assert out == {"SPY": -0.05}


def test_validate_weights_rejects_per_name_cap_breach():
    with pytest.raises(StrategyError, match="exceeds cap"):
        validate_weights({"SPY": PER_NAME_ABS_CAP + 0.01})


def test_validate_weights_rejects_gross_cap_breach():
    # Each weight under per-name cap (1.20) but total > GROSS_CAP (4.0).
    book = {f"T{i}": 1.10 for i in range(4)}  # gross = 4.40
    with pytest.raises(StrategyError, match="gross leverage"):
        validate_weights(book, long_only=False)


def test_validate_weights_rejects_non_string_key():
    with pytest.raises(StrategyError, match="non-empty str"):
        validate_weights({1: 0.5})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# weights_from_factor_panel reuses src/factor/portfolio.py
# ---------------------------------------------------------------------------


def _synthetic_panels(n_days: int = 80, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    tickers = ["AAA", "BBB", "CCC"]
    returns = pd.DataFrame(
        rng.normal(0, 0.01, size=(n_days, len(tickers))),
        index=dates,
        columns=tickers,
    )
    scores = pd.DataFrame(
        rng.normal(0, 1.0, size=(n_days, len(tickers))),
        index=dates,
        columns=tickers,
    )
    return scores, returns


def test_weights_from_factor_panel_long_only_drops_negatives():
    scores, returns = _synthetic_panels()
    asof = scores.index[-1]
    weights = weights_from_factor_panel(scores, returns, asof, long_only=True)
    assert all(w >= 0.0 for w in weights.values())
    assert sum(abs(w) for w in weights.values()) <= GROSS_CAP + 1e-9
    assert all(abs(w) <= PER_NAME_ABS_CAP + 1e-9 for w in weights.values())


def test_weights_from_factor_panel_rejects_missing_asof():
    scores, returns = _synthetic_panels()
    bad = pd.Timestamp("1999-01-04")
    with pytest.raises(StrategyError, match="not present"):
        weights_from_factor_panel(scores, returns, bad)


def test_weights_from_factor_panel_is_causal():
    """Slicing to ``[..., asof]`` means later rows cannot affect output."""
    scores, returns = _synthetic_panels()
    asof = scores.index[40]

    early = weights_from_factor_panel(scores, returns, asof)
    # Perturb a future row — must not change the past-only result.
    scores_perturbed = scores.copy()
    scores_perturbed.iloc[60] = scores_perturbed.iloc[60] * 100.0
    late = weights_from_factor_panel(scores_perturbed, returns, asof)
    assert early == late


# ---------------------------------------------------------------------------
# rebalance_loop_step — proves "the loop can call it"
# ---------------------------------------------------------------------------


def test_loop_can_call_stub_strategy_and_persist(tmp_path: Path):
    """The control-loop adapter drives the stub and writes target weights atomically."""
    stub = StubEqualWeightStrategy(
        tickers=("SPY", "QQQ", "IWM"), target_gross=0.9
    )
    state_path = tmp_path / "equity_target.json"
    asof = pd.Timestamp("2026-06-19")

    weights = rebalance_loop_step(stub, asof, state_path=state_path)

    assert state_path.exists()
    payload = json.loads(state_path.read_text())
    assert payload["asof"] == asof.isoformat()
    assert payload["n_names"] == 3
    assert abs(payload["gross"] - 0.9) < 1e-12
    assert payload["long_only"] is True
    assert payload["weights"] == weights
    assert set(weights) == {"SPY", "QQQ", "IWM"}


def test_loop_step_atomic_write_leaves_no_partial_files(tmp_path: Path):
    """A successful write leaves the target file but no leftover tmp siblings."""
    stub = StubEqualWeightStrategy(tickers=("SPY",), target_gross=0.5)
    state_path = tmp_path / "equity_target.json"

    rebalance_loop_step(stub, pd.Timestamp("2026-06-19"), state_path=state_path)

    sibling_tmps = [
        p for p in tmp_path.iterdir()
        if p.name.startswith(f".{state_path.name}") and p.name.endswith(".tmp")
    ]
    assert sibling_tmps == []


def test_loop_step_drops_out_of_universe_tickers(tmp_path: Path):
    """US-002's roster is the source of truth — extras get dropped, not silently filled."""
    stub = StubEqualWeightStrategy(
        tickers=("SPY", "QQQ", "MYSTERY"), target_gross=0.9
    )
    state_path = tmp_path / "equity_target.json"

    weights = rebalance_loop_step(
        stub,
        pd.Timestamp("2026-06-19"),
        state_path=state_path,
        universe=["SPY", "QQQ"],
    )
    assert set(weights) == {"SPY", "QQQ"}


def test_loop_step_rejects_non_strategy_object(tmp_path: Path):
    class NotAStrategy:
        pass

    with pytest.raises(StrategyError, match="protocol"):
        rebalance_loop_step(
            NotAStrategy(),  # type: ignore[arg-type]
            pd.Timestamp("2026-06-19"),
            state_path=tmp_path / "x.json",
        )


def test_loop_step_propagates_validation_failure(tmp_path: Path):
    """If a strategy emits an infeasible book, the loop step refuses to persist it."""

    class BadStrategy:
        def compute_target_weights(self, asof: pd.Timestamp):
            return {"SPY": PER_NAME_ABS_CAP + 1.0}

    state_path = tmp_path / "equity_target.json"
    with pytest.raises(StrategyError, match="exceeds cap"):
        rebalance_loop_step(
            BadStrategy(), pd.Timestamp("2026-06-19"), state_path=state_path
        )
    # Nothing was written.
    assert not state_path.exists()
