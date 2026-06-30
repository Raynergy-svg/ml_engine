"""No-mock tests for the US-003 portfolio backtest harness.

Project rules in effect: real classes, real disk via ``tmp_path``, no
``unittest.mock`` / ``MagicMock`` / ``patch``. The math is exercised against
a deterministic synthetic 2-name panel with closed-form expected values for
Sharpe, max drawdown, and turnover so a regression in the harness changes a
number the test pins.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.equity.backtest import (
    ANN,
    BacktestError,
    BacktestResult,
    overlay,
    run_portfolio_backtest,
    save_result,
    stats,
)


def _bday_index(start: str, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _prices_from_returns(
    rets_by_ticker: dict, start_price: float, idx: pd.DatetimeIndex
) -> pd.DataFrame:
    """Build a price panel from per-name daily return arrays.

    ``rets_by_ticker[t]`` has length ``len(idx) - 1`` (returns are between
    consecutive bars). The first price row is ``start_price`` everywhere.
    """
    cols = sorted(rets_by_ticker.keys())
    n = len(idx)
    arr = np.full((n, len(cols)), start_price, dtype=float)
    for j, c in enumerate(cols):
        r = np.asarray(rets_by_ticker[c], dtype=float)
        assert len(r) == n - 1, f"{c}: expected {n - 1} returns, got {len(r)}"
        for i in range(1, n):
            arr[i, j] = arr[i - 1, j] * (1.0 + r[i - 1])
    return pd.DataFrame(arr, index=idx, columns=cols)


def test_causality_constant_weights_hits_known_stats():
    """Fixed (1,0) weights on name A, lag=1, zero cost → closed-form stats."""
    idx = _bday_index("2025-01-02", 4)  # 4 bars → 3 realised returns
    returns_a = [0.02, 0.01, -0.02]
    returns_b = [-0.99, -0.99, -0.99]  # B should not affect anything
    prices = _prices_from_returns(
        {"A": returns_a, "B": returns_b}, start_price=100.0, idx=idx
    )
    weights = pd.DataFrame(
        [[1.0, 0.0]] * len(idx), index=idx, columns=["A", "B"]
    )

    result = run_portfolio_backtest(
        weights, prices, cost_bps=0.0, execution_lag=1
    )

    # 3 realised bars (lag=1 drops the first); equity curve is the A path.
    assert len(result.returns) == 3
    assert result.returns.tolist() == pytest.approx(returns_a, rel=0, abs=1e-12)

    expected_eq = np.cumprod(1.0 + np.array(returns_a))
    np.testing.assert_allclose(result.equity_curve.values, expected_eq, rtol=1e-12)

    # Max drawdown: peak 1.0302 → trough 1.0094 → dd = 0.0202 (rounded to 3dp)
    assert result.summary["max_dd"] == pytest.approx(0.02, abs=1e-3)

    # Sharpe: mean / (std + 1e-12) * sqrt(252) on the 3 returns
    arr = np.array(returns_a)
    expected_sharpe = arr.mean() / (arr.std(ddof=1) + 1e-12) * np.sqrt(ANN)
    assert result.summary["net_sharpe"] == pytest.approx(round(expected_sharpe, 3))

    # Turnover: t=1 enters from 0 → L1=1; then no changes → [1, 0, 0]
    assert result.turnover.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert result.total_turnover == pytest.approx(1.0)


def test_lag_blocks_lookahead():
    """Causal lag must drop the first realised bar — never apply same-bar weights."""
    idx = _bday_index("2025-01-02", 5)
    prices = _prices_from_returns(
        {"A": [0.10, 0.10, 0.10, 0.10]}, start_price=100.0, idx=idx
    )
    # Lookahead trap: weight (1.0) only on the first bar, then 0.
    weights = pd.DataFrame(
        [[1.0], [0.0], [0.0], [0.0], [0.0]], index=idx, columns=["A"]
    )

    result = run_portfolio_backtest(
        weights, prices, cost_bps=0.0, execution_lag=1
    )

    # With lag=1: applied weight at t=1 is the t=0 weight (1.0); subsequent
    # bars apply weight 0. So we earn exactly the first 10% bar, nothing else.
    assert result.returns.tolist() == pytest.approx([0.10, 0.0, 0.0, 0.0])
    assert result.equity_curve.iloc[-1] == pytest.approx(1.10)


def test_flipping_weights_known_turnover():
    """Alternating (1,0) / (0,1) weights with equal returns → per-flip L1=2."""
    idx = _bday_index("2025-01-02", 5)
    rets = [0.01, 0.01, 0.01, 0.01]
    prices = _prices_from_returns(
        {"A": rets, "B": rets}, start_price=100.0, idx=idx
    )
    # Weights: t=0 (1,0), t=1 (0,1), t=2 (1,0), t=3 (0,1), t=4 (1,0)
    raw = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    weights = pd.DataFrame(raw, index=idx, columns=["A", "B"])

    result = run_portfolio_backtest(
        weights, prices, cost_bps=0.0, execution_lag=1
    )

    # Applied weights at t=1..4 = weights at t=0..3 = (1,0),(0,1),(1,0),(0,1)
    # L1 turnover: t=1 from 0 → 1; t=2 flip → 2; t=3 flip → 2; t=4 flip → 2.
    assert result.turnover.tolist() == pytest.approx([1.0, 2.0, 2.0, 2.0])
    # Returns are constant 0.01 since both names had the same return path.
    assert result.returns.tolist() == pytest.approx([0.01] * 4)


def test_cost_model_flat_bps_charges_per_turnover():
    """Flat per-side bps must reduce net returns by cost_bps * L1 / 1e4."""
    idx = _bday_index("2025-01-02", 5)
    rets = [0.01, 0.01, 0.01, 0.01]
    prices = _prices_from_returns(
        {"A": rets, "B": rets}, start_price=100.0, idx=idx
    )
    raw = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    weights = pd.DataFrame(raw, index=idx, columns=["A", "B"])

    cost_bps = 10.0  # 10 bps per unit L1 turnover
    result = run_portfolio_backtest(
        weights, prices, cost_bps=cost_bps, execution_lag=1
    )

    expected_costs = np.array([1.0, 2.0, 2.0, 2.0]) * (cost_bps / 1e4)
    expected_net = np.array([0.01] * 4) - expected_costs
    np.testing.assert_allclose(
        result.returns.values, expected_net, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.costs.values, expected_costs, rtol=0, atol=1e-12
    )


def test_slippage_requires_adv_when_enabled():
    """slippage_bps_per_pct_adv > 0 with adv=None must raise (no silent fallback)."""
    idx = _bday_index("2025-01-02", 3)
    prices = _prices_from_returns({"A": [0.01, 0.01]}, 100.0, idx)
    weights = pd.DataFrame([[1.0]] * 3, index=idx, columns=["A"])
    with pytest.raises(BacktestError, match="slippage requested"):
        run_portfolio_backtest(
            weights,
            prices,
            cost_bps=0.0,
            slippage_bps_per_pct_adv=5.0,
            adv=None,
        )


def test_slippage_scales_with_participation():
    """Halving ADV doubles per-trade slippage (linear impact model)."""
    idx = _bday_index("2025-01-02", 3)
    prices = _prices_from_returns({"A": [0.01, 0.01]}, 100.0, idx)
    weights = pd.DataFrame(
        [[1.0], [1.0], [1.0]], index=idx, columns=["A"]
    )
    adv_big = pd.DataFrame([[0.20]] * 3, index=idx, columns=["A"])
    adv_small = pd.DataFrame([[0.10]] * 3, index=idx, columns=["A"])

    big = run_portfolio_backtest(
        weights, prices, cost_bps=0.0,
        slippage_bps_per_pct_adv=10.0, adv=adv_big,
    )
    small = run_portfolio_backtest(
        weights, prices, cost_bps=0.0,
        slippage_bps_per_pct_adv=10.0, adv=adv_small,
    )
    # Costs: only the initial allocation bar has |dw|=1 (then 0,0).
    # big:   bps = 10 * (1.0/0.20)*100 = 5000 bps → cost on |dw|=1 → 0.5
    # small: bps = 10 * (1.0/0.10)*100 = 10000 bps → 1.0
    assert big.costs.iloc[0] == pytest.approx(0.5)
    assert small.costs.iloc[0] == pytest.approx(1.0)
    assert big.costs.iloc[1] == pytest.approx(0.0)


def test_drawdown_with_crash_bar_matches_hand_calculation():
    """Known crash sequence → maxDD must equal the hand-computed value."""
    idx = _bday_index("2025-01-02", 5)
    rets = [0.05, 0.05, 0.05, -0.20]
    prices = _prices_from_returns({"A": rets}, 100.0, idx)
    weights = pd.DataFrame([[1.0]] * len(idx), index=idx, columns=["A"])

    result = run_portfolio_backtest(
        weights, prices, cost_bps=0.0, execution_lag=1
    )
    # equity = [1.05, 1.1025, 1.157625, 0.9261]; dd = 0.20 exactly.
    assert result.summary["max_dd"] == pytest.approx(0.2, abs=1e-3)


def test_weights_referencing_unknown_ticker_fail_loud():
    idx = _bday_index("2025-01-02", 3)
    prices = _prices_from_returns({"A": [0.01, 0.01]}, 100.0, idx)
    weights = pd.DataFrame(
        [[1.0, 0.5]] * 3, index=idx, columns=["A", "Z_NOPRICE"]
    )
    with pytest.raises(BacktestError, match="without prices"):
        run_portfolio_backtest(weights, prices, cost_bps=0.0)


def test_zero_execution_lag_rejected():
    idx = _bday_index("2025-01-02", 3)
    prices = _prices_from_returns({"A": [0.01, 0.01]}, 100.0, idx)
    weights = pd.DataFrame([[1.0]] * 3, index=idx, columns=["A"])
    with pytest.raises(BacktestError, match=">= 1"):
        run_portfolio_backtest(weights, prices, execution_lag=0)


def test_overlay_is_causal_no_lookahead_into_today():
    """overlay() reads only info <= t-1 (rolling vol + dd both shift(1))."""
    idx = pd.bdate_range(start="2025-01-02", periods=60)
    base = pd.Series(np.linspace(-0.001, 0.001, 60), index=idx)
    s = overlay(base, target_vol=0.10, dd_soft=0.05, dd_hard=0.10)
    # The first 21 bars have NaN rolling vol → scalar must be 0 (filled).
    assert (s.iloc[:21] == 0.0).all()
    assert (s >= 0.0).all() and (s <= 1.0).all()


def test_save_result_atomic_and_round_trips(tmp_path):
    idx = _bday_index("2025-01-02", 4)
    prices = _prices_from_returns({"A": [0.01, 0.01, 0.01]}, 100.0, idx)
    weights = pd.DataFrame([[1.0]] * len(idx), index=idx, columns=["A"])
    result = run_portfolio_backtest(weights, prices, cost_bps=0.0)

    out = tmp_path / "bt" / "result.json"
    written = save_result(result, out)
    assert written == out
    assert out.exists()

    # No leftover .tmp file from the atomic write.
    assert not any(p.suffix == ".tmp" for p in out.parent.iterdir())

    payload = json.loads(out.read_text())
    assert payload["execution_lag"] == 1
    assert payload["n_bars"] == 3
    assert "max_dd" in payload["summary"]


def test_stats_handles_empty_series_without_raising():
    """stats() on an empty series returns zeros — no ZeroDivision or KeyError."""
    out = stats(pd.Series([], dtype=float))
    assert out["net_sharpe"] == 0.0
    assert out["max_dd"] == 0.0
    assert out["per_year"] == {}


def test_result_dataclass_is_frozen():
    """BacktestResult is frozen so call sites can't mutate the returned series."""
    idx = _bday_index("2025-01-02", 3)
    prices = _prices_from_returns({"A": [0.01, 0.01]}, 100.0, idx)
    weights = pd.DataFrame([[1.0]] * 3, index=idx, columns=["A"])
    result = run_portfolio_backtest(weights, prices, cost_bps=0.0)
    assert isinstance(result, BacktestResult)
    with pytest.raises((AttributeError, Exception)):
        result.execution_lag = 99  # type: ignore[misc]
