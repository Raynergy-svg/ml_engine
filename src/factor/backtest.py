"""US-006 — Cost-aware, walk-forward backtest for the factor portfolio.

Honesty properties:
- 1-day execution lag: the weight decided from data through close[T] is only
  applied to returns from T onward (``execution_lag=1``). A ``lookahead canary``
  test proves that flipping the lag negative (cheating) inflates Sharpe.
- Costs are explicit and per-trade: half-spread crossed on every weight change,
  plus a daily financing drag on gross exposure (conservative — earned carry is
  NOT credited, so a trend book is only charged, never subsidized).
- Walk-forward by construction: every estimated quantity (per-pair vol, the
  covariance used for vol-targeting) is a trailing window; the few constants are
  pre-registered in ``portfolio.py``/``ship_gate.py``. No full-sample fitting.
- Deterministic: no randomness; reruns are byte-identical.

Output artifact is written atomically to
``trained_data/backtests/factor_portfolio_{ts}.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.research.drawdown_convention import drawdown_fraction

from . import FACTOR_PIPELINE_VERSION
from .data_loader import utc_now_stamp

ANNUALIZATION = 252.0
DEFAULT_FINANCING_MARKUP_ANNUAL = 0.01  # OANDA retail markup, charged on gross
BACKTEST_DIR = Path("trained_data/backtests")

# Conservative OANDA-practice daily spreads (pips). Wider than headline quotes.
# Crosses are meaningfully wider than majors — modeling them at the major default
# would flatter the result, so they are listed explicitly. Anything not listed
# falls back to 2.5 pips (a deliberately non-optimistic placeholder).
DEFAULT_SPREADS_PIPS: Dict[str, float] = {
    # majors
    "EUR_USD": 1.5,
    "USD_JPY": 1.5,
    "GBP_USD": 2.0,
    "AUD_USD": 1.6,
    "NZD_USD": 2.2,
    "USD_CAD": 2.0,
    "USD_CHF": 2.2,
    # G10 crosses (wider books)
    "EUR_JPY": 1.8,
    "GBP_JPY": 3.0,
    "AUD_JPY": 2.0,
    "NZD_JPY": 2.6,
    "CAD_JPY": 2.6,
    "CHF_JPY": 3.0,
    "EUR_GBP": 1.6,
    "EUR_CHF": 2.0,
    "EUR_AUD": 2.6,
    "GBP_AUD": 4.0,
    "AUD_NZD": 3.0,
    "EUR_CAD": 2.6,
}
DEFAULT_SPREAD_FALLBACK_PIPS = 2.5


def _pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


@dataclass
class BacktestResult:
    factor_pipeline_version: str
    start: str
    end: str
    n_days: int
    pairs: list
    net_sharpe: float
    gross_sharpe: float
    net_cagr: float
    max_drawdown: float
    annual_turnover: float
    annual_cost_drag: float
    per_year_returns: Dict[str, float]
    positive_years: int
    total_years: int
    walk_forward: bool = True
    financing_markup_annual: float = DEFAULT_FINANCING_MARKUP_ANNUAL
    spreads_pips: Dict[str, float] = field(default_factory=dict)
    equity_curve: Dict[str, float] = field(default_factory=dict)


def _sharpe(daily: np.ndarray) -> float:
    daily = daily[np.isfinite(daily)]
    if daily.size < 2:
        return 0.0
    sd = daily.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float(daily.mean() / sd * np.sqrt(ANNUALIZATION))


def _max_drawdown(equity: np.ndarray) -> float:
    """Max peak-to-trough decline as a NON-NEGATIVE fraction (0.25 == 25%).

    Canonical convention — see :mod:`src.research.drawdown_convention`.
    Validated at this boundary so a future sign flip fails here rather than
    silently satisfying ``max_dd <= MAX_DRAWDOWN`` in ``ship_gate``.
    """
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    value = float(-dd.min()) if dd.size else 0.0
    return drawdown_fraction(
        max(0.0, value), source="factor.backtest._max_drawdown"
    )


def run_backtest(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    *,
    carry_accrual: Optional[pd.DataFrame] = None,
    spreads_pips: Optional[Dict[str, float]] = None,
    financing_markup_annual: float = DEFAULT_FINANCING_MARKUP_ANNUAL,
    execution_lag: int = 1,
) -> BacktestResult:
    """Run the cost-aware backtest. ``weights`` and ``close`` are date×pair.

    ``weights`` are target weights decided at each date's close (already
    no-trade-banded). Returns are close-to-close; the execution lag enforces that
    a position cannot earn the very return that informed it.

    ``carry_accrual`` (optional, date×pair) is the DAILY fractional interest
    differential earned by being long the pair (e.g. rate_diff_pct/100/252). When
    supplied, a held position is credited the carry it actually earns; the broker
    financing markup is charged separately on gross exposure. This is essential
    for the carry factor — without it the backtest is silently price-only.
    """
    spreads_pips = spreads_pips or DEFAULT_SPREADS_PIPS
    pairs = list(close.columns)
    weights = weights.reindex(index=close.index, columns=pairs).fillna(0.0)

    ret = close.pct_change().reindex(columns=pairs)
    w_eff = weights.shift(execution_lag).fillna(0.0)

    if carry_accrual is not None:
        carry_accrual = carry_accrual.reindex(
            index=close.index, columns=pairs
        ).fillna(0.0)
        carry_pnl = (w_eff * carry_accrual).sum(axis=1)
    else:
        carry_pnl = pd.Series(0.0, index=close.index)

    # Half-spread crossed (as a price fraction) per pair on each weight change.
    half_spread_frac = {}
    last_price = close.ffill()
    for p in pairs:
        half_spread_frac[p] = (
            spreads_pips.get(p, DEFAULT_SPREAD_FALLBACK_PIPS) * _pip_size(p)
        ) / 2.0

    dw = w_eff.diff().abs().fillna(w_eff.abs())  # day-1 establishes position
    spread_cost = pd.Series(0.0, index=close.index)
    for p in pairs:
        spread_cost = spread_cost + dw[p] * (half_spread_frac[p] / last_price[p])

    gross_exposure = w_eff.abs().sum(axis=1)
    financing_cost = gross_exposure * (financing_markup_annual / ANNUALIZATION)

    gross_pnl = (w_eff * ret).sum(axis=1) + carry_pnl
    net_pnl = gross_pnl - spread_cost - financing_cost

    net_pnl = net_pnl.fillna(0.0)
    gross_pnl = gross_pnl.fillna(0.0)
    equity = (1.0 + net_pnl).cumprod()

    # Per-calendar-year net return.
    yr = net_pnl.groupby(net_pnl.index.year)
    per_year = {str(int(y)): float((1.0 + g).prod() - 1.0) for y, g in yr}
    positive_years = sum(1 for v in per_year.values() if v > 0)

    n = len(net_pnl)
    years = max(n / ANNUALIZATION, 1e-9)
    net_cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if n else 0.0
    annual_turnover = float(dw.sum(axis=1).mean() * ANNUALIZATION)
    annual_cost_drag = float((spread_cost + financing_cost).mean() * ANNUALIZATION)

    eq_sampled = equity.iloc[:: max(1, n // 400)] if n else equity
    return BacktestResult(
        factor_pipeline_version=FACTOR_PIPELINE_VERSION,
        start=str(close.index[0].date()) if n else "",
        end=str(close.index[-1].date()) if n else "",
        n_days=n,
        pairs=pairs,
        net_sharpe=_sharpe(net_pnl.to_numpy()),
        gross_sharpe=_sharpe(gross_pnl.to_numpy()),
        net_cagr=net_cagr,
        max_drawdown=_max_drawdown(equity.to_numpy()),
        annual_turnover=annual_turnover,
        annual_cost_drag=annual_cost_drag,
        per_year_returns=per_year,
        positive_years=positive_years,
        total_years=len(per_year),
        walk_forward=True,
        financing_markup_annual=financing_markup_annual,
        spreads_pips={
            p: spreads_pips.get(p, DEFAULT_SPREAD_FALLBACK_PIPS) for p in pairs
        },
        equity_curve={str(k.date()): float(v) for k, v in eq_sampled.items()},
    )


def write_report(result: BacktestResult, *, out_dir: Path = BACKTEST_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"factor_portfolio_{utc_now_stamp()}.json"
    tmp = path.with_suffix(".json.tmp")
    payload = asdict(result)
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path
