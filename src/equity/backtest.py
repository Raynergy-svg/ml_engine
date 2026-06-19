"""US-003 — Portfolio-weight backtest harness for the equity-beta harvester.

Takes a date x ticker target-weight panel and an aligned close-price panel,
applies a strictly causal execution lag (>= 1 bar), charges a minimal per-name
cost model (flat bps + ADV-scaled slippage), and returns the equity curve,
net-of-cost returns, turnover, and the standard summary stats (net Sharpe,
maxDD, CAGR, positive-year ratio) the harvester ship gate (US-005) is graded
against.

`stats()` and `overlay()` are the same helpers used by
``scripts/build_equity_harvester.py`` — the script imports them from here so
both the experimental research script and the runtime backtest share one
canonical implementation.

The depth-aware execution book (US-014) is out of scope here — that lives in
the execution path, not the backtest path. Slippage here is a simple linear
participation impact model parameterised by basis-points per percentage-of-ADV
traded.

Hard rules:

* Strictly causal: ``execution_lag >= 1``; the weight chosen at end-of-day
  ``t`` is applied to returns at ``t + execution_lag``.
* No mocks: this module uses real pandas/numpy on real (tmp_path) disk.
* Atomic state writes: ``save_result`` writes to ``.tmp`` then ``os.replace``.
* Fail-loud: shape/index/lag/NaN violations raise ``BacktestError`` — no silent
  forward-fill, no zero-imputation that hides upstream data gaps.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANN = 252
DEFAULT_COST_BPS = 2.0
DEFAULT_SLIPPAGE_BPS_PER_PCT_ADV = 0.0
DEFAULT_EXECUTION_LAG = 1


class BacktestError(RuntimeError):
    """Raised when backtest inputs are mis-shaped, mis-aligned, or NaN-bearing."""


def stats(pnl: pd.Series) -> Dict[str, object]:
    """Annualised summary stats for a per-bar return series.

    Identical to the helper previously defined in
    ``scripts/build_equity_harvester.py`` (the script now imports from here).
    Returns net Sharpe, maxDD, CAGR, positive-years ratio and per-year returns.
    """
    pnl = pnl.dropna()
    if pnl.empty:
        return {
            "net_sharpe": 0.0,
            "max_dd": 0.0,
            "cagr": 0.0,
            "pos_years": "0/0",
            "per_year": {},
        }
    eq = (1 + pnl).cumprod()
    peak = eq.cummax()
    dd = float(((peak - eq) / peak).max())
    sharpe = float(pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(ANN))
    cagr = float(eq.iloc[-1] ** (ANN / len(pnl)) - 1)
    per_year = {
        str(y): round(float((1 + g).prod() - 1), 4)
        for y, g in pnl.groupby(pnl.index.year)
    }
    pos = sum(v > 0 for v in per_year.values())
    return {
        "net_sharpe": round(sharpe, 3),
        "max_dd": round(dd, 3),
        "cagr": round(cagr, 4),
        "pos_years": f"{pos}/{len(per_year)}",
        "per_year": per_year,
    }


def overlay(
    base_ret: pd.Series,
    *,
    target_vol: float,
    dd_soft: float,
    dd_hard: float,
    max_lev: float = 1.0,
) -> pd.Series:
    """Causal vol-target + drawdown circuit-breaker exposure scalar.

    Uses only information <= ``t-1`` (``shift(1)`` on rolling vol and on the
    base book's drawdown) so the scalar at ``t`` cannot peek at ``t``'s return.
    Returns a series in ``[0, max_lev]`` aligned to ``base_ret.index``.
    """
    rvol = base_ret.rolling(21).std().mul(np.sqrt(ANN)).shift(1)
    s_vol = (target_vol / rvol).clip(upper=max_lev).fillna(0.0)
    eq = (1 + base_ret).cumprod()
    dd = (1 - eq / eq.cummax()).shift(1).fillna(0.0)
    s_dd = np.where(
        dd <= dd_soft,
        1.0,
        np.where(
            dd >= dd_hard,
            0.0,
            (dd_hard - dd) / (dd_hard - dd_soft),
        ),
    )
    return (s_vol * pd.Series(s_dd, index=base_ret.index)).clip(0.0, max_lev)


@dataclass(frozen=True)
class BacktestResult:
    """Output of :func:`run_portfolio_backtest`.

    All series share the post-lag return index (rows where a return can be
    realised given the requested ``execution_lag``).
    """

    equity_curve: pd.Series
    returns: pd.Series          # net of cost
    gross_returns: pd.Series    # before cost
    turnover: pd.Series         # per-bar L1 weight delta (post-lag)
    costs: pd.Series            # per-bar cost in return units
    summary: Dict[str, object] = field(default_factory=dict)
    avg_turnover: float = 0.0
    total_turnover: float = 0.0
    avg_cost: float = 0.0
    execution_lag: int = DEFAULT_EXECUTION_LAG
    span: str = ""

    def to_json_payload(self) -> Dict[str, object]:
        return {
            "summary": self.summary,
            "avg_turnover": self.avg_turnover,
            "total_turnover": self.total_turnover,
            "avg_cost": self.avg_cost,
            "execution_lag": self.execution_lag,
            "span": self.span,
            "equity_curve_last": float(self.equity_curve.iloc[-1])
            if len(self.equity_curve)
            else 1.0,
            "n_bars": int(len(self.returns)),
        }


def _coerce_panel(
    df: Union[pd.DataFrame, pd.Series],
    name: str,
) -> pd.DataFrame:
    """Coerce a series/frame to a DataFrame with a sorted DatetimeIndex."""
    if isinstance(df, pd.Series):
        if df.name is None:
            raise BacktestError(
                f"{name}: Series must have a .name (used as ticker column)"
            )
        df = df.to_frame(df.name)
    if not isinstance(df, pd.DataFrame):
        raise BacktestError(f"{name}: expected DataFrame, got {type(df)!r}")
    if df.empty:
        raise BacktestError(f"{name}: empty frame")
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.DatetimeIndex(df.index)
        except (TypeError, ValueError) as exc:
            raise BacktestError(
                f"{name}: index is not date-like ({exc})"
            ) from exc
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _align_weights_prices(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align weights to the price index and union of tickers.

    Missing weight cells become 0 (no position). Missing price cells raise —
    we never silently forward-fill prices for a name we intended to trade.
    """
    if prices.isna().any().any():
        # Allow NaN only in rows that have at least one priced ticker; require
        # that every ticker we'll hold has a price on the holding bar.
        bad_cols = prices.columns[prices.isna().any()].tolist()
        if bad_cols:
            raise BacktestError(
                f"prices: NaNs found in columns {bad_cols} — "
                "refusing to silently forward-fill prices in the backtest"
            )

    all_tickers = sorted(set(weights.columns) | set(prices.columns))
    prices_aligned = prices.reindex(columns=all_tickers)
    # Any ticker the user weights but has no price must fail loud, not 0-fill.
    weighted_tickers = [
        c for c in weights.columns if (weights[c].abs().sum() > 0)
    ]
    missing = [t for t in weighted_tickers if t not in prices.columns]
    if missing:
        raise BacktestError(
            f"weights reference tickers without prices: {missing}"
        )

    weights_aligned = weights.reindex(
        index=prices_aligned.index, columns=all_tickers
    ).fillna(0.0)
    return weights_aligned, prices_aligned


def _compute_costs(
    delta_w: pd.DataFrame,
    *,
    cost_bps: float,
    slippage_bps_per_pct_adv: float,
    adv: Optional[pd.DataFrame],
) -> pd.Series:
    """Per-bar cost in return units.

    * Flat per-side bps: ``cost_bps`` charged against ``sum_i |dw_i|``.
    * Linear ADV-impact slippage: each name's bps cost is
      ``slippage_bps_per_pct_adv * participation_pct``, where participation is
      ``|dw_i| / adv_i`` (ADV expressed as fraction-of-NAV). Cost is then
      applied to the traded notional ``|dw_i|``.
    """
    abs_dw = delta_w.abs()
    flat = (cost_bps / 1e4) * abs_dw.sum(axis=1)

    if slippage_bps_per_pct_adv == 0.0 or adv is None:
        return flat

    adv_aligned = adv.reindex(
        index=delta_w.index, columns=delta_w.columns
    )
    # Cells with missing or non-positive ADV: treat as "no liquidity info" and
    # fail loud if a trade actually wants to size into them — we won't silently
    # let an aggressive trade pay zero slippage because ADV was NaN.
    needs_adv = abs_dw > 0
    if (needs_adv & (adv_aligned.isna() | (adv_aligned <= 0))).any().any():
        raise BacktestError(
            "slippage requested but ADV is NaN or <= 0 for at least one "
            "non-zero trade — provide a positive ADV (fraction-of-NAV) for "
            "every name that trades"
        )

    safe_adv = adv_aligned.where(adv_aligned > 0, other=np.nan)
    # participation as percent (×100)
    participation_pct = (abs_dw / safe_adv) * 100.0
    per_name_bps = slippage_bps_per_pct_adv * participation_pct
    per_name_cost = (per_name_bps / 1e4) * abs_dw
    slip = per_name_cost.sum(axis=1).fillna(0.0)
    return flat + slip


def run_portfolio_backtest(
    weights: Union[pd.DataFrame, pd.Series],
    prices: pd.DataFrame,
    *,
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps_per_pct_adv: float = DEFAULT_SLIPPAGE_BPS_PER_PCT_ADV,
    adv: Optional[pd.DataFrame] = None,
    execution_lag: int = DEFAULT_EXECUTION_LAG,
) -> BacktestResult:
    """Run a causal portfolio backtest given target weights and prices.

    Parameters
    ----------
    weights:
        Date x ticker target weights. Rows are dated by the close that
        produced them; with ``execution_lag = k`` the row at ``t`` is applied
        to returns at ``t + k``.
    prices:
        Date x ticker close prices (already split/div adjusted — use
        ``src.equity.data_loader.load_or_fetch``).
    cost_bps:
        Flat per-side basis-points cost charged on per-bar L1 turnover.
    slippage_bps_per_pct_adv:
        Linear impact: bps per 1.0 percentage point of ADV participation.
        Set to 0 to disable (and ``adv`` may then be ``None``).
    adv:
        Optional date x ticker average daily volume expressed as
        **fraction of NAV** (so 0.05 means a name's ADV equals 5% of NAV).
        Required when ``slippage_bps_per_pct_adv > 0``.
    execution_lag:
        Bars between weight decision and execution. Must be >= 1; default 1
        is the standard "trade on next bar" causal contract.

    Returns
    -------
    BacktestResult
        Equity curve, net/gross returns, turnover, costs, and summary stats.
    """
    if execution_lag < 1:
        raise BacktestError(
            f"execution_lag must be >= 1 (causal), got {execution_lag}"
        )
    if cost_bps < 0:
        raise BacktestError(f"cost_bps must be >= 0, got {cost_bps}")
    if slippage_bps_per_pct_adv < 0:
        raise BacktestError(
            f"slippage_bps_per_pct_adv must be >= 0, got {slippage_bps_per_pct_adv}"
        )
    if slippage_bps_per_pct_adv > 0 and adv is None:
        raise BacktestError(
            "slippage requested (slippage_bps_per_pct_adv > 0) but adv is None"
        )

    weights = _coerce_panel(weights, "weights")
    prices = _coerce_panel(prices, "prices")
    if adv is not None:
        adv = _coerce_panel(adv, "adv")

    weights, prices = _align_weights_prices(weights, prices)

    if len(prices) < execution_lag + 1:
        raise BacktestError(
            f"prices: need at least execution_lag + 1 = {execution_lag + 1} "
            f"rows to realise a return, got {len(prices)}"
        )

    # Per-name returns aligned to the price index (first row is NaN).
    asset_ret = prices.pct_change()
    # Applied weights at t: the weight chosen at t - execution_lag (else 0).
    applied = weights.shift(execution_lag).fillna(0.0)
    # Delta of applied weights = realised turnover at the execution bar.
    delta_w = applied.diff().fillna(applied)

    # Restrict to bars where a return is actually realised.
    realised_idx = prices.index[execution_lag:]
    asset_ret = asset_ret.loc[realised_idx]
    applied = applied.loc[realised_idx]
    delta_w = delta_w.loc[realised_idx]

    if asset_ret.isna().any().any():
        bad = asset_ret.columns[asset_ret.isna().any()].tolist()
        raise BacktestError(
            f"prices: returns contain NaN for tickers {bad} on the realised "
            "window — fix the input price panel"
        )

    gross = (applied * asset_ret).sum(axis=1)
    costs = _compute_costs(
        delta_w,
        cost_bps=cost_bps,
        slippage_bps_per_pct_adv=slippage_bps_per_pct_adv,
        adv=adv,
    )
    net = gross - costs
    eq = (1 + net).cumprod()
    turnover = delta_w.abs().sum(axis=1)

    summary = stats(net)
    span = f"{realised_idx.min().date()}->{realised_idx.max().date()}"

    return BacktestResult(
        equity_curve=eq,
        returns=net,
        gross_returns=gross,
        turnover=turnover,
        costs=costs,
        summary=summary,
        avg_turnover=float(turnover.mean()),
        total_turnover=float(turnover.sum()),
        avg_cost=float(costs.mean()),
        execution_lag=execution_lag,
        span=span,
    )


def save_result(result: BacktestResult, path: Path) -> Path:
    """Atomically persist the result summary to ``path`` (``.tmp`` + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_json_payload()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


__all__ = [
    "ANN",
    "BacktestError",
    "BacktestResult",
    "DEFAULT_COST_BPS",
    "DEFAULT_EXECUTION_LAG",
    "DEFAULT_SLIPPAGE_BPS_PER_PCT_ADV",
    "overlay",
    "run_portfolio_backtest",
    "save_result",
    "stats",
]
