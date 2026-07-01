"""Sleeve combiner — HRP across sleeve RETURN streams + combined-book gate.

Combines independent sleeve net-return series into one book by risk-budgeting
across them (HRP applied to the sleeve return matrix; for 2 sleeves HRP reduces
to inverse-variance). Strictly causal: sleeve weights at a rebalance use a
trailing window through ``t-1`` and are held until the next rebalance. The
combined return series is scored by the SAME ship gate (`evaluate_gate`) — a
sleeve that passes alone but wrecks the combined drawdown does not clear the book.

This is the legitimate "multiple frames": combining weakly-correlated
positive-expectancy STREAMS, not correlated re-slicings of one stream. OFFLINE
evaluation, NOT a live loop (running:NO). NON-hot-path; baseline NOT modified.

PRE-REGISTERED (anti-p-hacking, fixed before results): HRP-across-sleeves,
lookback 252d, monthly (21d) rebalance. The Phase-0 test is a SINGLE pre-registered
comparison ({harvester + trend} vs harvester-alone, full + OOS) — no iterate-to-pass.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK = 252
DEFAULT_STEP = 21
SLEEVE_BAKEOFF_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE_book.json"


def _risk_budget(window: pd.DataFrame) -> pd.Series:
    """Risk-budget weights across sleeve RETURN streams (HRP).

    For <=2 sleeves HRP reduces exactly to inverse-variance, so we compute it
    directly (and robustly: a zero/NaN-variance window falls back to equal
    weight). For >=3 sleeves we use the full HRP clustering, with the same
    inverse-variance fallback if the correlation matrix is degenerate.
    """
    import numpy as np

    cols = list(window.columns)
    n = len(cols)
    var = window.var().replace(0.0, np.nan)
    if var.isna().all() or n == 0:
        return pd.Series(1.0 / max(n, 1), index=cols)
    if n <= 2:
        iv = (1.0 / var).fillna(0.0)
        return iv / iv.sum() if iv.sum() > 0 else pd.Series(1.0 / n, index=cols)
    try:
        from src.equity.hrp import _hrp_from_returns

        clean = window.dropna(axis=1, how="any")
        clean = clean.loc[:, clean.var() > 0]
        if clean.shape[1] >= 3:
            w = _hrp_from_returns(clean).reindex(cols).fillna(0.0)
            return w / w.sum() if w.sum() > 0 else pd.Series(1.0 / n, index=cols)
    except ValueError:
        pass
    iv = (1.0 / var).fillna(0.0)
    return iv / iv.sum() if iv.sum() > 0 else pd.Series(1.0 / n, index=cols)


def combine_sleeves(
    returns_df: pd.DataFrame,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    step: int = DEFAULT_STEP,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Combine sleeve return streams; return (combined_returns, sleeve_weight_panel).

    Weights are HRP across the sleeve return matrix (inverse-variance for 2),
    re-estimated every ``step`` from the trailing ``lookback`` window through
    ``t-1`` (causal) and held between rebalances. Pre-window rows use equal weight.
    """
    idx = returns_df.index
    sleeves = list(returns_df.columns)
    n = len(sleeves)
    if n == 0:
        raise ValueError("returns_df must have >= 1 sleeve column")
    wp = pd.DataFrame(0.0, index=idx, columns=sleeves)
    last: Optional[pd.Series] = None
    for i in range(len(idx)):
        if i > 0 and (last is None or i % int(step) == 0):
            window = returns_df.iloc[max(0, i - int(lookback)): i].dropna()
            if len(window) >= max(20, int(lookback) // 4):
                last = _risk_budget(window[sleeves]).reindex(sleeves).fillna(0.0)
        if last is not None:
            wp.iloc[i] = last.values
    zero_rows = wp.sum(axis=1) <= 0
    if zero_rows.any():
        wp.loc[zero_rows] = 1.0 / n
    combined = (wp * returns_df.fillna(0.0)).sum(axis=1)
    return combined, wp


def gate_combined(returns: pd.Series) -> Dict[str, object]:
    """Score a return series through the ship gate (combined-book gate)."""
    from src.equity.backtest import stats
    from src.factor.ship_gate import evaluate_gate

    s = stats(returns.dropna())
    per_year = s.get("per_year", {}) or {}
    pos = sum(1 for v in per_year.values() if float(v) > 0)
    tot = len(per_year)
    net_sharpe = float(s.get("net_sharpe", 0.0))
    max_dd = float(s.get("max_dd", 1.0))
    verdict = evaluate_gate(
        {
            "net_sharpe": net_sharpe,
            "positive_years": pos,
            "total_years": tot,
            "max_drawdown": max_dd,
            "walk_forward": True,
        }
    )
    return {
        "net_sharpe": round(net_sharpe, 4),
        "max_dd": round(max_dd, 4),
        "positive_years": int(pos),
        "total_years": int(tot),
        "gate_pass": bool(verdict.passed),
    }


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise


def run_sleeve_bakeoff(
    *,
    snapshot_path: Path = Path("market_data/equity/universe_snapshot_pit.json"),
    out_path: Path = Path(SLEEVE_BAKEOFF_PATH_DEFAULT),
    oos_fraction: float = 0.34,
    generated_at: str = "",
) -> Dict[str, object]:
    """OFFLINE Phase-0 test: does {harvester + trend} beat harvester-alone on the
    gate, full-sample AND out-of-sample? running:NO — backtest only."""
    import yfinance as yf

    from src.equity.ship_gate import build_equal_weight_panel
    from src.equity.trend_sleeve import TREND_ETFS, trend_sleeve_returns
    from src.equity.universe import UniverseSnapshot
    from src.equity.variant_eval import book_return_series

    snap = UniverseSnapshot.load(Path(snapshot_path))
    members = list(snap.membership.columns)
    idx = snap.membership.index
    start = idx.min().date().isoformat()
    end = (idx.max() + pd.Timedelta(days=1)).date().isoformat()
    syms = list(dict.fromkeys(members + list(TREND_ETFS)))
    raw = yf.download(syms, start=start, end=end, progress=False, auto_adjust=True, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index, utc=True).normalize()

    stock_px = close.reindex(columns=members).reindex(index=idx).ffill()
    etf_px = close.reindex(columns=list(TREND_ETFS)).dropna(how="all")

    harv_ret = book_return_series(snap, stock_px, build_equal_weight_panel(snap, stock_px))
    trend_ret = trend_sleeve_returns(etf_px)

    R = pd.concat({"harvester": harv_ret, "trend": trend_ret}, axis=1).dropna()
    combined, sleeve_w = combine_sleeves(R)

    def _compare(R_win: pd.DataFrame, combined_win: pd.Series) -> dict:
        harv_alone = gate_combined(R_win["harvester"])
        comb = gate_combined(combined_win)
        return {
            "harvester_alone": harv_alone,
            "combined_book": comb,
            "trend_alone": gate_combined(R_win["trend"]),
            "combined_beats_harvester": (
                float(comb["net_sharpe"]) > float(harv_alone["net_sharpe"])
            ),
            "net_sharpe_margin": round(
                float(comb["net_sharpe"]) - float(harv_alone["net_sharpe"]), 4
            ),
            "avg_sleeve_corr": round(float(R_win.corr().iloc[0, 1]), 4),
        }

    cidx = R.index
    split = int(len(cidx) * (1.0 - float(oos_fraction)))
    oos = cidx[split:]

    # Raised bar (anti-p-hacking): stability across disjoint ~5yr blocks the
    # single pre-registered rule never tuned to — a real edge should persist, a
    # dredged one shows up in one regime only.
    sub_periods = []
    years = sorted(set(cidx.year))
    for y0 in range(years[0], years[-1] + 1, 5):
        blk = cidx[(cidx.year >= y0) & (cidx.year < y0 + 5)]
        if len(blk) < 252:
            continue
        h = gate_combined(R.loc[blk, "harvester"])
        c = gate_combined(combined.loc[blk])
        sub_periods.append({
            "start": blk.min().date().isoformat(),
            "end": blk.max().date().isoformat(),
            "harvester_net_sharpe": h["net_sharpe"],
            "combined_net_sharpe": c["net_sharpe"],
            "combined_maxdd": c["max_dd"],
            "margin": round(float(c["net_sharpe"]) - float(h["net_sharpe"]), 4),
            "combined_beats": float(c["net_sharpe"]) > float(h["net_sharpe"]),
        })

    payload = {
        "generated_at": generated_at,
        "running": "NO (offline backtest evaluation)",
        "pre_registered": {
            "trend_etfs": list(TREND_ETFS),
            "trend_rule": "price > 200d SMA (shift1), EW among on-assets else cash, monthly",
            "combiner": "HRP-across-sleeve-returns (inverse-variance for 2), 252d/21d",
            "test": "combined {harvester+trend} vs harvester-alone net Sharpe, full + OOS",
        },
        "common_window": {"start": cidx.min().date().isoformat(), "end": cidx.max().date().isoformat()},
        "sub_period_stability": sub_periods,
        "full_sample": _compare(R, combined),
        "out_of_sample": {
            "start": oos.min().date().isoformat(),
            "end": oos.max().date().isoformat(),
            "fraction": float(oos_fraction),
            **_compare(R.loc[oos], combined.loc[oos]),
        },
    }
    _atomic_write_json(payload, Path(out_path))
    return payload
