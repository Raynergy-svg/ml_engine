#!/usr/bin/env python3
"""PRE-REGISTERED sleeve-combination tests (offline; running:NO).

Design fixed in docs/experiment-sleeve-combinations-2026-06-29.md (before results).
Builds three already-validated sleeve streams (carry / multi-asset trend / equity
harvester), HRP-combines the pre-registered pairs + 10% vol-target, and DECOMPOSES
return-vs-risk so a real RETURN edge is separated from another drawdown-reducer.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

logger = logging.getLogger("sleeve_combos")
OUT = REPO_ROOT / "trained_data" / "backtests" / "sleeve_combinations_result.json"
OOS_FRACTION = 0.35
CARRY_COST_BPS = 15.0


def _norm(s: pd.Series) -> pd.Series:
    s = pd.Series(s).dropna()
    s.index = pd.to_datetime(s.index, utc=True).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def carry_stream(refresh: bool = False) -> pd.Series:
    """G10+EM FX carry, crash-aware overlay, net of 15bps — reuses the validated pipeline."""
    spec = importlib.util.spec_from_file_location(
        "em_carry", REPO_ROOT / "scripts" / "experiment_em_carry.py")
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    from src.factor.portfolio import (
        apply_no_trade_band, enforce_guards, target_weights, weekly_rebalance_mask)
    from src.factor.signals import carry_signal, combine_signals

    close = em.load_prices(em.GLOBAL, refresh)
    rd = em.build_rate_diff(close.index, em.GLOBAL, refresh)
    returns = close.pct_change()
    accrual = rd / 100.0 / 252.0
    raw_w = target_weights(combine_signals([carry_signal(rd)]), returns)
    held = enforce_guards(apply_no_trade_band(
        raw_w, band=0.05, rebalance_mask=weekly_rebalance_mask(close.index)))
    held = em.crash_aware_overlay(held, em._gross_daily(held, returns, accrual, list(close.columns)))
    gross = em._gross_daily(held, returns, accrual, list(close.columns))
    dw = held.shift(1).fillna(0.0).diff().abs().fillna(held.shift(1).abs())
    cost = dw.sum(axis=1) * (CARRY_COST_BPS / 1e4)
    return _norm(gross - cost)


def trend_stream() -> pd.Series:
    from src.equity.multi_asset_trend import combined_portfolio
    px = pd.read_parquet(REPO_ROOT / "market_data" / "multi_asset" / "panel.parquet")
    have = [c for c in px.columns if px[c].notna().sum() > 250]
    return _norm(combined_portfolio(px[have]))


def harvester_stream() -> pd.Series:
    from src.equity.ship_gate import build_equal_weight_panel
    from src.equity.universe import UniverseSnapshot
    from src.equity.variant_eval import book_return_series
    snap = UniverseSnapshot.load(REPO_ROOT / "market_data" / "equity" / "universe_snapshot_pit.json")
    prices = pd.read_parquet(REPO_ROOT / "market_data" / "equity" / "sp500_prices.parquet")
    members = [c for c in snap.membership.columns if c in prices.columns]
    px = prices[members]
    ew = build_equal_weight_panel(snap, px)
    return _norm(book_return_series(snap, px, ew))


def hrp_combine(streams: dict) -> pd.Series:
    """HRP across the sleeve return matrix + 10% vol-target (the FIXED combiner)."""
    from src.equity.backtest import overlay
    from src.equity.ship_gate import DEFAULT_DD_HARD, DEFAULT_DD_SOFT
    from src.equity.sleeve_combiner import combine_sleeves
    df = pd.DataFrame(streams).dropna()
    if df.shape[1] < 2 or len(df) < 252:
        return pd.Series(dtype=float)
    combined, _w = combine_sleeves(df)
    combined = combined.dropna()
    scalar = overlay(combined.fillna(0.0), target_vol=0.10,
                     dd_soft=DEFAULT_DD_SOFT, dd_hard=DEFAULT_DD_HARD, max_lev=3.0)
    return (combined * scalar.reindex(combined.index).fillna(0.0)).dropna()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from src.equity.multi_asset_trend import decompose, ew_buy_hold

    logger.info("building sleeve streams ...")
    streams = {}
    for name, fn in (("trend", trend_stream), ("harvester", harvester_stream), ("carry", carry_stream)):
        try:
            s = fn()
            if not s.empty:
                streams[name] = s
                logger.info("  %-9s %d days %s..%s", name, len(s), s.index.min().date(), s.index.max().date())
        except Exception as exc:  # noqa: BLE001
            logger.error("  %s stream FAILED: %s", name, exc)

    px = pd.read_parquet(REPO_ROOT / "market_data" / "multi_asset" / "panel.parquet")
    have = [c for c in px.columns if px[c].notna().sum() > 250]
    bh = _norm(ew_buy_hold(px[have]))

    def block(net):
        if net.empty:
            return None
        idx = net.index
        oos = idx[int(len(idx) * (1 - OOS_FRACTION)):]
        return {"full": decompose(net), "oos": {"start": str(oos.min().date()), "end": str(oos.max().date()),
                                                **decompose(net.loc[oos])}}

    combos = {}
    pairs = [("carry+trend", ("carry", "trend")), ("trend+harvester", ("trend", "harvester"))]
    for cname, members in pairs:
        if all(m in streams for m in members):
            sub = {m: streams[m] for m in members}
            common = pd.DataFrame(sub).dropna().index
            combined = hrp_combine(sub)
            combos[cname] = {
                "members": list(members),
                "common_window": {"start": str(common.min().date()), "end": str(common.max().date()), "n": len(common)},
                "combined": block(combined),
                "constituents": {m: block(streams[m].loc[streams[m].index.isin(common)]) for m in members},
            }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "running": "NO (offline pre-registered test)",
        "pre_registration": "docs/experiment-sleeve-combinations-2026-06-29.md",
        "ship_gate_min_net_sharpe": 0.40,
        "metric_note": "ann_return = RETURN side; ann_vol/max_dd = RISK side. Combo Sharpe > BOTH "
                       "constituents AND buy-hold by a real OOS margin = RETURN edge; Sharpe up only "
                       "via lower vol/DD (ann_return not up) = drawdown-only (the known finding, not alpha).",
        "buy_and_hold_baseline": block(bh),
        "combinations": combos,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
