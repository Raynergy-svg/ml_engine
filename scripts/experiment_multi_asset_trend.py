#!/usr/bin/env python3
"""Run the PRE-REGISTERED diversified multi-asset trend test (offline; running:NO).

Design fixed in docs/experiment-multi-asset-trend-2026-06-29.md (before results).
Fetches the fixed 22-asset panel (yfinance, free), applies the ONE canonical trend rule
uniformly, HRP-combines + vol-targets into one portfolio, and reports the ship-gate
result full + OOS + sub-periods vs baselines (EW buy-hold, 60/40). Portfolio-level
judgment; per-asset Sharpes are transparency-only. Honest negative = success.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

logger = logging.getLogger("experiment_multi_asset_trend")

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG",
            "DBC", "USO", "UNG", "DBA", "GLD", "SLV", "VNQ", "UUP", "FXE", "FXY",
            "BTC-USD", "ETH-USD"]
CACHE = REPO_ROOT / "market_data" / "multi_asset" / "panel.parquet"
OUT = REPO_ROOT / "trained_data" / "backtests" / "multi_asset_trend_result.json"
OOS_FRACTION = 0.35


def fetch_panel() -> pd.DataFrame:
    if CACHE.exists():
        logger.info("loading cached panel %s", CACHE)
        return pd.read_parquet(CACHE)
    import yfinance as yf
    raw = yf.download(UNIVERSE, start="1990-01-01", progress=False, auto_adjust=True, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index, utc=True).normalize()
    close = close.reindex(columns=UNIVERSE).dropna(how="all")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        close.to_parquet(CACHE)
    except Exception:  # noqa: BLE001
        pass
    return close


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from src.equity.multi_asset_trend import (
        combined_portfolio, ew_buy_hold, gate_eval, per_asset_sharpes, sixty_forty,
    )
    px = fetch_panel()
    have = [c for c in UNIVERSE if c in px.columns and px[c].notna().sum() > 250]
    logger.info("panel: %d dates, %d/%d assets with data", len(px), len(have), len(UNIVERSE))
    px = px[have]

    net = combined_portfolio(px)
    if net.empty:
        logger.error("empty portfolio")
        return 1
    idx = net.index
    split = int(len(idx) * (1.0 - OOS_FRACTION))
    oos = idx[split:]
    ew = ew_buy_hold(px)
    sf = sixty_forty(px)

    def sub_blocks():
        out = []
        years = sorted(set(idx.year))
        for y0 in range(years[0], years[-1] + 1, 5):
            blk = idx[(idx.year >= y0) & (idx.year < y0 + 5)]
            if len(blk) >= 252:
                out.append({"start": str(blk.min().date()), "end": str(blk.max().date()),
                            **{"trend": gate_eval(net.loc[blk])["net_sharpe"]}})
        return out

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "running": "NO (offline pre-registered test)",
        "pre_registration": "docs/experiment-multi-asset-trend-2026-06-29.md",
        "universe": have, "n_assets": len(have),
        "window": {"start": str(idx.min().date()), "end": str(idx.max().date()), "n_days": len(idx)},
        "ship_gate_min_net_sharpe": 0.40,
        "diversified_trend": {
            "full_sample": gate_eval(net),
            "out_of_sample": {"start": str(oos.min().date()), "end": str(oos.max().date()),
                              "fraction": OOS_FRACTION, **gate_eval(net.loc[oos])},
            "sub_periods": sub_blocks(),
        },
        "baselines": {
            "ew_buy_hold_full": gate_eval(ew),
            "ew_buy_hold_oos": gate_eval(ew.loc[ew.index.intersection(oos)]),
            "sixty_forty_full": gate_eval(sf) if not sf.empty else None,
        },
        "per_asset_trend_sharpe_TRANSPARENCY_ONLY": per_asset_sharpes(px),
        "note": "PORTFOLIO-level judgment. Per-asset Sharpes are transparency, NOT the decision. "
                "ETF current-survivor selection at asset-class level (managed-futures proxy standard).",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
