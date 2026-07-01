#!/usr/bin/env python3
"""Run the PRE-REGISTERED OANDA position-book contrarian sentiment test (offline; running:NO).

Design is fixed in docs/experiment-oanda-sentiment-2026-06-29.md (written before results).
Fetches ~2y of daily (17:00Z) position books for 7 FX majors from OANDA **practice**
(read-only; cached), builds the strictly-causal signal/return panels, and reports the
pre-registered metrics: pooled IC (+t), per-instrument IC, contrarian-portfolio net
Sharpe (cost-aware) full + OOS + disjoint-year sub-periods. Honest negative = success.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

logger = logging.getLogger("experiment_oanda_sentiment")

MAJORS = ["EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"]
CACHE = REPO_ROOT / "market_data" / "oanda_books" / "sentiment_books_2y.json"
OUT = REPO_ROOT / "trained_data" / "backtests" / "oanda_sentiment_result.json"
START = "2024-06-29"
END = "2026-06-29"
OOS_FRACTION = 0.35


def _aligned_1700z(d: pd.Timestamp) -> str:
    return d.tz_localize("UTC").replace(hour=17, minute=0, second=0, microsecond=0)\
        .isoformat().replace("+00:00", "Z")


def fetch_books() -> dict:
    if CACHE.exists():
        logger.info("loading cached books %s", CACHE)
        return json.loads(CACHE.read_text())
    from src.utils.oanda_practice import OandaApiError, OandaPracticeClient
    c = OandaPracticeClient.from_env()
    days = pd.bdate_range(START, END)
    out = {inst: [] for inst in MAJORS}
    for inst in MAJORS:
        seen = set()
        for d in days:
            t = _aligned_1700z(d)
            try:
                r = c._request("GET", f"/instruments/{inst}/positionBook", params={"time": t})
            except OandaApiError as exc:
                logger.debug("%s %s skip: %s", inst, t, exc)
                continue
            pb = (r or {}).get("positionBook") or {}
            bt = pb.get("time")
            if bt and bt not in seen:   # dedup stale weekend repeats
                seen.add(bt)
                out[inst].append(r)
            time.sleep(0.06)
        logger.info("%s: %d unique books", inst, len(out[inst]))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    return out


def _gate_stats(net: pd.Series) -> dict:
    from src.equity.backtest import stats
    s = stats(net.dropna())
    per_year = s.get("per_year", {}) or {}
    return {"net_sharpe": round(float(s.get("net_sharpe", 0.0)), 3),
            "max_dd": round(float(s.get("max_dd", 1.0)), 3),
            "positive_years": sum(1 for v in per_year.values() if float(v) > 0),
            "total_years": len(per_year)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from src.equity.oanda_sentiment import (
        build_panels, contrarian_portfolio, per_instrument_ic, pooled_ic,
    )
    books = fetch_books()
    sig, ret = build_panels(books)
    if sig.empty:
        logger.error("no panel built")
        return 1
    logger.info("panel: %d dates x %d instruments", len(sig), sig.shape[1])

    idx = sig.index
    split = int(len(idx) * (1.0 - OOS_FRACTION))
    is_idx, oos_idx = idx[:split], idx[split:]

    def block(s, r):
        return {"pooled_ic": pooled_ic(s, r),
                "per_instrument_ic": per_instrument_ic(s, r),
                "portfolio": _gate_stats(contrarian_portfolio(s, r))}

    sub_periods = []
    for y in sorted(set(idx.year)):
        yi = idx[idx.year == y]
        if len(yi) >= 100:
            sub_periods.append({"year": int(y), **{
                "ic": pooled_ic(sig.loc[yi], ret.loc[yi])["ic"],
                "net_sharpe": _gate_stats(contrarian_portfolio(sig.loc[yi], ret.loc[yi]))["net_sharpe"]}})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "running": "NO (offline pre-registered test)",
        "pre_registration": "docs/experiment-oanda-sentiment-2026-06-29.md",
        "universe": MAJORS, "window": {"start": START, "end": END},
        "n_dates": len(sig), "ship_gate_min_net_sharpe": 0.40,
        "full_sample": block(sig, ret),
        "in_sample": block(sig.loc[is_idx], ret.loc[is_idx]),
        "out_of_sample": {"start": str(oos_idx.min().date()), "end": str(oos_idx.max().date()),
                          "fraction": OOS_FRACTION, **block(sig.loc[oos_idx], ret.loc[oos_idx])},
        "sub_period_by_year": sub_periods,
        "data_limitation": "2y window cannot satisfy ship-gate MIN_TOTAL_YEARS=10; "
                           "positive => promising-not-shippable, negative => conclusive no-edge.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
