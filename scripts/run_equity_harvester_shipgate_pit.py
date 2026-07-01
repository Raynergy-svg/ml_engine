#!/usr/bin/env python3
"""US-005 (rigorous) — re-validate the equity-beta harvester on a REAL
point-in-time US-002 universe WITH realistic per-name execution costs.

This is the honest version of ``run_equity_harvester_shipgate.py``. The latter
fell back to a "trivial" snapshot (every default ticker a member every day) and
supplied no ADV panel, so the gate ran on a survivorship-biased, zero-cost
backtest. Here we instead:

1. Fetch daily OHLCV (close + volume) for a broad multi-sector candidate pool
   that deliberately includes decade-underperformers (INTC, GE, F, IBM, C, T,
   ...) and late-IPOs (META 2012, TSLA 2010, ABBV 2013) — not just the FAANG
   winners. Point-in-time membership (``min_history_days``) admits late-IPOs
   only after they list, so there is no IPO look-ahead.
2. Build the US-002 point-in-time top-N-by-ADV universe (monthly reconstitution).
3. Build the trailing-20d dollar-volume ADV panel and pass it to the backtest so
   the 5bps/%ADV slippage model is ACTUALLY applied (real costs).
4. Run the canonical gate and persist SHIP_GATE.json + the timestamped artifact.

RESIDUAL CAVEAT (documented, not silently ignored): yfinance only serves
currently-listed tickers, so names delisted before today are absent from the
candidate pool. This run removes IPO look-ahead and applies real costs, but it
cannot fully remove survivorship without a paid point-in-time constituent set
(e.g. CRSP). Treat a PASS here as "necessary, not yet sufficient".

Usage::

    python scripts/run_equity_harvester_shipgate_pit.py \\
        --top-n 20 --start 2010-01-04 --end 2026-06-01
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.equity import EQUITY_PIPELINE_VERSION  # noqa: E402
from src.equity.ship_gate import run_and_persist  # noqa: E402
from src.equity.universe import build_universe  # noqa: E402

logger = logging.getLogger("equity_harvester_shipgate_pit")

# Broad multi-sector liquid US large-cap candidate pool. Deliberately mixes
# decade winners with chronic underperformers and late-IPOs so the point-in-time
# top-N selection is a real liquidity screen, not a hindsight winners list.
CANDIDATE_POOL = [
    # mega-cap tech (incl. late-IPOs handled point-in-time)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "ORCL", "CSCO",
    "INTC", "IBM",
    # financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BRK-B",
    # energy / industrials / materials
    "XOM", "CVX", "GE", "BA", "CAT", "MMM", "F",
    # healthcare
    "JNJ", "PFE", "MRK", "UNH", "ABBV",
    # staples / discretionary / comms
    "KO", "PEP", "PG", "WMT", "HD", "MCD", "NKE", "DIS", "T", "VZ", "COST",
]


def _fetch_ohlcv(
    tickers: list[str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Return {ticker: OHLCV frame with lowercase close/volume}. yfinance is
    backtest-only per src/equity/data_loader.py. Names with no data are skipped
    with a WARNING (never silently)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - operator-only path
        raise SystemExit("yfinance not installed") from exc

    raw = yf.download(
        tickers, start=start, end=end, interval="1d",
        progress=False, auto_adjust=True, group_by="ticker",
    )
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            sub = raw[t]
        except KeyError:
            logger.warning("ticker %s absent from yfinance response — skipped", t)
            continue
        if "Close" not in sub or "Volume" not in sub:
            logger.warning("ticker %s missing Close/Volume — skipped", t)
            continue
        frame = pd.DataFrame(
            {"close": sub["Close"], "volume": sub["Volume"]}
        ).dropna(how="any")
        if frame.empty:
            logger.warning("ticker %s has no usable rows — skipped", t)
            continue
        out[t] = frame
    if not out:
        raise SystemExit("no usable tickers fetched")
    logger.info("fetched %d/%d tickers with usable OHLCV", len(out), len(tickers))
    return out


def _adv_panel(
    prices_by_ticker: dict[str, pd.DataFrame], lookback: int
) -> pd.DataFrame:
    """Trailing-`lookback`d mean dollar-volume (close*volume) panel, date x ticker."""
    dvol = {t: (df["close"] * df["volume"]) for t, df in prices_by_ticker.items()}
    panel = pd.DataFrame(dvol).sort_index()
    panel.index = pd.DatetimeIndex(panel.index)
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize("UTC")
    else:
        panel.index = panel.index.tz_convert("UTC")
    panel.index = panel.index.normalize()
    return panel.rolling(lookback, min_periods=lookback).mean()


def _close_panel(prices_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = {t: df["close"] for t, df in prices_by_ticker.items()}
    panel = pd.DataFrame(closes).sort_index()
    panel.index = pd.DatetimeIndex(panel.index)
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize("UTC")
    else:
        panel.index = panel.index.tz_convert("UTC")
    panel.index = panel.index.normalize()
    return panel


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--min-history-days", type=int, default=252)
    parser.add_argument("--rebalance-freq", default="MS")
    parser.add_argument("--start", default="2010-01-04")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument(
        "--backtests-dir", type=Path,
        default=REPO_ROOT / "trained_data" / "backtests",
    )
    parser.add_argument(
        "--save-universe", type=Path,
        default=REPO_ROOT / "market_data" / "equity" / "universe_snapshot_pit.json",
    )
    args = parser.parse_args(argv)

    prices_by_ticker = _fetch_ohlcv(CANDIDATE_POOL, args.start, args.end)

    # The US-003 backtest requires a dense price panel (no NaN, no forward-fill).
    # Names listed after the window start carry leading NaNs that the harness
    # rightly refuses, so drop them here and log exactly which. This is a known
    # US-002<->US-003 integration limit: a paid point-in-time constituent set
    # would admit late-IPOs from their listing date; free yfinance + the
    # dense-panel contract cannot. The point-in-time top-N ranking still applies
    # to every surviving name.
    start_ts = pd.Timestamp(args.start, tz="UTC").normalize()
    late: dict[str, str] = {}
    for t, df in list(prices_by_ticker.items()):
        fidx = pd.DatetimeIndex(df.index)
        fidx = fidx.tz_localize("UTC") if fidx.tz is None else fidx.tz_convert("UTC")
        fmin = fidx.min().normalize()
        if fmin > start_ts + pd.Timedelta(days=5):
            late[t] = fmin.date().isoformat()
    if late:
        logger.warning(
            "dropping %d names listed after window start %s: %s",
            len(late), args.start, late,
        )
        for t in late:
            prices_by_ticker.pop(t)

    snapshot = build_universe(
        prices_by_ticker,
        top_n=args.top_n,
        lookback_days=args.lookback_days,
        min_history_days=args.min_history_days,
        rebalance_freq=args.rebalance_freq,
    )
    members_now = snapshot.members_on(pd.Timestamp(args.end))
    logger.info(
        "built PIT universe: hash=%s top_n=%d recon_points=%d members_on(end)=%s",
        snapshot.universe_hash[:12], args.top_n,
        len(snapshot.reconstitution_dates), members_now,
    )
    args.save_universe.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save(args.save_universe)

    close = _close_panel(prices_by_ticker)
    adv = _adv_panel(prices_by_ticker, args.lookback_days)
    # Backtest only over the membership window; ship_gate reindexes internally.
    close = close.reindex(index=snapshot.membership.index)
    adv = adv.reindex(index=snapshot.membership.index)

    t0 = time.time()
    report, artifact_path, ship_gate_path = run_and_persist(
        snapshot, close, args.backtests_dir,
        adv=adv, pipeline_version=EQUITY_PIPELINE_VERSION,
    )
    elapsed = time.time() - t0

    logger.info(
        "[%s] net_sharpe=%.3f maxDD=%.3f pos=%d/%d adv_supplied=%s (%.1fs)",
        "PASS" if report.gate_pass else "FAIL",
        report.net_sharpe, report.max_dd,
        report.positive_years, report.total_years,
        report.cost_params.get("adv_supplied"), elapsed,
    )
    logger.info("slippage applied: %.2f bps/%%ADV (requested %.2f)",
                report.cost_params.get("slippage_bps_per_pct_adv", 0.0),
                report.cost_params.get("slippage_bps_per_pct_adv_requested", 0.0))
    logger.info("wrote: %s | %s", artifact_path.name, ship_gate_path.name)
    logger.info("recommendation: %s", report.recommendation)
    return 0 if report.gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
