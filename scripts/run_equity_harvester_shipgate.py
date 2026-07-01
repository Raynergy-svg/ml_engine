#!/usr/bin/env python3
"""US-005 — driver: re-validate the equity-beta harvester on the single-stock
universe (US-002) via the US-003 cost-aware harness and write
``trained_data/backtests/SHIP_GATE.json`` (canonical) + a timestamped
``equity_harvester_singlestock_*.json`` artifact.

The gate logic + atomic writes live in ``src.equity.ship_gate`` so they are
unit-tested without yfinance or any live source. This script is the
operator-facing wrapper that loads the deployed universe snapshot, pulls
price data, and persists the artifacts.

Usage::

    python scripts/run_equity_harvester_shipgate.py \\
        --universe-path market_data/equity/universe_snapshot.json \\
        --start 2010-01-04 --end 2026-06-01

If ``--universe-path`` is omitted the script falls back to the default
single-stock universe in ``src.equity.DEFAULT_UNIVERSE`` and builds an
unconditional point-in-time membership table from the loaded prices.
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

from src.equity import DEFAULT_UNIVERSE, EQUITY_PIPELINE_VERSION  # noqa: E402
from src.equity.ship_gate import run_and_persist  # noqa: E402
from src.equity.universe import (  # noqa: E402
    UniverseRules,
    UniverseSnapshot,
    compute_universe_hash,
)

logger = logging.getLogger("equity_harvester_shipgate")


def _load_prices_yfinance(
    tickers: list[str], start: str, end: str
) -> pd.DataFrame:
    """Backtest-only price loader — yfinance is documented as backtest-only in
    src/equity/data_loader.py. Returns a date x ticker close-price panel.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - operator-only path
        raise SystemExit(
            "yfinance not installed; install or pre-stage a CSV price panel"
        ) from exc

    df = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        progress=False,
        auto_adjust=True,
    )["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0] if isinstance(tickers, list) else tickers)
    df = df.dropna(how="all")
    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC").normalize()
    return df


def _trivial_snapshot(prices: pd.DataFrame) -> UniverseSnapshot:
    """Fallback snapshot when no on-disk universe is provided: every priced
    name is a member every day. Hash is deterministic over (date, ticker).
    """
    membership = prices.notna()
    # tz_localize is NOT idempotent — it raises on an already-aware index
    # regardless of the target tz. prices.index is already UTC-aware (set in
    # _load_prices_yfinance), so localize only when naive, then normalize.
    idx = pd.DatetimeIndex(membership.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    membership.index = idx.normalize()
    h = compute_universe_hash(membership)
    return UniverseSnapshot(
        membership=membership,
        rules=UniverseRules(top_n=membership.shape[1]),
        built_at=pd.Timestamp.utcnow().isoformat(),
        pipeline_version=EQUITY_PIPELINE_VERSION,
        universe_hash=h,
        reconstitution_dates=(
            pd.Timestamp(membership.index.min()).date().isoformat(),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-path", type=Path, default=None)
    parser.add_argument("--start", default="2010-01-04")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument(
        "--backtests-dir",
        type=Path,
        default=REPO_ROOT / "trained_data" / "backtests",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Override ticker list (default: src.equity.DEFAULT_UNIVERSE).",
    )
    args = parser.parse_args(argv)

    if args.universe_path is not None and args.universe_path.exists():
        snapshot = UniverseSnapshot.load(args.universe_path)
        tickers = sorted(snapshot.membership.columns)
        logger.info(
            "loaded universe snapshot from %s (%d tickers, hash=%s)",
            args.universe_path,
            len(tickers),
            snapshot.universe_hash[:12],
        )
    else:
        tickers = list(args.tickers or DEFAULT_UNIVERSE)
        snapshot = None
        logger.info(
            "no universe snapshot provided — falling back to %d default tickers",
            len(tickers),
        )

    prices = _load_prices_yfinance(tickers, args.start, args.end)
    if snapshot is None:
        snapshot = _trivial_snapshot(prices)

    t0 = time.time()
    report, artifact_path, ship_gate_path = run_and_persist(
        snapshot,
        prices,
        args.backtests_dir,
        pipeline_version=EQUITY_PIPELINE_VERSION,
    )
    elapsed = time.time() - t0

    logger.info(
        "[%s] net_sharpe=%.3f maxDD=%.3f pos=%d/%d hash=%s asof=%s (%.1fs)",
        "PASS" if report.gate_pass else "FAIL",
        report.net_sharpe,
        report.max_dd,
        report.positive_years,
        report.total_years,
        report.universe_hash[:12],
        report.asof,
        elapsed,
    )
    logger.info("wrote artifact: %s", artifact_path)
    logger.info("wrote ship gate: %s", ship_gate_path)
    logger.info("recommendation: %s", report.recommendation)
    return 0 if report.gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
