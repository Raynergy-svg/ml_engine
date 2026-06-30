#!/usr/bin/env python3
"""H1 — live driver for the equity-beta harvester (PAPER/PRACTICE ONLY).

Operator-authorized (2026-06-25) entrypoint that drives ONE (or a looped) real
rebalance cycle of the validated, ship-gate-passing single-stock harvester. This
is the previously-held H1 step: it takes ``src.equity.runner.run_shadow_rebalance``
(the gated, non-hot-path tick) and actually invokes it from a live process.

HARD LINE — PAPER/PRACTICE ONLY. ``oanda_environment`` stays ``practice``; the
only broker lane this script will ever build is IBKR **paper** (TWS port 7497).
It never constructs a real-money/live broker and never flips the environment. The
global halt + ship-gate + drawdown + staleness rails in ``decide_cycle`` govern
every tick; a halted ``.claude/state.json`` REFUSES regardless of this script.

Two fill lanes:
  * ``--broker shadow`` (DEFAULT): deterministic simulated fills, no broker.
    A real process executes a real cycle and SIMULATES a paper order. Verified.
  * ``--broker ibkr-paper``: attempts a live IBKR **paper** connection (port 7497)
    and, if connected, places whole-share paper orders via the broker's documented
    ``place_equity_order``. The fill path is UNVERIFIED end-to-end until a live
    paper gateway exists (it cannot be exercised without IB Gateway + ib_insync +
    an IBKR paper login) — so absent a connection it reports the exact blocker and,
    unless ``--no-fallback`` is set, falls back to the shadow lane for the cycle.

Usage::

    python scripts/run_equity_harvester.py                 # one shadow cycle
    python scripts/run_equity_harvester.py --broker ibkr-paper
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.equity.rebalance import Order  # noqa: E402
from src.equity.runner import run_shadow_rebalance  # noqa: E402
from src.equity.universe import UniverseSnapshot  # noqa: E402

logger = logging.getLogger("run_equity_harvester")

_SNAPSHOT_REL = "market_data/equity/universe_snapshot_pit.json"
_SHIP_GATE_REL = "trained_data/backtests/SHIP_GATE.json"
IBKR_PAPER_PORT = 7497  # TWS paper-trading port (7496 = live; never used here)


def _fetch_close_panel(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """yfinance daily adjusted closes (PRICES ONLY), UTC-normalized date×ticker."""
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, interval="1d",
                      progress=False, auto_adjust=True, group_by="ticker")
    cols = {}
    for t in tickers:
        try:
            sub = raw[t]
        except KeyError:
            continue
        if "Close" in sub:
            cols[t] = sub["Close"]
    panel = pd.DataFrame(cols).sort_index()
    panel.index = pd.to_datetime(panel.index, utc=True).normalize()
    return panel


def _make_ibkr_paper_fill(broker, prices: pd.DataFrame) -> Callable[[Order], bool]:
    """Build a PAPER execute_order that places whole-share orders on IBKR paper.

    UNVERIFIED end-to-end (requires a live paper gateway to exercise). Uses the
    tested ``whole_share_round`` for sizing and the broker's documented
    ``place_equity_order``; an EQUITY ship-gate guard must already be unlocked.
    """
    from src.brokers.instrument import Instrument
    from src.equity.order_lifecycle import whole_share_round

    nav = float(broker.get_nav())
    last_close = prices.ffill().iloc[-1]

    def _fill(order: Order) -> bool:
        price = float(last_close.get(order.ticker, float("nan")))
        if not (price > 0):
            logger.warning("paper fill SKIP %s — no price", order.ticker)
            return False
        shares = int(whole_share_round(
            ticker=order.ticker, target_weight=order.target_weight,
            nav=nav, price=price,
        ).shares)
        if shares <= 0:
            return True  # nothing to trade for this name
        instrument = Instrument(
            symbol=order.ticker, broker_symbol=order.ticker, asset_class="EQUITY",
            price_precision=2, margin_requirement=0.0, exchange="SMART", currency="USD",
        )
        result = broker.place_equity_order(instrument, order.side, shares)
        ok = getattr(result, "filled_quantity", 0) > 0 or getattr(result, "status", "") in ("FILLED", "SUBMITTED")
        logger.info("paper order %s %s x%d -> %s", order.side, order.ticker, shares, result)
        return bool(ok)

    return _fill


def _connect_ibkr_paper(snapshot, prices, root: Path):
    """Attempt a live IBKR PAPER connection. Returns (fill_callback or None, blocker_msg)."""
    try:
        from src.brokers.factory import create_broker
    except Exception as exc:  # pragma: no cover
        return None, f"broker factory import failed: {exc}"
    try:
        broker = create_broker(
            broker_type="ibkr", oanda_environment="practice",
            ibkr_host="127.0.0.1", ibkr_port=IBKR_PAPER_PORT, ibkr_client_id=1,
        )
    except Exception as exc:
        return None, (f"could not construct IBKR paper broker ({type(exc).__name__}: {exc}). "
                      "Likely missing dependency `ib_insync` (pip install ib_insync).")
    try:
        broker.connect()
    except Exception as exc:
        return None, (f"IBKR paper connect FAILED ({type(exc).__name__}: {exc}). "
                      f"Need IB Gateway/TWS running on 127.0.0.1:{IBKR_PAPER_PORT} (paper) "
                      "+ an IBKR paper login.")
    if not broker.is_connected():
        return None, (f"IBKR paper broker not connected after connect() — start IB Gateway "
                      f"on 127.0.0.1:{IBKR_PAPER_PORT} (paper) and log in.")
    try:  # unlock EQUITY contract construction for THIS validated universe
        from src.brokers.ibkr import enforce_equity_ship_gate
        enforce_equity_ship_gate(root / _SHIP_GATE_REL, snapshot.universe_hash)
    except Exception as exc:
        return None, f"equity ship-gate guard refused unlock ({type(exc).__name__}: {exc})."
    logger.info("IBKR PAPER connected (port %d), NAV=%.2f", IBKR_PAPER_PORT, broker.get_nav())
    return _make_ibkr_paper_fill(broker, prices), ""


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--broker", choices=["shadow", "ibkr-paper"], default="shadow")
    ap.add_argument("--no-fallback", action="store_true",
                    help="if ibkr-paper can't connect, FAIL instead of shadow fallback")
    ap.add_argument("--history-days", type=int, default=420)
    ap.add_argument("--loop", type=float, default=0.0,
                    help="seconds between ticks; >0 runs a persistent loop (process stays alive)")
    ap.add_argument("--max-cycles", type=int, default=0,
                    help="stop after N cycles in --loop mode (0 = run until killed)")
    args = ap.parse_args(argv)

    from src.scanner.config import ScannerConfig
    config = ScannerConfig()
    config.enable_equity_harvester = True  # operator-authorized for this run
    assert config.oanda_environment == "practice", "HARD LINE: env must stay practice"

    snapshot = UniverseSnapshot.load(REPO_ROOT / _SNAPSHOT_REL)
    members = list(snapshot.membership.columns)
    end = (snapshot.membership.index.max() + pd.Timedelta(days=2)).date().isoformat()
    start = (snapshot.membership.index.max() - pd.Timedelta(days=int(args.history_days))).date().isoformat()
    logger.info("fetching prices for %d members %s..%s", len(members), start, end)
    close = _fetch_close_panel(members, start, end)
    close = close.reindex(index=snapshot.membership.index).ffill()
    asof = snapshot.membership.index.max()

    fill = None
    if args.broker == "ibkr-paper":
        fill, blocker = _connect_ibkr_paper(snapshot, close, REPO_ROOT)
        if fill is None:
            logger.error("IBKR PAPER lane unavailable: %s", blocker)
            if args.no_fallback:
                print(f"PAPER_BLOCKER: {blocker}")
                return 2
            logger.warning("falling back to SHADOW fills for this cycle")

    lane = "ibkr-paper" if fill is not None else "shadow"

    def _tick() -> "object":
        result = run_shadow_rebalance(
            config=config, snapshot=snapshot, prices=close, asof=asof,
            project_root=REPO_ROOT, execute_order=fill,
        )
        n_orders = len(result.plan.orders) if result.plan is not None else 0
        print(f"CYCLE_RESULT: ran={result.ran} reason={result.reason} "
              f"lane={lane} orders={n_orders} asof={asof.date()}", flush=True)
        return result

    if args.loop and args.loop > 0:
        import time
        logger.info("LOOP mode: tick every %.0fs (max_cycles=%s) — Ctrl-C / kill to stop",
                    args.loop, args.max_cycles or "inf")
        n = 0
        while True:
            _tick()  # decide_cycle re-reads halt every tick — re-halting stops trades
            n += 1
            if args.max_cycles and n >= args.max_cycles:
                logger.info("reached --max-cycles=%d, exiting", args.max_cycles)
                return 0
            time.sleep(float(args.loop))

    return 0 if _tick().ran else 1


if __name__ == "__main__":
    raise SystemExit(main())
