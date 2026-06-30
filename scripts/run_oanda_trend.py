#!/usr/bin/env python3
"""H1 — OANDA practice TREND demo driver (NON-directional; PRACTICE only).

Operator-chosen lane (2026-06-25): run the validated trend / managed-futures
strategy (price vs MA, long-or-flat, shift(1) causal — NOT the retired directional
FX models) against OANDA v20 **practice** endpoints. Reuses the practice-pinned
``OandaPracticeClient`` (base URL hard-pinned to api-fxpractice; no live path).

Flow: query tradable instruments -> fetch candles -> trend signal -> (demo) market
orders on the practice account. running:YES only if a real process executed a real
cycle and placed a real paper order (L-017/L-018 — no false claims).

Blocker (operator-side): a valid/refreshed OANDA **practice** token + account id.
    export OANDA_API_TOKEN=<practice personal access token>
    export OANDA_ACCOUNT_ID=<101-... practice account id>
(or put them in the project .env). The earlier 401 was a stale token.

Usage::
    python scripts/run_oanda_trend.py                 # one cycle (places demo orders)
    python scripts/run_oanda_trend.py --dry-run       # signal only, no orders
    python scripts/run_oanda_trend.py --loop 3600     # persistent daily-trend loop
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("run_oanda_trend")

# Curated liquid trend universe — intersected with what the account actually
# enables (FX majors + metals + common index/commodity CFDs). Trend runs on
# whatever subset is tradable; no directional view, just price-vs-MA long/flat.
CANDIDATE_INSTRUMENTS = [
    "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_JPY", "GBP_JPY", "EUR_GBP",
    "XAU_USD", "XAG_USD",                       # metals
    "SPX500_USD", "NAS100_USD", "US30_USD",     # equity index CFDs
    "DE30_EUR", "UK100_GBP",
    "BCO_USD", "WTICO_USD", "NATGAS_USD",       # commodity CFDs
]


def _select_instruments(client) -> List[str]:
    """Tradable-instruments endpoint -> the subset of our candidates the account has."""
    resp = client.get_instruments() or {}
    available = {i.get("name") for i in resp.get("instruments", []) or []}
    chosen = [i for i in CANDIDATE_INSTRUMENTS if i in available]
    logger.info("account exposes %d instruments; trend universe = %d: %s",
                len(available), len(chosen), chosen)
    return chosen or sorted(available)[:20]


def _make_client():
    """Build the practice client from env, returning (client, blocker_msg)."""
    try:
        from src.utils.oanda_practice import OandaPracticeClient
        return OandaPracticeClient.from_env(), ""
    except OSError as exc:  # missing env vars
        return None, (f"{exc}  -> set OANDA_API_TOKEN + OANDA_ACCOUNT_ID "
                      "(practice) in the environment or project .env.")
    except Exception as exc:  # pragma: no cover
        return None, f"OANDA client init failed: {type(exc).__name__}: {exc}"


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="signal only, place NO orders")
    ap.add_argument("--granularity", default="D")
    ap.add_argument("--sma", type=int, default=100)
    ap.add_argument("--gross-leverage", type=float, default=None,
                    help="total exposure as a multiple of NAV (env OANDA_GROSS_LEVERAGE; "
                         "default 3.0; capped at 15x to stay clear of margin call)")
    ap.add_argument("--atr-sl-mult", type=float, default=None,
                    help="stop-loss = entry - mult*ATR (env OANDA_ATR_SL_MULT; default 2.0)")
    ap.add_argument("--no-sl", action="store_true", help="disable protective stop-loss")
    ap.add_argument("--enable-tp", action="store_true",
                    help="attach take-profit (default OFF — trend rides winners)")
    ap.add_argument("--max-margin-util", type=float, default=None,
                    help="margin rail: cap projected margin at this fraction of NAV "
                         "(env OANDA_MAX_MARGIN_UTIL; default 0.50)")
    ap.add_argument("--loop", type=float, default=0.0, help="seconds between cycles (>0 = persist)")
    ap.add_argument("--max-cycles", type=int, default=0)
    args = ap.parse_args(argv)

    from src.scanner.config import ScannerConfig
    from src.equity.oanda_trend import (
        DEFAULT_ATR_SL_MULT, DEFAULT_GROSS_LEVERAGE, DEFAULT_MAX_MARGIN_UTIL,
        run_oanda_trend_cycle,
    )
    config = ScannerConfig()
    assert config.oanda_environment == "practice", "HARD LINE: env must stay practice"

    # Dials: CLI > env > default. Leverage clamped in the cycle; SL/TP + margin guard too.
    import os as _os

    def _f(cli, env, default):
        if cli is not None:
            return cli
        try:
            return float(_os.getenv(env, default))
        except (ValueError, TypeError):
            logger.warning("bad %s -> default %s", env, default)
            return default

    gross_leverage = _f(args.gross_leverage, "OANDA_GROSS_LEVERAGE", DEFAULT_GROSS_LEVERAGE)
    atr_sl_mult = _f(args.atr_sl_mult, "OANDA_ATR_SL_MULT", DEFAULT_ATR_SL_MULT)
    max_margin_util = _f(args.max_margin_util, "OANDA_MAX_MARGIN_UTIL", DEFAULT_MAX_MARGIN_UTIL)
    enable_sl = not args.no_sl
    enable_tp = bool(args.enable_tp or _os.getenv("OANDA_ENABLE_TP"))
    logger.info("gross_leverage = %.1fx NAV (dial via --gross-leverage / OANDA_GROSS_LEVERAGE)",
                gross_leverage)

    client, blocker = _make_client()
    if client is None:
        logger.error("OANDA practice lane unavailable: %s", blocker)
        print(f"OANDA_BLOCKER: {blocker}")
        return 2

    # Live connectivity + tradable-instruments probe (surfaces a 401 stale token precisely).
    try:
        instruments = _select_instruments(client)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("tradable-instruments query FAILED: %s", msg)
        print(f"OANDA_BLOCKER: tradable-instruments query failed ({msg}). "
              "Most likely a stale/invalid practice token (401) — refresh OANDA_API_TOKEN.")
        return 2

    def _cycle() -> int:
        # AXIOM control override: operator may dial gross-leverage live from the
        # dashboard. The override is read each cycle so a running loop honors it
        # without restart; run_oanda_trend_cycle re-clamps to the 15x cap.
        lev = gross_leverage
        try:
            import json as _json
            _ov = _json.loads((REPO_ROOT / "trained_data" / "axiom" / "control_overrides.json").read_text())
            if isinstance(_ov.get("gross_leverage"), (int, float)):
                lev = float(_ov["gross_leverage"])
        except (OSError, ValueError, TypeError):
            pass
        try:
            r = run_oanda_trend_cycle(
                client=client, config=config, instruments=instruments,
                project_root=REPO_ROOT, granularity=args.granularity,
                sma_window=args.sma, gross_leverage=lev,
                enable_sl=enable_sl, enable_tp=enable_tp, atr_sl_mult=atr_sl_mult,
                max_margin_util=max_margin_util, dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"OANDA_BLOCKER: cycle failed ({type(exc).__name__}: {exc}) — "
                  "likely a stale practice token (401). Refresh OANDA_API_TOKEN.")
            return 2
        # Monitoring for the TUI (best-effort, non-blocking): refresh account
        # state + ingest the transaction ledger (trade history + realized P&L).
        try:
            from src.brokers import oanda_v20 as v20
            from src.scanner.automation.tier7_state import write_tier7_state
            v20.snapshot_account_state(client)
            v20.TransactionLedger(client).sync()
            write_tier7_state(REPO_ROOT)   # read-only Tier 7 snapshot for AXIOM
        except Exception as exc:  # never let monitoring break a cycle
            logger.warning("monitor snapshot failed (non-blocking): %s", exc)
        on = sum(1 for v in r.targets.values() if v > 0)
        print(f"CYCLE_RESULT: ran={r.ran} reason={r.reason} on={on}/{len(r.targets)} "
              f"orders_placed={r.orders_placed}", flush=True)
        return 0 if r.ran else 1

    if args.loop and args.loop > 0:
        import time
        logger.info("LOOP mode: cycle every %.0fs (max=%s)", args.loop, args.max_cycles or "inf")
        n = 0
        while True:
            _cycle()
            n += 1
            if args.max_cycles and n >= args.max_cycles:
                return 0
            time.sleep(float(args.loop))
    return _cycle()


if __name__ == "__main__":
    raise SystemExit(main())
