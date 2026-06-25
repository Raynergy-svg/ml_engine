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
    ap.add_argument("--loop", type=float, default=0.0, help="seconds between cycles (>0 = persist)")
    ap.add_argument("--max-cycles", type=int, default=0)
    args = ap.parse_args(argv)

    from src.scanner.config import ScannerConfig
    from src.equity.oanda_trend import run_oanda_trend_cycle
    config = ScannerConfig()
    assert config.oanda_environment == "practice", "HARD LINE: env must stay practice"

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
        try:
            r = run_oanda_trend_cycle(
                client=client, config=config, instruments=instruments,
                project_root=REPO_ROOT, granularity=args.granularity,
                sma_window=args.sma, dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"OANDA_BLOCKER: cycle failed ({type(exc).__name__}: {exc}) — "
                  "likely a stale practice token (401). Refresh OANDA_API_TOKEN.")
            return 2
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
