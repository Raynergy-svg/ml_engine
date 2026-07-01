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
_SINGLETON_LOCK_FH = None  # holds the singleton flock for this process's lifetime

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


def _control_override_path() -> Path:
    return REPO_ROOT / "trained_data" / "axiom" / "control_overrides.json"


def _read_control_leverage(fallback: float) -> float:
    """Live AXIOM override: dashboard writes the gross-leverage dial here."""
    try:
        import json as _json
        _ov = _json.loads(_control_override_path().read_text(encoding="utf-8"))
        if isinstance(_ov.get("gross_leverage"), (int, float)):
            return float(_ov["gross_leverage"])
    except (OSError, ValueError, TypeError):
        pass
    return fallback


def _override_mtime() -> float:
    try:
        return _control_override_path().stat().st_mtime
    except OSError:
        return 0.0


def _singleton_lock_path() -> Path:
    return REPO_ROOT / "trained_data" / "axiom" / "trend_loop.singleton.lock"


def _acquire_singleton_lock(lock_path: Path):
    """Exclusive, non-blocking process lock (verifier finding, 2026-07-01): two
    concurrent ``run_oanda_trend`` processes each read a position snapshot, each
    independently pass the risk gate against that (now-stale) snapshot, and each
    place an order — reintroducing duplicate tickets via a process-concurrency
    route instead of the original sizing-bug route (matches this project's prior
    "two-writer collision" incidents). Returns an open file object the caller
    MUST keep referenced for the process lifetime (closing/GC'ing it releases
    the lock) — a live process holding it makes this call return ``None``, and
    the OS releases the flock automatically on process exit or crash, so there
    is no stale-lock-file cleanup problem.
    """
    import fcntl
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    import os as _os2
    fh.write(str(_os2.getpid()))
    fh.flush()
    return fh


def _sleep_until_next_cycle(seconds: float, *, poll_s: float = 5.0) -> None:
    """Sleep for the cadence, but wake early when AXIOM controls change."""
    import time
    deadline = time.monotonic() + float(seconds)
    seen = _override_mtime()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(float(poll_s), remaining))
        current = _override_mtime()
        if current > seen:
            logger.info("AXIOM control override changed — waking trend cycle early")
            return


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
                    help="attach take-profit (kept for compatibility; TP is ON by default)")
    ap.add_argument("--no-tp", action="store_true",
                    help="disable take-profit; diagnostic only, not Tier 7 default flow")
    ap.add_argument("--max-margin-util", type=float, default=None,
                    help="margin rail: cap projected margin at this fraction of NAV "
                         "(env OANDA_MAX_MARGIN_UTIL; default 0.50)")
    ap.add_argument("--risk-pct", type=float, default=None,
                    help="RISK GATE rule 3: risk this fraction of NAV per new position, "
                         "sized off the ATR stop distance (env OANDA_RISK_PCT; default 0.01, "
                         "clamped to [0.005, 0.03]). gross_leverage is now a CEILING only.")
    ap.add_argument("--max-bucket-risk-r", type=float, default=None,
                    help="RISK GATE rule 2: cap total risk per shared-currency bucket "
                         "(e.g. all yen-cross longs) at this many R-units "
                         "(env OANDA_MAX_BUCKET_RISK_R; default 2.0)")
    ap.add_argument("--loop", type=float, default=0.0, help="seconds between cycles (>0 = persist)")
    ap.add_argument("--max-cycles", type=int, default=0)
    args = ap.parse_args(argv)

    from src.scanner.config import ScannerConfig
    from src.equity.oanda_trend import (
        DEFAULT_ATR_SL_MULT, DEFAULT_GROSS_LEVERAGE, DEFAULT_MAX_MARGIN_UTIL,
        run_oanda_trend_cycle,
    )
    from src.equity.trend_risk_gates import DEFAULT_MAX_BUCKET_RISK_R, DEFAULT_RISK_PCT
    config = ScannerConfig()
    assert config.oanda_environment == "practice", "HARD LINE: env must stay practice"

    # SINGLETON GUARD (verifier finding, 2026-07-01): refuse to start a second
    # concurrent order-placing process — kept referenced for main()'s whole
    # lifetime (module-global so it survives this function's local scope).
    global _SINGLETON_LOCK_FH
    _SINGLETON_LOCK_FH = _acquire_singleton_lock(_singleton_lock_path())
    if _SINGLETON_LOCK_FH is None:
        logger.error("another run_oanda_trend process already holds the singleton "
                     "lock (%s) — refusing to start", _singleton_lock_path())
        print(f"OANDA_BLOCKER: singleton lock held — another run_oanda_trend "
              f"process is already running ({_singleton_lock_path()})")
        return 3

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
    risk_pct = _f(args.risk_pct, "OANDA_RISK_PCT", DEFAULT_RISK_PCT)
    max_bucket_risk_r = _f(args.max_bucket_risk_r, "OANDA_MAX_BUCKET_RISK_R", DEFAULT_MAX_BUCKET_RISK_R)
    enable_sl = not args.no_sl
    enable_tp = not bool(args.no_tp or _os.getenv("OANDA_DISABLE_TP"))
    _effective_lev = _read_control_leverage(gross_leverage)
    if abs(_effective_lev - gross_leverage) > 1e-9:
        logger.info("gross_leverage = %.2fx NAV EFFECTIVE — AXIOM control override active "
                    "(supersedes default %.1fx from --gross-leverage / OANDA_GROSS_LEVERAGE)",
                    _effective_lev, gross_leverage)
    else:
        logger.info("gross_leverage = %.1fx NAV (dial via --gross-leverage / OANDA_GROSS_LEVERAGE; "
                    "no AXIOM override active)", gross_leverage)

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
        lev = _read_control_leverage(gross_leverage)
        logger.info("effective gross_leverage for this cycle = %.2fx", lev)
        try:
            r = run_oanda_trend_cycle(
                client=client, config=config, instruments=instruments,
                project_root=REPO_ROOT, granularity=args.granularity,
                sma_window=args.sma, gross_leverage=lev,
                enable_sl=enable_sl, enable_tp=enable_tp, atr_sl_mult=atr_sl_mult,
                max_margin_util=max_margin_util, risk_pct=risk_pct,
                max_bucket_risk_r=max_bucket_risk_r, dry_run=args.dry_run,
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
            _sleep_until_next_cycle(float(args.loop))
    return _cycle()


if __name__ == "__main__":
    raise SystemExit(main())
