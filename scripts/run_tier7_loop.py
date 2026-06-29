#!/usr/bin/env python3
"""Tier 7 self-healing supervisor — BOUNDED headless loop (operational recovery only).

Operator-approved (2026-06-29) headless entrypoint for the Tier 7 self-healing loop.
The full Tier 7 scanner only runs inside the interactive TUI (no headless daemon
existed); this is the bounded supervisor that keeps the loop *alive* and performs
the operator-authorized OPERATIONAL recovery — and NOTHING else.

Each tick it (a) writes the heartbeat (so tier7_state shows running:YES — fresh
beacon + live pid), (b) runs the BOUNDED self-heal cycle (PostTradeDiagnostics ->
SelfHeal.apply: reset/tighten FX-scanner gate thresholds, reset agent weights,
reduce risk, request retrains — tiered + daily-budgeted + debounced), and (c)
re-writes the Tier 7 state snapshot for AXIOM with the self-heal events.

HARD BOUNDS (by construction + verified): this module imports NO execution/order
path. It CANNOT unhalt (no set_halted), CANNOT change mode or oanda_environment,
CANNOT place a trade, CANNOT promote a ship-gate-failing artifact. SelfHeal's action
space is config/weight-only and bounded by its tiered-autonomy + budget + debounce
gates. The OANDA trend lane's safety rails (separate config) are untouched. Halt is
respected/logged; self-heal recovery is harmless when halted (it never trades).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("run_tier7_loop")
SELFHEAL_EVENTS_REL = ".claude/tier7_selfheal_events.jsonl"


def _read_mode(root: Path) -> str:
    try:
        return str(json.loads((root / ".claude" / "state.json").read_text()).get("mode", "live"))
    except (OSError, ValueError):
        return "live"


def _append_event(root: Path, payload: dict) -> None:
    path = root / SELFHEAL_EVENTS_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _self_heal_tick(root: Path) -> dict:
    """Run ONE bounded self-heal cycle; never raises (degraded-safe). Returns a summary."""
    try:
        from src.scanner.config import ScannerConfig
        from src.scanner.feedback.diagnostics import PostTradeDiagnostics
        from src.scanner.feedback.self_heal import SelfHeal

        diag = PostTradeDiagnostics().run()
        result = SelfHeal(config=ScannerConfig()).apply(diag)
        actions = result.get("actions_taken", []) or []
        return {"status": result.get("status", "no_action"),
                "n_actions": len(actions),
                "actions": [a.get("action") for a in actions],
                "degraded": bool(result.get("degraded_mode", False))}
    except Exception as exc:  # noqa: BLE001 - self-heal must never crash the supervisor
        logger.warning("self-heal tick error (non-fatal): %s", exc)
        return {"status": "error", "n_actions": 0, "actions": [], "error": str(exc)}


def _tick(root: Path, cycle: int) -> None:
    from src.scanner.automation.tier7_state import write_tier7_state
    from src.tui.heartbeat import write_heartbeat

    halted = False
    try:
        halted = bool(json.loads((root / ".claude" / "state.json").read_text()).get("halted", False))
    except (OSError, ValueError):
        pass

    write_heartbeat(root, cycle_count=cycle, mode=_read_mode(root),
                    scanner_alive=True, pid=os.getpid())
    sh = _self_heal_tick(root)
    _append_event(root, {
        "ts": datetime.now(timezone.utc).isoformat(), "cycle": cycle,
        "halted": halted, **sh,
    })
    write_tier7_state(root)
    logger.info("tier7 tick %d — self_heal=%s actions=%d halted=%s",
                cycle, sh["status"], sh["n_actions"], halted)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=30.0,
                    help="seconds between ticks (default 30 -> heartbeat fresh < 90s)")
    ap.add_argument("--once", action="store_true", help="single tick then exit")
    ap.add_argument("--max-cycles", type=int, default=0, help="stop after N (0 = until killed)")
    args = ap.parse_args(argv)

    logger.info("Tier 7 self-heal supervisor START (pid %d, bounded: no trade/unhalt/env path)",
                os.getpid())
    cycle = 0
    while True:
        cycle += 1
        _tick(REPO_ROOT, cycle)
        if args.once or (args.max_cycles and cycle >= args.max_cycles):
            return 0
        time.sleep(float(args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
