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


def _self_heal_tick(root: Path, max_autonomy_level: int) -> dict:
    """Run ONE bounded self-heal cycle; never raises (degraded-safe). Returns a summary.

    ``max_autonomy_level`` CLAMPS the self-heal action ceiling (default 3). At 3 only
    trivially-reversible operational heals auto-apply (reset agent weight, reset gate
    threshold to default); LEVEL_4 (tighten/reduce-risk) and LEVEL_5 (retrain_gates /
    retrain_rl_position_sizer) are REFUSED. This is the doctrine-matching clamp — the
    ScannerConfig dataclass default is 5, which would let the unattended loop auto-retrain.
    """
    try:
        from src.scanner.config import ScannerConfig
        from src.scanner.feedback.diagnostics import PostTradeDiagnostics
        from src.scanner.feedback.self_heal import SelfHeal

        cfg = ScannerConfig()
        cfg.self_heal_max_autonomy_level = int(max_autonomy_level)  # clamp (default 3)
        diag = PostTradeDiagnostics().run()
        result = SelfHeal(config=cfg).apply(diag)
        actions = result.get("actions_taken", []) or []
        applied = [a for a in actions if a.get("success")]
        return {"status": result.get("status", "no_action"),
                "n_actions": len(actions),
                "n_applied": len(applied),
                "actions": [a.get("action") for a in actions],
                "recommendations": result.get("recommendations", []) or [],
                "degraded": bool(result.get("degraded_mode", False))}
    except Exception as exc:  # noqa: BLE001 - self-heal must never crash the supervisor
        logger.warning("self-heal tick error (non-fatal): %s", exc)
        return {"status": "error", "n_actions": 0, "actions": [], "error": str(exc)}


def _foreign_writer_owns_heartbeat(root: Path, *, fresh_s: float = 25.0) -> bool:
    """True when ANOTHER live process (the TUI, cadence ~10s) wrote a fresh heartbeat.

    Two-writer collision fix (2026-07-03 autonomy audit): when the TUI runs it writes
    the REAL scanner_alive every ~10s; the supervisor must not clobber that beacon
    with its own. Fresh + foreign pid + pid alive => the other writer owns the file.
    """
    try:
        hb = json.loads((root / ".claude" / "heartbeat.json").read_text())
        pid = int(hb.get("pid", 0))
        ts = datetime.fromisoformat(str(hb.get("ts_iso", "")).replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        if pid and pid != os.getpid() and age <= fresh_s:
            os.kill(pid, 0)  # raises if dead
            return True
    except (OSError, ValueError, OverflowError):
        pass
    return False


def _tick(root: Path, cycle: int, max_autonomy_level: int) -> None:
    from src.scanner.automation.maintenance_report import (
        build_maintenance_report, write_maintenance_report,
    )
    from src.scanner.automation.tier7_state import write_tier7_state
    from src.tui.heartbeat import write_heartbeat

    halted = False
    try:
        halted = bool(json.loads((root / ".claude" / "state.json").read_text()).get("halted", False))
    except (OSError, ValueError):
        pass

    # Honest beacon (L-017, fixed 2026-07-03): this supervisor is NOT the scanner and
    # cannot prove one is running, so it writes scanner_alive=False (fail-closed
    # honesty) plus additive writer/supervisor keys. If a fresh FOREIGN heartbeat
    # exists (the TUI writing its real value every ~10s), skip the write entirely —
    # the interactive process owns the beacon while it lives.
    if not _foreign_writer_owns_heartbeat(root):
        write_heartbeat(root, cycle_count=cycle, mode=_read_mode(root),
                        scanner_alive=False, pid=os.getpid(),
                        extra={"writer": "tier7_supervisor", "supervisor_alive": True})
    sh = _self_heal_tick(root, max_autonomy_level)
    # P1 headless adjustment consumption — OPT-IN (TIER7_CONSUME_ADJUSTMENTS=1).
    # Default OFF: until the engine startup path calls config_overlay.apply_overlay,
    # consuming here would hide adjustments from a future TUI session (it skips
    # consumed ids and doesn't read the overlay yet). Operator flips both together.
    if os.getenv("TIER7_CONSUME_ADJUSTMENTS", "0") == "1":
        try:
            from src.scanner.automation.config_overlay import consume_approved_adjustments
            consumed = consume_approved_adjustments(root)
            if consumed.get("consumed") or consumed.get("orphans_skipped"):
                sh = {**sh, "overlay_consumed": consumed["consumed"],
                      "overlay_orphans": consumed["orphans_skipped"]}
        except Exception as exc:  # noqa: BLE001 — never kill the tick
            logger.warning("overlay consume error (non-fatal): %s", exc)
    _append_event(root, {
        "ts": datetime.now(timezone.utc).isoformat(), "cycle": cycle,
        "halted": halted, **sh,
    })
    try:
        report = build_maintenance_report(root, supervisor_pid=os.getpid(),
                                          recommendations=sh.get("recommendations"))
        write_maintenance_report(root, report)
        needs = report.get("needs_attention", [])
        if needs:
            logger.info("maintenance: NEEDS ATTENTION — %s", "; ".join(needs))
    except Exception as exc:  # noqa: BLE001 — reporting must never kill the tick
        logger.warning("maintenance report error (non-fatal): %s", exc)
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
    ap.add_argument("--max-autonomy", type=int, default=None,
                    help="self-heal action ceiling (env TIER7_MAX_AUTONOMY; default 5 = full: "
                         "3=reversible operational heals only, 4=guarded, 5=allow retrains). "
                         "Operator-raised to 5 on 2026-06-29. Lower with --max-autonomy 3 if desired.")
    args = ap.parse_args(argv)

    # Self-heal autonomy ceiling. Operator RAISED the default 3->5 on 2026-06-29 (full
    # autonomy). Structural immutables hold even at L5 (verifier-confirmed + disk re-derived):
    # the retrain handlers only rewrite sklearn gate baselines / a marker — they CANNOT
    # unhalt, flip env, touch real money, promote/un-quarantine, or write a .keras champion.
    # Bounded to [3,5]; bad input -> 5.
    raw = args.max_autonomy if args.max_autonomy is not None else os.getenv("TIER7_MAX_AUTONOMY", "5")
    try:
        max_autonomy = min(5, max(3, int(raw)))
    except (ValueError, TypeError):
        max_autonomy = 5   # default after the 2026-06-29 operator raise
    logger.info("Tier 7 self-heal supervisor START (pid %d, bounded: no trade/unhalt/env path; "
                "self_heal max_autonomy_level=%d)", os.getpid(), max_autonomy)

    # Code-freshness self-restart (2026-07-03 autonomy fix): this daemon may run
    # OUTSIDE launchd (nohup), so a committed fix would otherwise never go live.
    # On a debounced change (git HEAD moved, or this script edited) it re-execs
    # itself with the same argv — the fresh process re-reads halt/env/config from
    # disk, so this can only make state HONEST, never bypass a gate.
    from src.scanner.automation.code_freshness import CodeFreshness
    freshness = CodeFreshness(REPO_ROOT, watch_files=[Path(__file__).resolve()])

    cycle = 0
    while True:
        cycle += 1
        _tick(REPO_ROOT, cycle, max_autonomy)
        if args.once or (args.max_cycles and cycle >= args.max_cycles):
            return 0
        reason = freshness.changed()
        if reason:
            logger.info("code changed on disk (%s) — re-exec for fresh code (pid %d)",
                        reason, os.getpid())
            _append_event(REPO_ROOT, {
                "ts": datetime.now(timezone.utc).isoformat(), "cycle": cycle,
                "status": "self_restart", "reason": reason, "n_actions": 0, "actions": [],
            })
            os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())]
                     + sys.argv[1:])
        time.sleep(float(args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
