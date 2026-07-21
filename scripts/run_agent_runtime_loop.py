#!/usr/bin/env python3
"""Run one or more AXIOM resident-loop cycles (OBSERVE->DIAGNOSE->PROPOSE->ACT->VERIFY->LOG).

Shadow by default: unless the operator has explicitly written
``{"agent_autonomy_enabled": true}`` to trained_data/axiom/loop_autonomy.json
(see ``--enable-autonomy`` / ``--disable-autonomy`` below), OPERATIONAL and
DE-ESCALATION actions the loop proposes are logged as shadow intent only and
never executed. ESCALATION actions (unhalt, arm, leverage increase, model/
strategy/code promotion) are ALWAYS proposal-only, regardless of this flag.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent_runtime.loop import ResidentLoop, is_autonomy_enabled, set_autonomy_enabled  # noqa: E402


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run one cycle and exit (default)")
    parser.add_argument("--loop", action="store_true", help="run repeated cycles")
    parser.add_argument("--interval", type=float, default=300.0, help="seconds between cycles in --loop mode")
    parser.add_argument("--max-cycles", type=int, default=1, help="maximum cycles to run")
    parser.add_argument("--timeout", type=int, default=90, help="claude CLI timeout per cycle (seconds)")
    parser.add_argument(
        "--enable-autonomy", action="store_true",
        help="operator opt-in: persist agent_autonomy_enabled=true before running",
    )
    parser.add_argument(
        "--disable-autonomy", action="store_true",
        help="persist agent_autonomy_enabled=false (the default) before running",
    )
    args = parser.parse_args(argv)

    if args.enable_autonomy and args.disable_autonomy:
        parser.error("--enable-autonomy and --disable-autonomy are mutually exclusive")
    if args.enable_autonomy:
        set_autonomy_enabled(True)
    elif args.disable_autonomy:
        set_autonomy_enabled(False)

    resident = ResidentLoop(project_root=REPO_ROOT)
    print(json.dumps({"agent_autonomy_enabled": is_autonomy_enabled(resident.autonomy_path)}, sort_keys=True))

    cycles = 1 if args.once or not args.loop else max(1, args.max_cycles)
    for idx in range(cycles):
        try:
            result = resident.run_cycle(
                actor="scripts/run_agent_runtime_loop.py",
                timeout_seconds=args.timeout,
            )
        except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill the loop
            print(json.dumps({"cycle_error": repr(exc)}, sort_keys=True))
        else:
            print(json.dumps({
                "cycle_id": result.cycle_id,
                "cli_available": result.cli_available,
                "autonomy_enabled": result.autonomy_enabled,
                "proposed_actions": len(result.proposed_actions),
                "executed": sum(1 for o in result.outcomes if o.executed),
                "shadow_logged": sum(1 for o in result.outcomes if o.shadow),
                "proposed_for_operator": sum(1 for o in result.outcomes if o.proposal),
                "denied": sum(1 for o in result.outcomes if o.denied),
                "verified": result.verified,
            }, indent=2, sort_keys=True))
        if args.loop and idx < cycles - 1:
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
