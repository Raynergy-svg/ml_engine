#!/usr/bin/env python3
"""Run one or more AXIOM operator epochs.

This uses the local Claude CLI subscription session by default. It refuses to
run when ANTHROPIC_API_KEY is present unless AXIOM_OPERATOR_ALLOW_API_KEY=1 is
set, preventing accidental API billing.
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

from src.axiom_operator.runner import AxiomOperator  # noqa: E402


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run one epoch and exit")
    parser.add_argument("--loop", action="store_true", help="run repeated epochs")
    parser.add_argument("--interval", type=float, default=300.0, help="seconds between epochs in --loop mode")
    parser.add_argument("--max-epochs", type=int, default=1, help="maximum epochs to run")
    parser.add_argument("--timeout", type=int, default=120, help="Claude CLI timeout per epoch")
    args = parser.parse_args(argv)

    epochs = 1 if args.once or not args.loop else max(1, args.max_epochs)
    operator = AxiomOperator(project_root=REPO_ROOT)
    for idx in range(epochs):
        result = operator.run_once(
            observation={
                "source": "scripts/run_axiom_operator.py",
                "mode": "loop" if args.loop else "once",
                "epoch_index": idx + 1,
            },
            timeout_seconds=args.timeout,
        )
        print(json.dumps({
            "ok": result.ok,
            "status": result.status,
            "epoch": result.epoch,
            "action": result.action,
            "error": result.error,
            "next_action": result.session.get("next_action"),
        }, indent=2, sort_keys=True))
        if result.status in {"held", "waiting"}:
            break
        if args.loop and idx < epochs - 1:
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
