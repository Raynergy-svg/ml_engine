#!/usr/bin/env python3
"""Run the bounded daily, analysis-only forward-capture job graph."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    python = sys.executable
    jobs = (
        ("track_b_filings", [python, "scripts/capture_track_b_new_filings.py"], 3600, {0}),
        ("track_b_rebalance", [python, "scripts/run_track_b_shadow.py", "--refresh-prices"], 1800, {0, 1}),
        ("hedge_forward", [python, "-m", "src.hedge.hedged_shadow_lane"], 1800, {0}),
        ("p2_readiness", [python, "scripts/check_risk_target_p2_readiness.py"], 300, {0, 3}),
    )
    results: dict[str, dict[str, object]] = {}
    hard_failure = False
    for name, command, timeout, accepted_codes in jobs:
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                timeout=timeout,
                check=False,
                text=True,
                capture_output=True,
            )
            results[name] = {
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
            if completed.returncode not in accepted_codes:
                hard_failure = True
        except subprocess.TimeoutExpired as exc:
            hard_failure = True
            results[name] = {"timeout": timeout, "error": str(exc)}
    print(json.dumps(results, indent=2, sort_keys=True))
    return 1 if hard_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
