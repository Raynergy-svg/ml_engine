#!/usr/bin/env python3
"""Run basic planner tool-selection evals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.planner_runtime import PlannerRuntime


def _matches_expected_steps(plan_steps, expected_steps):
    if len(plan_steps) != len(expected_steps):
        return False
    for actual, expected in zip(plan_steps, expected_steps):
        if actual.tool != expected.get("tool"):
            return False
        expected_args = expected.get("args") or {}
        for key, value in expected_args.items():
            if actual.args.get(key) != value:
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Buddy planner evals.")
    parser.add_argument("--eval-file", default="evals/planner_eval.json")
    parser.add_argument("--config", default="config/config_intel_optimized.yaml")
    args = parser.parse_args()

    runtime = PlannerRuntime.create_default(args.config)
    eval_rows = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
    passed = 0
    for row in eval_rows:
        plan = runtime.plan_request(
            row["user_message"],
            runtime_context=row.get("runtime_context") or {},
            chat_history=row.get("chat_history") or [],
        )
        got_tools = [step.tool for step in plan.steps]
        expected_tools = row.get("expected_tools")
        expected_steps = row.get("expected_steps")
        expect_clarification = bool(row.get("expect_clarification", False))
        expected_question_contains = row.get("expected_question_contains") or []

        ok = True
        detail = ""
        if expected_tools is not None:
            ok = got_tools == expected_tools
            detail = f"expected={expected_tools} got={got_tools}"
        elif expected_steps is not None:
            ok = _matches_expected_steps(plan.steps, expected_steps)
            detail = f"expected_steps={expected_steps} got={[{'tool': step.tool, 'args': step.args} for step in plan.steps]}"
        elif expect_clarification:
            ok = plan.needs_clarification and all(token.lower() in str(plan.clarification_question or "").lower() for token in expected_question_contains)
            detail = f"clarification={plan.clarification_question!r}"

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {row['id']}: {detail}")
        if ok:
            passed += 1
    print(f"{passed}/{len(eval_rows)} passed")
    return 0 if passed == len(eval_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
