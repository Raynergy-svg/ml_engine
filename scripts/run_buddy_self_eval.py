#!/usr/bin/env python3
"""Run Buddy self-knowledge eval prompts against the grounded answer path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.buddy_knowledge import answer_buddy_question


def _load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Eval file must contain a JSON array")
    return [case for case in data if isinstance(case, dict)]


def _score_case(answer: str, case: dict[str, Any]) -> tuple[bool, list[str]]:
    answer_lower = answer.lower()
    failures: list[str] = []

    for needle in case.get("expected_contains", []):
        if str(needle).lower() not in answer_lower:
            failures.append(f"missing '{needle}'")

    expected_any = case.get("expected_any_contains", [])
    if expected_any:
        if not any(str(needle).lower() in answer_lower for needle in expected_any):
            failures.append(
                "missing any of {" + ", ".join(repr(str(needle)) for needle in expected_any) + "}"
            )

    for needle in case.get("forbidden_contains", []):
        if str(needle).lower() in answer_lower:
            failures.append(f"forbidden '{needle}'")

    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Buddy self-knowledge benchmark prompts.")
    parser.add_argument(
        "--eval-file",
        default="evals/buddy_self_knowledge_eval.json",
        help="Path to eval JSON file.",
    )
    parser.add_argument(
        "--config",
        default="config/config_intel_optimized.yaml",
        help="Buddy config path used for grounded lookups.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM summarization on top of grounded facts instead of deterministic summaries.",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    cases = _load_cases(eval_path)

    passed = 0
    for case in cases:
        prompt = str(case.get("prompt", "")).strip()
        answer = answer_buddy_question(
            prompt,
            config_path=args.config,
            use_llm=bool(args.use_llm),
        )
        ok, failures = _score_case(answer, case)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case.get('id', 'unknown')} :: {prompt}")
        if not ok:
            for failure in failures:
                print(f"  - {failure}")
            print("  Answer:")
            for line in answer.splitlines():
                print(f"    {line}")
        passed += int(ok)

    total = len(cases)
    print(f"\nSummary: {passed}/{total} passed")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
