#!/usr/bin/env python3
"""Validate the workflow_run boundary PoC kit without network access."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION = Path("/tmp/claude-code-action-bounty")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require(condition: bool, message: str, failures: list[str]) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def validate_workflows(failures: list[str]) -> None:
    parent = read(ROOT / "workflows/01-parent-pr-ci.yml")
    child = read(ROOT / "workflows/02-child-workflow-run-claude.yml")

    require(
        "name: POC Parent PR CI" in parent,
        "parent workflow has expected name",
        failures,
    )
    require(
        'workflows: ["POC Parent PR CI"]' in child,
        "child workflow targets parent workflow by exact name",
        failures,
    )
    require(
        "vars.CLAUDE_WORKFLOW_RUN_POC_ENABLED == 'true'" in child,
        "child workflow has explicit repository-variable safety gate",
        failures,
    )
    require(
        "ANTHROPIC_API_KEY" in child,
        "child workflow uses normal Anthropic API secret placeholder",
        failures,
    )
    require(
        "POC_FAKE_MARKER" in child and "FAKE-H1-POC-MARKER" in child,
        "child workflow uses fake marker only",
        failures,
    )
    require(
        "BENIGN_WORKFLOW_RUN_BOUNDARY_POC" in parent
        and "BENIGN_WORKFLOW_RUN_BOUNDARY_POC" in child,
        "marker connects parent-controlled data to child Claude prompt",
        failures,
    )
    require(
        "Do not read or print real secrets" in parent
        and "Do not read, print, or copy any real secrets" in child,
        "workflows include explicit no-real-secrets safety language",
        failures,
    )
    require(
        "curl " not in child and "wget " not in child,
        "child workflow has no external exfiltration shell commands",
        failures,
    )


def validate_report(failures: list[str]) -> None:
    report = read(ROOT / "report-draft.md")
    evidence = read(ROOT / "evidence/source-findings.md")

    require(
        "Kill condition" in read(ROOT / "README.md"),
        "README contains kill condition",
        failures,
    )
    require(
        "This is not reportable if" in evidence,
        "evidence notes generic workflow_run risk is not enough",
        failures,
    )
    require(
        "<Fill after live test>" in report,
        "report draft keeps actual result as placeholder until live proof exists",
        failures,
    )
    require(
        "Suggested remediation" in report,
        "report includes remediation section",
        failures,
    )


def validate_action_source_if_present(failures: list[str]) -> None:
    if not ACTION.exists():
        print(f"[SKIP] local action source not found at {ACTION}")
        return

    run_ts = read(ACTION / "src/entrypoints/run.ts")
    context_ts = read(ACTION / "src/github/context.ts")
    agent_ts = read(ACTION / "src/modes/agent/index.ts")
    example = read(ACTION / "examples/ci-failure-auto-fix.yml")

    require(
        re.search(r"if\s*\(\s*isEntityContext\(context\)\s*\)", run_ts) is not None
        and "checkWritePermissions" in run_ts,
        "run.ts gates write-permission check on isEntityContext(context)",
        failures,
    )
    require(
        '"workflow_run"' in context_ts and "AUTOMATION_EVENT_NAMES" in context_ts,
        "context.ts classifies workflow_run as automation event",
        failures,
    )
    require(
        "checkHumanActor" in agent_ts and "checkWritePermissions" not in agent_ts,
        "agent mode checks human actor but not write permissions",
        failures,
    )
    require(
        "workflow_run:" in example
        and "downloadJobLogsForWorkflowRun" in example
        and "contents: write" in example,
        "ci-failure example combines workflow_run logs with write-capable permissions",
        failures,
    )


def main() -> int:
    failures: list[str] = []
    validate_workflows(failures)
    validate_report(failures)
    validate_action_source_if_present(failures)

    if failures:
        print("\nValidation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nAll PoC kit checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

