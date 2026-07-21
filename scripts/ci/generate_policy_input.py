#!/usr/bin/env python3
"""Produce the PR-context JSON document that .ci/policy/rego evaluates.

CI-only tooling: reads git diff + PR labels + evidence files already on disk.
Never touches runtime state, config, or execution paths. See
.ci/policy/README.md for the full input schema this script must match.

Usage:
    python scripts/ci/generate_policy_input.py --base-ref <sha_or_ref> \
        --head-ref <sha_or_ref> --output /tmp/policy_input.json

Env var fallbacks (used when the matching --flag is omitted): POLICY_BASE_REF,
POLICY_HEAD_REF, POLICY_PR_NUMBER, POLICY_PR_LABELS (comma-separated).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Substrings looked for in ADDED diff lines (git diff lines starting with a
# single "+", excluding the "+++ b/path" file header) anywhere in the diff,
# regardless of which file they land in. Kept intentionally narrow — this is
# a tripwire for the specific Hard NOs in CLAUDE.md, not a general secret/lint
# scanner (gitleaks and flake8 already cover those in code-quality.yml).
RISKY_LIVE_DIFF_PATTERNS = [
    re.compile(r'oanda_environment\s*=\s*["\']live["\']'),
    re.compile(r"\barmed\s*=\s*True\b"),
    re.compile(r'"halted"\s*:\s*false', re.IGNORECASE),
    re.compile(r"\bLIVE_CONFIRMATION_TOKEN\b"),
    re.compile(r"\bunhalt\w*\s*\("),
]

LIVE_ARM_EVIDENCE_PREFIX = ".ci/policy/evidence/live-arm/"
MODEL_PROMOTION_EVIDENCE_PREFIX = ".ci/policy/evidence/model-promotion/"
TEMPLATE_BASENAME = "TEMPLATE.md"

# trained_data/backtests/ output files are timestamped by every producer in
# this repo (e.g. equity_harvester_singlestock_20260702T050603Z.json,
# crypto_cash_and_carry_20260707T015642Z.json). Requiring the stamp here
# stops an incidental touch of a stale/unrelated backtest artifact in that
# directory (rename, delete, formatting fix) from counting as "fresh
# eval-gate evidence" for a model-promotion change.
EVAL_REPORT_TIMESTAMP_RE = re.compile(r"\d{8}T\d{6}Z")


def run_git(args: list[str], repo_root: str = REPO_ROOT) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def changed_files(base_ref: str, head_ref: str, repo_root: str = REPO_ROOT) -> list[str]:
    out = run_git(["diff", "--name-only", base_ref, head_ref], repo_root)
    return [line.strip() for line in out.splitlines() if line.strip()]


def risky_diff_patterns_found(base_ref: str, head_ref: str, repo_root: str = REPO_ROOT) -> list[str]:
    diff_text = run_git(["diff", base_ref, head_ref], repo_root)
    added_lines = [
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    hits: list[str] = []
    for pattern in RISKY_LIVE_DIFF_PATTERNS:
        for line in added_lines:
            if pattern.search(line):
                hits.append(pattern.pattern)
                break
    return hits


def is_evidence_file(path: str, prefix: str) -> bool:
    if not path.startswith(prefix):
        return False
    return os.path.basename(path) != TEMPLATE_BASENAME


def build_input(
    base_ref: str,
    head_ref: str,
    pr_number: str,
    pr_labels: list[str],
    repo_root: str = REPO_ROOT,
) -> dict:
    files = changed_files(base_ref, head_ref, repo_root)

    live_arm_checklist_changed = any(
        is_evidence_file(f, LIVE_ARM_EVIDENCE_PREFIX) for f in files
    )
    ship_gate_files = [f for f in files if f.startswith("trained_data/backtests/") and "SHIP_GATE" in f]
    eval_report_files = [
        f
        for f in files
        if f.startswith("trained_data/backtests/")
        and f not in ship_gate_files
        and EVAL_REPORT_TIMESTAMP_RE.search(os.path.basename(f))
    ]

    return {
        "pr": {"number": pr_number, "base_ref": base_ref, "head_ref": head_ref},
        "changed_files": files,
        "diff_signals": {
            "risky_live_patterns_found": risky_diff_patterns_found(base_ref, head_ref, repo_root),
        },
        "review_evidence": {
            "checklist_file_changed": live_arm_checklist_changed,
            "labels": pr_labels,
        },
        "gate_evidence": {
            "ship_gate_files_changed": ship_gate_files,
            "eval_report_files_changed": eval_report_files,
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default=os.environ.get("POLICY_BASE_REF"))
    parser.add_argument("--head-ref", default=os.environ.get("POLICY_HEAD_REF", "HEAD"))
    parser.add_argument("--pr-number", default=os.environ.get("POLICY_PR_NUMBER", ""))
    parser.add_argument(
        "--pr-labels",
        default=os.environ.get("POLICY_PR_LABELS", ""),
        help="Comma-separated PR label names",
    )
    parser.add_argument("--output", default="-", help="Output path, or '-' for stdout")
    parser.add_argument(
        "--repo-root",
        default=REPO_ROOT,
        help="Git repo root to diff (default: this repo). Overridable for testing against a fixture repo.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.base_ref:
        print("error: --base-ref (or POLICY_BASE_REF) is required", file=sys.stderr)
        return 2

    labels = [label.strip() for label in args.pr_labels.split(",") if label.strip()]
    doc = build_input(args.base_ref, args.head_ref, args.pr_number, labels, args.repo_root)
    text = json.dumps(doc, indent=2, sort_keys=True)

    if args.output == "-":
        print(text)
    else:
        with open(args.output, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
