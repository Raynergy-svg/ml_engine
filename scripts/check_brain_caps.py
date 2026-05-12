#!/usr/bin/env python3
"""CLI for the brain-caps check (pick #6).

Exits:
    0 — no files over cap (or .claude/brain/ missing)
    0 with output — files over cap, exit code controlled by --strict
    1 — under --strict mode, any cap overage is a failure (for CI)

Examples:
    python scripts/check_brain_caps.py            # informational
    python scripts/check_brain_caps.py --strict   # fail CI on overage
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo-relative import: this script must run from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.scanner.automation.brain_caps import check_brain_caps, format_warnings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when any file is over its warn threshold (for CI).",
    )
    args = ap.parse_args()

    warnings = check_brain_caps()
    if not warnings:
        print("OK: no .claude/brain/*.md files exceed cap.")
        return 0

    print(format_warnings(warnings))
    if args.strict:
        print(
            "\nRun under --strict — failing. Trim the listed files or "
            "archive into .claude/brain/.archive/, then re-run.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
