#!/usr/bin/env python3
"""verify_managed_anchor.py — confirm the managed trust anchor is installed AND un-tamperable.

Run AFTER the operator's one privileged install step (no sudo needed to verify). Checks FROM DISK:
  1. managed-settings.json + the wrapper exist at the SYSTEM managed dir for this OS.
  2. managed-settings.json is valid and wires the Stop hook to the wrapper.
  3. THE LOAD-BEARING PROPERTY: neither file is writable by the CURRENT (worker) user — i.e. they are
     root-owned, so the worker cannot edit or replace them. This is what makes managed settings
     un-overridable in practice (combined with the documented precedence: Managed > CLI > Local >
     Project > User; "Managed - can't be overridden by anything", code.claude.com/docs/en/settings).
Exit 0 = anchor active + un-tamperable. Exit 1 = not installed or not secure (worker could tamper).
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
from pathlib import Path

WRAPPER_NAME = "ml_engine_gate_wrapper.py"


def managed_dir() -> Path:
    s = platform.system()
    if s == "Darwin":
        return Path("/Library/Application Support/ClaudeCode")
    if s == "Windows":
        return Path(r"C:\Program Files\ClaudeCode")  # NOT C:\ProgramData (deprecated v2.1.75)
    return Path("/etc/claude-code")  # Linux / WSL


def audit(d: Path) -> tuple[bool, list[str], dict]:
    """Return (ok, problems, info). Testable against any directory."""
    ms = d / "managed-settings.json"
    wr = d / WRAPPER_NAME
    problems: list[str] = []
    info = {"dir": str(d), "managed_settings": str(ms), "wrapper": str(wr)}

    if not ms.exists():
        problems.append(f"{ms} NOT installed (run the privileged install step)")
    else:
        try:
            j = json.loads(ms.read_text())
            cmds = [h.get("command", "")
                    for grp in j.get("hooks", {}).get("Stop", []) for h in grp.get("hooks", [])]
            if not any(WRAPPER_NAME in c for c in cmds):
                problems.append("managed-settings.json Stop hook does not wire the wrapper")
        except Exception as e:
            problems.append(f"managed-settings.json invalid JSON ({e})")
        # un-tamperable property: worker (current user) must NOT be able to write it
        if os.access(ms, os.W_OK):
            problems.append(f"{ms} is WRITABLE by the current user — NOT un-tamperable "
                            "(must be root-owned; re-run install with sudo)")

    if not wr.exists():
        problems.append(f"{wr} NOT installed")
    elif os.access(wr, os.W_OK):
        problems.append(f"{wr} is WRITABLE by the current user — NOT un-tamperable (must be root-owned)")

    return (not problems), problems, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="override managed dir (for testing)")
    a = ap.parse_args()
    d = Path(a.dir) if a.dir else managed_dir()
    ok, problems, info = audit(d)
    if ok:
        print(f"MANAGED ANCHOR: ACTIVE — {info['managed_settings']} present, wires the wrapper, and is "
              "NOT user-writable (root-owned/un-tamperable).")
        print("Precedence: Managed > CLI > Local > Project > User — local .claude/settings.json cannot "
              "override it (code.claude.com/docs/en/settings). The worker cannot disable the Stop hook.")
        return 0
    print("MANAGED ANCHOR: NOT ACTIVE / NOT SECURE:")
    for p in problems:
        print(f"  ✗ {p}")
    print("Until installed, the in-repo gates + code-level guards remain primary (advisory anchor).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
