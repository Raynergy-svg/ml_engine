#!/usr/bin/env python3
"""ml_engine_gate_wrapper.py — the ROOT-OWNED managed Stop-hook gate (the trust anchor).

Installed to the system managed dir (macOS /Library/Application Support/ClaudeCode/, Linux
/etc/claude-code/, Windows C:\\Program Files\\ClaudeCode\\) and wired as a Stop hook in
managed-settings.json. Managed settings sit ABOVE local .claude/settings.json in precedence and
"can't be overridden by anything" (verified against code.claude.com/docs/en/settings, 2026-06-23).
Claude Code's write access is confined to the project tree, so the worker cannot edit or remove this
root-owned file/hook without admin — which the operator controls.

SELF-CONTAINED ON PURPOSE: it re-derives the gate-script SHA-256 hashes itself (it does NOT call the
in-repo checkers, which a worker could neuter), using a baked-in canonical script list + the in-repo
manifest, and only then trusts the in-repo risk_monitor. TARGET_REPO is baked (NOT env-overridable),
so a worker can't point it at a clean fixture. Scoped: a no-op for any other project so it does not
interfere system-wide.

Exit 0 = allow the turn to stop. Exit 2 = BLOCK (Claude Code keeps the turn going; stderr shown to
Claude). Loop-guarded via stop_hook_active so it blocks at most once and never traps.

Residual floor (documented): editing an in-repo gate script AND its in-repo manifest together is
git-visible; baking the manifest hash into THIS root-owned file (requiring re-install on gate
changes) would close that too. See INSTALL.md.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

TARGET_REPO = Path("/Users/buddy/Documents/ml_engine")  # baked; not env-overridable (no bypass)

# Canonical pinned set — kept in lockstep with .claude/loop/_integrity.py GATE_SCRIPTS. Baked here
# (root-owned) so manifest entry-drop is caught even if the in-repo copy is edited. NOTE: changing
# the pinned set requires re-running the install step (see INSTALL.md).
GATE_SCRIPTS = [
    ".claude/loop/verify_gate.py",
    ".claude/loop/loop_gate.py",
    ".claude/loop/record_cycle.py",
    ".claude/loop/record_verdict.py",
    ".claude/loop/_integrity.py",
    ".claude/loop/gen_manifest.py",
    ".claude/loop/tests/test_loop_enforcement.py",
    ".claude/tools/risk_monitor.sh",
    ".claude/tools/stop_gate.sh",
    ".claude/tools/session_context_boot.sh",
]


def run_gate(target: Path, data: dict) -> tuple[int, str]:
    """Return (exit_code, message). 0 = allow, 2 = block. Pure/testable (no sys.exit, no stdin)."""
    if data.get("stop_hook_active") is True:
        return 0, "loop guard (stop_hook_active) — never trap"
    cwd = Path(data.get("cwd") or os.getcwd())
    try:
        in_scope = cwd == target or target in cwd.parents
    except Exception:
        in_scope = False
    if not in_scope:
        return 0, "out of scope (no-op for other projects)"

    man_path = target / ".claude/loop/gate_manifest.json"
    try:
        man = json.loads(man_path.read_text())
    except Exception as e:
        return 2, f"gate_manifest.json unreadable == unsafe ({e})"
    if not isinstance(man, dict) or not man:
        return 2, "gate_manifest.json malformed/empty == unsafe"
    drift: list[str] = []
    for rel in GATE_SCRIPTS:  # coverage: a dropped pin can't hide a neutered script
        if rel not in man:
            drift.append(f"{rel}:unlisted")
    for rel, expected in man.items():
        p = target / rel
        if not p.exists():
            drift.append(f"{rel}:missing")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            drift.append(f"{rel}:modified")
    if drift:
        return 2, f"gate-script drift (neutered/altered enforcement scripts): {drift}"

    mon = target / ".claude/tools/risk_monitor.sh"
    # absolute /bin/bash so the worker can't shadow `bash` via PATH; fail-closed if absent.
    bash_bin = "/bin/bash" if Path("/bin/bash").exists() else "bash"
    try:
        r = subprocess.run([bash_bin, str(mon)], capture_output=True, text=True, timeout=60,
                           env={**os.environ, "RISK_MONITOR_REPO": str(target)})
    except Exception as e:
        return 2, f"risk_monitor failed == unsafe ({e})"
    if r.returncode != 0:
        return 2, "risk monitor ALARM:\n" + (r.stdout or "")[-1000:]
    return 0, "gate scripts intact (hash-verified), risk monitor GREEN"


def main() -> int:
    raw = ""
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
    except Exception:
        raw = ""
    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {}
    code, msg = run_gate(TARGET_REPO, data)
    if code != 0:
        print(f"MANAGED GATE: HALT-SAFETY — {msg}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
