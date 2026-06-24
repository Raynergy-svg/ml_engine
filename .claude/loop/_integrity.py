"""Shared gate-script integrity check — SHA-256 of every enforcement script vs a committed manifest.

Closes the red-team's #2 systemic gap: a worker editing verify_gate.py to `return 0`, or
risk_monitor.sh to `echo GREEN; exit 0`, or pointing the Stop hook at a stub. Any content drift in a
gate script is detected here and the gate fails closed.

Both verify_gate.py (full integrity check) and loop_gate.py (pre-decide HALT) call this, so neutering
ONE checker doesn't disable the manifest — the other still catches the drift. Irreducible floor
(documented): a worker who edits a gate script AND updates gate_manifest.json in the same change
evades content-hashing — but that is a git-visible, human-reviewable diff, and there is always a
root checker that cannot hash itself. Regenerate the manifest with gen_manifest.py on legit changes.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

MANIFEST_REL = ".claude/loop/gate_manifest.json"

# A lesson counts as real only if it has all five field LABELS, a non-trivial body LENGTH, and a
# unique title. This makes "add an empty ## L-099 to bump the lesson counter" fail closed
# (red-team #3): shallow/empty/dup lessons are problems (gate FAIL) and are NOT counted toward the
# lessons_count progress signal. NOTE: this enforces STRUCTURE, not MEANING — a well-formed but
# vacuous lesson still passes; semantic quality is the irreducible floor (human review). See L-009.
REQUIRED_LESSON_FIELDS = ("Trigger", "Root cause", "Rule", "Scope", "Source")
_MIN_LESSON_BODY = 140


def audit_lessons(text: str) -> tuple[int, list[str]]:
    """Return (well_formed_count, problems). Well-formed = '## L-NNN — <title>' with a unique
    non-empty title and a body containing all REQUIRED_LESSON_FIELDS and >= _MIN_LESSON_BODY chars."""
    blocks = re.split(r'(?m)^(?=##\s+L-\d+\b)', text)
    well = 0
    problems: list[str] = []
    seen: dict[str, str] = {}
    for b in blocks:
        m = re.match(r'##\s+(L-\d+)\s*[—-]\s*(.*)', b)
        if not m:
            continue
        lid, title = m.group(1), m.group(2).strip()
        body = b[m.end():]
        issues = []
        if not title:
            issues.append("empty-title")
        missing = [f for f in REQUIRED_LESSON_FIELDS if f.lower() not in body.lower()]
        if missing:
            issues.append("missing[" + "+".join(missing) + "]")
        if len(body.strip()) < _MIN_LESSON_BODY:
            issues.append("too-short")
        key = title.lower()
        if key and key in seen:
            issues.append(f"dup-of-{seen[key]}")
        elif key:
            seen[key] = lid
        if issues:
            problems.append(f"{lid}:{'/'.join(issues)}")
        else:
            well += 1
    return well, problems

# Canonical set of enforcement scripts that MUST be hash-pinned. Single source of truth, imported by
# gen_manifest.py. gate_drift cross-checks the manifest covers all of these, so dropping an entry to
# hide a neutered script is itself caught (fail-closed) rather than relying on manual diff review.
GATE_SCRIPTS = [
    ".claude/loop/verify_gate.py",
    ".claude/loop/loop_gate.py",
    ".claude/loop/record_cycle.py",
    ".claude/loop/_integrity.py",
    ".claude/loop/gen_manifest.py",
    ".claude/tools/risk_monitor.sh",
    ".claude/tools/stop_gate.sh",
    ".claude/tools/session_context_boot.sh",
]


def gate_drift(repo: Path) -> tuple[list[str], str | None]:
    """Return (drifted, error). ([], None) == every gate script is listed AND matches the manifest.

    drifted entries: 'path:unlisted' (canonical script missing from manifest — entry-drop attack),
    'path:missing' (file gone), 'path:modified' (hash drift). A missing/unreadable manifest is an
    error (fail-closed: the caller treats it as not-ok)."""
    man_path = repo / MANIFEST_REL
    try:
        man = json.loads(man_path.read_text())
    except Exception as e:
        return [], f"gate_manifest.json unreadable == cannot verify gate integrity ({e})"
    if not isinstance(man, dict) or not man:
        return [], "gate_manifest.json malformed/empty == cannot verify gate integrity"
    drift: list[str] = []
    for rel in GATE_SCRIPTS:  # coverage: every canonical gate must be pinned
        if rel not in man:
            drift.append(f"{rel}:unlisted")
    for rel, expected in man.items():
        p = repo / rel
        if not p.exists():
            drift.append(f"{rel}:missing")
        elif hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            drift.append(f"{rel}:modified")
    return drift, None
