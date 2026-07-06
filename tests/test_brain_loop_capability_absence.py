"""Capability-ABSENCE tests for src/brain_loop — the structural half of the
hard constraint "the brain never places a trade and never arms/bypasses the
human checkpoint for flatten / set_gross_leverage / start_loop / unhalt."

Scoped per L-015 (a whole-text string scan false-positives on docs/comments
that legitimately NAME the danger to explain the boundary — several
docstrings in this very package say "never calls flatten/start_loop/unhalt").
These checks match structural CALL forms (``name(``) and import paths, not
mere word mentions, so the modules can keep documenting what they refuse to
do without tripping their own guard test.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BRAIN_LOOP_FILES = sorted((REPO_ROOT / "src" / "brain_loop").glob("*.py")) + [
    REPO_ROOT / "scripts" / "run_brain_loop.py"
]

# Structural call forms — a real invocation, not a docstring/comment mention.
FORBIDDEN_CALL_PATTERNS = {
    "flatten(": re.compile(r"\bflatten\s*\("),
    "set_gross_leverage(": re.compile(r"\bset_gross_leverage\s*\("),
    "start_loop(": re.compile(r"\bstart_loop\s*\("),
    "unhalt(": re.compile(r"\bunhalt\s*\("),
    "set_halted(False": re.compile(r"set_halted\s*\(\s*False"),
    "set_arm_state(": re.compile(r"set_arm_state\s*\("),
    "arm(": re.compile(r"\barm\s*\("),
    "disarm(": re.compile(r"\bdisarm\s*\("),
}

# Import paths that would give brain_loop trade/broker/control-action reach.
FORBIDDEN_IMPORT_SUBSTRINGS = (
    "src.scanner.execution",
    "src.brokers",
    "dashboard.server.control",  # the HTTP action handlers (control_safety guards are fine to read)
    "oanda_v20",
    "oanda_practice",
)

FORBIDDEN_CREDENTIAL_TOKENS = (
    "OANDA_API_TOKEN",
    "OANDA_ACCOUNT_ID",
    "ANTHROPIC_API_KEY",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_brain_loop_package_exists_and_has_expected_modules():
    names = {p.name for p in (REPO_ROOT / "src" / "brain_loop").glob("*.py")}
    assert {
        "__init__.py",
        "hypothesis_registry.py",
        "gate_runner.py",
        "promotion.py",
        "monitor.py",
        "derisk.py",
        "cycle.py",
    } <= names


def test_no_forbidden_control_action_calls():
    violations = []
    for path in BRAIN_LOOP_FILES:
        source = _read(path)
        for label, pattern in FORBIDDEN_CALL_PATTERNS.items():
            if pattern.search(source):
                violations.append(f"{path.relative_to(REPO_ROOT)}: {label}")
    assert violations == [], f"forbidden control-action call forms found: {violations}"


_IMPORT_LINE = re.compile(r"^\s*(import|from)\s+\S", re.MULTILINE)


def test_no_execution_broker_or_control_action_imports():
    """Scoped to actual `import`/`from` statement lines (L-015: a whole-text
    scan would false-positive on this very module's docstrings, which name
    these paths to document that they are NOT imported)."""
    violations = []
    for path in BRAIN_LOOP_FILES:
        for line in _read(path).splitlines():
            if not _IMPORT_LINE.match(line):
                continue
            for needle in FORBIDDEN_IMPORT_SUBSTRINGS:
                if needle in line:
                    violations.append(f"{path.relative_to(REPO_ROOT)}: {line.strip()!r}")
    assert violations == [], f"forbidden imports found: {violations}"


def test_no_direct_oanda_or_anthropic_credential_access():
    violations = []
    for path in BRAIN_LOOP_FILES:
        source = _read(path)
        for needle in FORBIDDEN_CREDENTIAL_TOKENS:
            if needle in source:
                violations.append(f"{path.relative_to(REPO_ROOT)}: {needle!r}")
    assert violations == [], f"forbidden credential access found: {violations}"


def test_no_requests_import_no_direct_network_calls():
    """brain_loop never talks to a broker/API directly — it only reads/writes
    local disk state and shells out to the (also-local) backtest harness."""
    violations = []
    for path in BRAIN_LOOP_FILES:
        source = _read(path)
        if re.search(r"^\s*(import requests|from requests)", source, re.MULTILINE):
            violations.append(str(path.relative_to(REPO_ROOT)))
    assert violations == [], f"unexpected network-library import found: {violations}"


def test_derisk_is_the_only_module_that_imports_state_engine():
    """Only derisk.py (and cycle.py, which reads-only via get_halted) may
    touch StateEngine — hypothesis_registry / gate_runner / promotion /
    monitor must never be able to mutate halt state directly."""
    allowed = {"derisk.py", "cycle.py"}
    violations = []
    for path in BRAIN_LOOP_FILES:
        if path.name in allowed or path.name == "run_brain_loop.py":
            continue
        source = _read(path)
        if "state_engine" in source.lower():
            violations.append(str(path.relative_to(REPO_ROOT)))
    assert violations == [], f"unexpected StateEngine reference found: {violations}"


def test_only_shadow_target_is_ever_auto_proposed_outside_promotion_module():
    """Every propose_promotion call site OUTSIDE promotion.py itself must use
    target='shadow' — a live promotion is only ever written by a human/
    external decision, never auto-triggered by brain_loop's own orchestration."""
    violations = []
    for path in BRAIN_LOOP_FILES:
        if path.name == "promotion.py":
            continue
        source = _read(path)
        if "propose_promotion(" in source and 'target="live"' in source:
            violations.append(str(path.relative_to(REPO_ROOT)))
    assert violations == [], f"autonomous live-promotion call site found: {violations}"
