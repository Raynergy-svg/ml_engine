"""Structural no-authority boundary for the hedge-evaluation evidence worker.

Roadmap §12 acceptance criteria:
    - Remote worker cannot read `.claude/state.json`.
    - Remote worker has no broker credentials.

Operationalized as deterministic, re-runnable checks (per L-005, "advisory !=
enforced"), not docstring promises:

1. AST import scan — the producer-side modules import nothing that can reach the
   broker path, the trade execution/gate path, or the state engine.
2. Source-literal scan — those modules contain no `.claude` / `state.json` /
   broker / OANDA path literals; the worker is handed bytes and has no
   repository-root or control-file parameter.
3. Transitive import closure — importing the producer modules in a fresh
   interpreter loads no broker / execution / gate / state-engine module even
   *through* their dependency on the analysis-only hedge scorecard.
4. Capability contract — the profile the worker runs under grants no authority.

NO MOCKS — pure static analysis plus a real capability-profile construction.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from src.evidence.hedge_eval.manifests import build_capability_profile

REPO_ROOT = Path(__file__).resolve().parent.parent

# The producer half: everything a remote worker runs. The local importer and
# slice orchestrator are local-authority code and are intentionally excluded.
WORKER_MODULES = [
    "src/evidence/hedge_eval/worker.py",
    "src/evidence/hedge_eval/evaluation.py",
    "src/evidence/hedge_eval/manifests.py",
    "src/evidence/hedge_eval/models.py",
]

FORBIDDEN_IMPORT_SUBSTRINGS = [
    "src.scanner",
    "src.brokers",
    "scanner.execution",
    "scanner.gates",
    "scanner.state_engine",
    "brokers.factory",
    "oanda_practice",
    "oanda.client",
    "state_engine",
]

# Operative control-plane / credential path literals a remote worker must never
# reference. Checked against *code* string constants only (docstrings, which
# explain the boundary, are excluded).
FORBIDDEN_OPERATIVE_LITERALS = [
    ".claude/state.json",
    "state.json",
    "oanda_environment",
    "OANDA_API_KEY",
    "OANDA_TOKEN",
    "api-fxpractice",
    "stream-fxpractice",
    "trained_data/models",
]


def _imported_module_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Ids of the string-constant nodes that are module/class/function docstrings."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _code_string_constants(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    docstrings = _docstring_nodes(tree)
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            values.append(node.value)
    return values


def test_worker_modules_import_no_broker_or_control_plane():
    violations = []
    for rel_path in WORKER_MODULES:
        path = REPO_ROOT / rel_path
        assert path.exists(), f"expected worker module missing: {rel_path}"
        for imported in _imported_module_names(path):
            for forbidden in FORBIDDEN_IMPORT_SUBSTRINGS:
                if forbidden in imported:
                    violations.append((rel_path, imported, forbidden))
    assert not violations, f"worker module imports a forbidden authority symbol: {violations}"


def test_worker_modules_contain_no_control_or_credential_literals():
    violations = []
    for rel_path in WORKER_MODULES:
        constants = _code_string_constants(REPO_ROOT / rel_path)
        for value in constants:
            lowered = value.lower()
            for literal in FORBIDDEN_OPERATIVE_LITERALS:
                if literal.lower() in lowered:
                    violations.append((rel_path, literal, value))
    assert not violations, f"worker code references a control/credential literal: {violations}"


def test_worker_transitive_import_closure_is_authority_free():
    """Stronger than the direct-import scan: import the producer modules in a
    fresh interpreter and assert the FULL transitive closure (sys.modules)
    contains no broker / execution / gate / state-engine module. This catches a
    future regression where a worker dep pulls one in *through* the hedge
    scorecard or any other dependency."""
    forbidden = [
        "src.scanner", "src.brokers", "scanner.execution", "scanner.gates",
        "state_engine", "oanda", "brokers.factory",
    ]
    code = (
        "import importlib, sys\n"
        "importlib.import_module('src.evidence.hedge_eval.worker')\n"
        "importlib.import_module('src.evidence.hedge_eval.evaluation')\n"
        "importlib.import_module('src.evidence.hedge_eval.manifests')\n"
        f"forbidden = {forbidden!r}\n"
        "bad = sorted(m for m in sys.modules if any(f in m for f in forbidden))\n"
        "print('\\n'.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"worker import failed: {result.stderr[-2000:]}"
    loaded_bad = [line for line in result.stdout.splitlines() if line.strip()]
    assert not loaded_bad, f"worker transitively imported forbidden modules: {loaded_bad}"


def test_capability_profile_grants_no_authority():
    profile = build_capability_profile()
    assert profile.may_read_broker_credentials is False
    assert profile.may_place_or_cancel_orders is False
    assert profile.may_read_operator_keys is False
    assert profile.may_change_halts is False
    assert profile.may_change_live_gate is False
    assert profile.may_write_champion_pointer is False
    assert profile.may_modify_local_models is False
    assert profile.may_approve_evidence is False
    assert profile.network_endpoints == ()
    # Read scope is confined to approved snapshots; nothing under .claude.
    assert all("snapshot" in scope for scope in profile.read_scopes)
    assert all(".claude" not in scope for scope in profile.read_scopes)
