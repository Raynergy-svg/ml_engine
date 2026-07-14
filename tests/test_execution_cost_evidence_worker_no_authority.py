from __future__ import annotations

import ast
from pathlib import Path

from src.evidence.execution_cost.manifests import build_capability_profile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "src.evidence.execution_cost"
PRODUCER_FILES = (
    ROOT / "src/evidence/execution_cost/evaluation.py",
    ROOT / "src/evidence/execution_cost/manifests.py",
    ROOT / "src/evidence/execution_cost/models.py",
    ROOT / "src/evidence/execution_cost/worker.py",
)


def test_capability_profile_is_authority_free():
    profile = build_capability_profile()
    assert profile.network_endpoints == ()
    for field in (
        "may_read_broker_credentials", "may_place_or_cancel_orders",
        "may_read_operator_keys", "may_change_halts", "may_change_live_gate",
        "may_write_champion_pointer", "may_modify_local_models", "may_approve_evidence",
    ):
        assert getattr(profile, field) is False


def test_producer_import_graph_excludes_execution_and_local_authority():
    forbidden = (
        "src.brokers", "src.scanner.execution", "src.trading.execution",
        "src.scanner.automation.state_engine", "src.evidence.store",
        "src.evidence.transition_policy", "src.evidence.execution_cost.local_import",
    )
    for path in PRODUCER_FILES:
        tree = ast.parse(path.read_text(), filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = PACKAGE.split(".")
                    base = ".".join(parts[: len(parts) - node.level + 1])
                    module = f"{base}.{node.module}" if node.module else base
                else:
                    module = node.module
                if module:
                    imported.append(module)
                    imported.extend(f"{module}.{alias.name}" for alias in node.names)
        assert not [name for name in imported if name.startswith(forbidden)], path


def test_worker_has_no_order_install_or_promotion_api():
    source = (ROOT / "src/evidence/execution_cost/worker.py").read_text()
    for forbidden_call in (
        "place_order(", "cancel_order(", "promote(", "set_champion(",
        "pickle.loads(", "torch.load(", "joblib.load(",
    ):
        assert forbidden_call not in source


def test_dashboard_reader_imports_stdlib_only():
    path = ROOT / "src/evidence/execution_cost/dashboard.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert set(imports) <= {"__future__", "json", "pathlib"}
