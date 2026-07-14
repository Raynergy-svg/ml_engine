from __future__ import annotations

import ast
from pathlib import Path

from src.evidence.crypto_momentum.manifests import build_capability_profile

ROOT = Path(__file__).resolve().parents[1]
PRODUCER_FILES = (
    ROOT / "src/evidence/crypto_momentum/evaluation.py",
    ROOT / "src/evidence/crypto_momentum/manifests.py",
    ROOT / "src/evidence/crypto_momentum/models.py",
    ROOT / "src/evidence/crypto_momentum/worker.py",
    ROOT / "src/crypto/momentum_scorecard.py",
)


def test_capability_profile_is_authority_free():
    profile = build_capability_profile()
    assert profile.network_endpoints == ()
    for field in (
        "may_read_broker_credentials",
        "may_place_or_cancel_orders",
        "may_read_operator_keys",
        "may_change_halts",
        "may_change_live_gate",
        "may_write_champion_pointer",
        "may_modify_local_models",
        "may_approve_evidence",
    ):
        assert getattr(profile, field) is False


def test_producer_import_graph_excludes_execution_and_local_authority():
    forbidden = (
        "src.brokers",
        "src.scanner.execution",
        "src.scanner.automation.state_engine",
        "src.evidence.store",
        "src.evidence.transition_policy",
        "src.evidence.crypto_momentum.local_import",
    )
    for path in PRODUCER_FILES:
        tree = ast.parse(path.read_text(), filename=str(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not [name for name in imported if name.startswith(forbidden)], path


def test_worker_has_no_promotion_or_order_api():
    source = (ROOT / "src/evidence/crypto_momentum/worker.py").read_text()
    for forbidden_call in ("place_order(", "cancel_order(", "promote(", "set_champion("):
        assert forbidden_call not in source
