"""Structural boundary test for the risk-target model family (2026-07-08).

Operationalizes the hard constraint (docs/prereg-risk-target-vol-drawdown-
2026-07-08.md §3): the risk-target pipeline must NEVER import the trade
execution path, the gate/inference path, the state engine, or any broker
client — it is offline/research-only and must never enter the hot path.
Per L-005 ("advisory != enforced"), this is a deterministic, re-runnable
check, not just a docstring promise.

NO MOCKS — pure static analysis (AST) of the actual source files on disk.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RISK_TARGET_MODULES = [
    "src/training/labels/forward_drawdown_state_label.py",
    "src/training/labels/realized_volatility_regime_label.py",
    "src/training/risk_target_features.py",
    "src/training/risk_target_readout.py",
    "src/training/risk_target_splits.py",
    "src/training/trainers/risk_target_trainer.py",
    "cli/risk_target_training.py",
    "scripts/experiment_risk_target_vol_drawdown.py",
]

FORBIDDEN_IMPORT_SUBSTRINGS = [
    "scanner.execution",
    "scanner.gates",
    "scanner.state_engine",
    "brokers.factory",
    "oanda_practice",
    "oanda.client",
    "src.brokers",
]


def _imported_module_names(path: Path) -> list:
    tree = ast.parse(path.read_text(), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_no_hotpath_or_broker_imports_in_risk_target_modules():
    violations = []
    for rel_path in RISK_TARGET_MODULES:
        path = REPO_ROOT / rel_path
        assert path.exists(), f"expected risk-target module missing: {rel_path}"
        for imported in _imported_module_names(path):
            for forbidden in FORBIDDEN_IMPORT_SUBSTRINGS:
                if forbidden in imported:
                    violations.append((rel_path, imported, forbidden))
    assert not violations, f"risk-target module imports a hot-path/broker symbol: {violations}"


def test_realized_volatility_regime_label_still_has_no_broker_coupling():
    """The vol-regression function was ADDED to an existing, already-live
    (FX direction ensemble) label module — confirm the addition did not
    introduce a new import at the module level that would couple this
    file to the hot path."""
    path = REPO_ROOT / "src/training/labels/realized_volatility_regime_label.py"
    for imported in _imported_module_names(path):
        for forbidden in FORBIDDEN_IMPORT_SUBSTRINGS:
            assert forbidden not in imported, f"unexpected hot-path import: {imported}"


def test_risk_target_trainer_does_not_import_live_fx_hotpath_lgbm_classes():
    """RiskTargetTrainer must not subclass or import LightGBMRiskTrainer/
    LightGBMMomentumTrainer (lightgbm_trainers.py) — those classes are
    wired into the live FX direction ensemble's hot path via
    src/core/modular_inference.py. Sharing an import edge with them would
    be a structural coupling risk even if unused at runtime."""
    path = REPO_ROOT / "src/training/trainers/risk_target_trainer.py"
    imported = _imported_module_names(path)
    assert not any("lightgbm_trainers" in name for name in imported)
