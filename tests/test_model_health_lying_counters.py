"""Regression guard: model health flags reflect actual attribute state.

Mythos audit 2026-05-04 — operator pasted a brain-feed snapshot where:

    ▸ Model health: 13/14 loaded (momentum: lightgbm)
    ✗ momentum_lightgbm        ← contradicts the line above
    ✗ transformer_direction    ← debug log showed it DID load
    ✓ tcn_volatility           ← actually checked the direction attribute

Two distinct lying counters in two files:

1. gates.py:1914-1917 (GateEvaluator.get_model_health):
   `momentum_lightgbm: self._momentum_model_type == "lightgbm"
                       and self._catboost_momentum is not None`
   But _load_lgbm_momentum (line 902) stores into self._xgboost_momentum,
   not self._catboost_momentum. So the flag was always False even when
   lightgbm WAS the active momentum model.

2. engine.py:6838-6839 (Scanner.get_model_health, modular ensemble group):
   `transformer_direction: getattr(mei, "transformer", None)`  ← attr name
   `tcn_volatility:        getattr(mei, "tcn", None)`           ← attr name
   The actual ModularEnsembleInference attributes are `mei.tcn` (direction
   model — legacy slot name) and `mei.tcn_volatility_model` (volatility
   classifier). The pre-fix checks were swapped: direction looked for
   "transformer" (doesn't exist), volatility looked at "tcn" (actually
   the direction model).

NO MOCKS. Tests use real GateEvaluator instances with a tmp_path
model_dir (no model files needed — we drive the internal attributes
directly to model the post-load state) and real ModularEnsembleInference
attribute layout assertions.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.scanner.gates import GateEvaluator


def _make_evaluator(tmp_path: Path) -> GateEvaluator:
    """Real GateEvaluator with empty model_dir — no models loaded."""
    return GateEvaluator(model_dir=str(tmp_path))


def test_momentum_lightgbm_flag_tracks_xgboost_slot(tmp_path: Path) -> None:
    """When _load_lgbm_momentum stores into _xgboost_momentum and sets
    _momentum_model_type='lightgbm', the health flag must read True.

    The pre-fix check `_catboost_momentum is not None` was wrong because
    _load_lgbm_momentum never touches _catboost_momentum.
    """
    ge = _make_evaluator(tmp_path)
    # Simulate post-load state from _load_lgbm_momentum (real attribute
    # writes — no mock library, just direct assignment that matches the
    # actual loader's behavior at gates.py:902-903).
    ge._xgboost_momentum = object()  # any non-None placeholder
    ge._momentum_model_type = "lightgbm"

    health = ge.get_model_health()

    assert health["momentum_lightgbm"] is True, (
        f"momentum_lightgbm reported False even though "
        f"_xgboost_momentum is not None and _momentum_model_type='lightgbm'. "
        f"Pre-fix bug: the check looked at _catboost_momentum which is None. "
        f"Health snapshot: {health}"
    )
    assert health["momentum_xgboost"] is False, (
        "momentum_xgboost should be False when type is 'lightgbm' "
        "(only one cascade slot at a time)"
    )
    assert health["momentum_catboost"] is False


def test_momentum_xgboost_flag_tracks_xgboost_slot(tmp_path: Path) -> None:
    """The xgboost cascade also stores into _xgboost_momentum — but with
    type='xgboost'. The flag must distinguish the two cascade modes via
    _momentum_model_type.
    """
    ge = _make_evaluator(tmp_path)
    ge._xgboost_momentum = object()
    ge._momentum_model_type = "xgboost"

    health = ge.get_model_health()

    assert health["momentum_xgboost"] is True
    assert health["momentum_lightgbm"] is False
    assert health["momentum_catboost"] is False


def test_momentum_catboost_flag_tracks_catboost_slot(tmp_path: Path) -> None:
    """CatBoost is the primary momentum cascade — uses its own slot."""
    ge = _make_evaluator(tmp_path)
    ge._catboost_momentum = object()
    ge._momentum_model_type = "catboost"

    health = ge.get_model_health()

    assert health["momentum_catboost"] is True
    assert health["momentum_xgboost"] is False
    assert health["momentum_lightgbm"] is False


def test_no_momentum_model_loaded_reports_all_false(tmp_path: Path) -> None:
    """Negative control: nothing loaded → all three momentum flags False."""
    ge = _make_evaluator(tmp_path)
    health = ge.get_model_health()

    assert health["momentum_catboost"] is False
    assert health["momentum_xgboost"] is False
    assert health["momentum_lightgbm"] is False


def test_modular_ensemble_attribute_layout_unchanged() -> None:
    """The fix in engine.py:6838-6839 depends on the ModularEnsembleInference
    storing the direction model in `self.tcn` (legacy slot name) and the
    volatility classifier in `self.tcn_volatility_model`. If a future
    refactor renames either, this test catches it.

    We don't need to instantiate a real ModularEnsembleInference (heavy
    deps); reading the source is enough to prove the attribute names
    are present in the code.
    """
    src_path = Path("src/core/modular_inference.py")
    text = src_path.read_text(encoding="utf-8")

    # Direction model slot: line ~1609 `self.tcn = TransformerDirectionTrainer()`
    assert "self.tcn = TransformerDirectionTrainer()" in text, (
        "ModularEnsembleInference no longer stores the direction model in "
        "self.tcn — engine.py:6838's `getattr(mei, 'tcn')` check will go "
        "stale. Update the health-check attribute name."
    )
    # Volatility model slot: line ~1584 `self.tcn_volatility_model = None`
    assert "self.tcn_volatility_model" in text, (
        "ModularEnsembleInference no longer has self.tcn_volatility_model — "
        "engine.py:6839's `getattr(mei, 'tcn_volatility_model')` check is "
        "stale. Update the health-check attribute name."
    )
