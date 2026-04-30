"""Mythos audit 2026-04-30 — Tier 7 per-pair gate routing.

Pre-fix architecture (Tier 1.5):
    Scanner constructed ONE GateEvaluator pointed at trained_data/models/joint/.
    Every pair scanned used the same gate models, regardless of whether
    per-pair correlation-transferred models existed for that pair. Disk
    evidence on 04-30: joint/ files mtime Apr 28; per-pair EUR_USD/ files
    mtime Apr 30 02:07 (today's retrain). Two generations of models judging
    the same trade — scanner gates on stale joint, inference reads fresh
    per-pair.

The Tier 7 fix:
    GateEvaluator gains a `use_per_pair_routing` flag. When enabled AND
    instrument is passed AND a per-pair directory exists for that
    instrument, evaluate_all_gates delegates to a lazy-built sub-evaluator
    that loads the pair's correlation-transferred models. Sub-evaluators
    share the parent's TCN volatility regime (universal). When no per-pair
    dir exists for an instrument (e.g. USD_JPY today, dropped by the
    correlation threshold), falls back to self/joint — joint is the
    safety net, not the primary path.

The new flag is orthogonal to use_joint_only — old callers see no change
(test_gate_evaluator_core's existing test_non_joint_mode still pins
model_dir==base_model_dir).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scanner.gates import GateEvaluator


def _make_models_tree(tmp_path) -> Path:
    """Build a minimal model-tree skeleton:
        <tmp>/joint/  — empty dir (joint fallback exists)
        <tmp>/EUR_USD/ — empty dir (per-pair exists for EUR_USD only)
        no USD_JPY dir (must fall back to joint)
    """
    base = tmp_path / "models"
    (base / "joint").mkdir(parents=True)
    (base / "EUR_USD").mkdir()
    return base


# --- Init flag semantics --------------------------------------------------


def test_default_use_per_pair_routing_is_false_for_backward_compat(tmp_path):
    base = _make_models_tree(tmp_path)
    ev = GateEvaluator(base, use_joint_only=True)
    assert ev.use_per_pair_routing is False


def test_use_per_pair_routing_orthogonal_to_use_joint_only(tmp_path):
    """The new flag must NOT affect model_dir resolution. The existing
    test_non_joint_mode contract (model_dir == base_model_dir when
    use_joint_only=False) must still hold."""
    base = _make_models_tree(tmp_path)
    ev = GateEvaluator(base, use_joint_only=False, use_per_pair_routing=True)
    assert ev.model_dir == ev.base_model_dir


def test_use_per_pair_routing_can_be_enabled(tmp_path):
    base = _make_models_tree(tmp_path)
    ev = GateEvaluator(base, use_joint_only=True, use_per_pair_routing=True)
    assert ev.use_per_pair_routing is True


# --- _get_pair_evaluator routing ------------------------------------------


def test_routing_returns_self_when_instrument_missing(tmp_path):
    base = _make_models_tree(tmp_path)
    ev = GateEvaluator(base, use_per_pair_routing=True)
    assert ev._get_pair_evaluator(None) is ev
    assert ev._get_pair_evaluator("") is ev


def test_routing_returns_self_when_routing_disabled(tmp_path):
    base = _make_models_tree(tmp_path)
    ev = GateEvaluator(base, use_per_pair_routing=False)
    assert ev._get_pair_evaluator("EUR_USD") is ev


def test_routing_returns_self_when_no_per_pair_dir(tmp_path):
    """USD_JPY directory doesn't exist — must fall back to self
    (joint/global). This is the legitimate behavior for low-correlation
    pairs the master training drops."""
    base = _make_models_tree(tmp_path)
    ev = GateEvaluator(base, use_per_pair_routing=True)
    assert ev._get_pair_evaluator("USD_JPY") is ev


def test_routing_caches_no_per_pair_result(tmp_path):
    """Filesystem checks are cached — repeated calls for the same
    instrument-without-per-pair don't hit the disk again."""
    base = _make_models_tree(tmp_path)
    ev = GateEvaluator(base, use_per_pair_routing=True)
    ev._get_pair_evaluator("USD_JPY")
    assert "USD_JPY" in ev._pair_evaluators
    assert ev._pair_evaluators["USD_JPY"] is ev


def test_routing_builds_sub_evaluator_when_per_pair_dir_exists(tmp_path):
    """EUR_USD dir exists — must build a NEW evaluator (not self)."""
    base = _make_models_tree(tmp_path)
    ev = GateEvaluator(base, use_per_pair_routing=True)
    sub = ev._get_pair_evaluator("EUR_USD")
    assert sub is not ev
    assert sub.model_dir == base / "EUR_USD"


def test_sub_evaluator_recursion_disabled(tmp_path):
    """Sub-evaluators have routing OFF in their internal cache so they
    can never re-route — prevents infinite recursion."""
    base = _make_models_tree(tmp_path)
    ev = GateEvaluator(base, use_per_pair_routing=True)
    sub = ev._get_pair_evaluator("EUR_USD")
    assert sub._pair_evaluators == {}


def test_sub_evaluator_inherits_parent_tcn(tmp_path):
    """TCN volatility regime is universal — sub-evaluator must share
    the parent's TCN model rather than try to load its own (per-pair
    dirs don't carry TCN files)."""
    base = _make_models_tree(tmp_path)
    parent = GateEvaluator(base, use_per_pair_routing=True)
    parent._tcn_volatility = MagicMock(name="shared_tcn")
    parent._tcn_scaler = MagicMock(name="shared_scaler")
    parent._tcn_feature_names = ["foo", "bar"]
    sub = parent._get_pair_evaluator("EUR_USD")
    assert sub._tcn_volatility is parent._tcn_volatility
    assert sub._tcn_scaler is parent._tcn_scaler
    assert sub._tcn_feature_names == ["foo", "bar"]


def test_routing_repeated_call_returns_cached_sub(tmp_path):
    base = _make_models_tree(tmp_path)
    ev = GateEvaluator(base, use_per_pair_routing=True)
    sub1 = ev._get_pair_evaluator("EUR_USD")
    sub2 = ev._get_pair_evaluator("EUR_USD")
    assert sub1 is sub2


# --- evaluate_all_gates delegation ----------------------------------------


def test_evaluate_all_gates_delegates_when_routing_enabled(tmp_path):
    """The full delegation path: evaluate_all_gates sees routing
    enabled + instrument set + per-pair dir exists → calls
    sub.evaluate_all_gates with instrument=None."""
    base = _make_models_tree(tmp_path)
    ev = GateEvaluator(base, use_per_pair_routing=True)
    fake_sub = MagicMock(spec=GateEvaluator)
    fake_sub.evaluate_all_gates.return_value = {"delegated": True}
    ev._pair_evaluators["EUR_USD"] = fake_sub

    import pandas as pd
    result = ev.evaluate_all_gates(
        pd.DataFrame({"close": [1.0, 1.1, 1.2]}),
        instrument="EUR_USD",
    )
    assert result == {"delegated": True}
    fake_sub.evaluate_all_gates.assert_called_once()
    # Critical: instrument=None passed to the sub so the sub doesn't
    # re-route recursively.
    _, kwargs = fake_sub.evaluate_all_gates.call_args
    assert kwargs.get("instrument") is None


def test_evaluate_all_gates_no_delegation_without_routing_flag(tmp_path):
    base = _make_models_tree(tmp_path)
    ev = GateEvaluator(base, use_per_pair_routing=False)
    fake_sub = MagicMock(spec=GateEvaluator)
    ev._pair_evaluators["EUR_USD"] = fake_sub  # would be used if we routed

    ev._models_loaded = True
    ev.evaluate_momentum = MagicMock(return_value=(0.5, False))
    ev.evaluate_confidence = MagicMock(return_value=50.0)
    ev.evaluate_risk = MagicMock(return_value=(0.01, 0.0))
    ev.evaluate_transformer = MagicMock(return_value=(None, 0.5))
    ev.evaluate_meta_labeler = MagicMock(return_value=0.0)

    import pandas as pd
    ev.evaluate_all_gates(
        pd.DataFrame({"close": [1.0]}),
        instrument="EUR_USD",
    )
    fake_sub.evaluate_all_gates.assert_not_called()


# --- Failure isolation ----------------------------------------------------


def test_sub_evaluator_construction_failure_falls_back_to_self(tmp_path, monkeypatch):
    """If building the sub raises (corrupt pair-dir, partial files), fall
    back to self/joint and cache the no-op so we don't retry on every
    scan."""
    base = _make_models_tree(tmp_path)
    ev = GateEvaluator(base, use_per_pair_routing=True)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(GateEvaluator, "_load_lgbm_momentum", boom)
    monkeypatch.setattr(GateEvaluator, "_load_xgboost_momentum", lambda self: False)
    monkeypatch.setattr(GateEvaluator, "_load_catboost_momentum", lambda self: False)

    sub = ev._get_pair_evaluator("EUR_USD")
    assert sub is ev
    assert ev._pair_evaluators["EUR_USD"] is ev
