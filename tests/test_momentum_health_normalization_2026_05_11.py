"""Momentum health-cascade normalization tests (added 2026-05-11).

GateEvaluator's momentum cascade is sequential fallback (catboost → xgboost →
lightgbm); only one backend ever loads. Pre-fix, ``Scanner.get_model_health()``
returned three booleans in the ``loaded`` detail dict (one True for the active
backend, two False for the unused alternatives). The TUI diagnostics screen
rendered each as a row, so the operator saw two red "MISSING" entries for
backends that were never expected to load — the lying-counter pattern.

This commit adds ``_normalize_momentum_health`` to collapse the unused cascade
alternatives before they reach the TUI's renderer. Pure function, no side
effects; ``Scanner.get_model_health`` applies it just before the return.

NO MOCKS rule (CLAUDE.md / .claude/rules/improvement.md "No-Mock Rule"):
the function under test is pure (dict → dict), so the tests pass real dicts.
No mocks needed.
"""

from __future__ import annotations

from src.scanner.engine import _normalize_momentum_health


# ---------------------------------------------------------------------------
# Active-backend case: only the winning row survives
# ---------------------------------------------------------------------------


def test_lightgbm_active_drops_catboost_and_xgboost_false_alternatives() -> None:
    """When LightGBM is the active backend, the False catboost/xgboost rows
    must be dropped so the TUI does not render them as MISSING."""
    health = {
        "momentum_catboost": False,
        "momentum_xgboost": False,
        "momentum_lightgbm": True,
        "confidence_ridge": True,
    }
    out = _normalize_momentum_health(health, active_backend="lightgbm")
    assert "momentum_catboost" not in out
    assert "momentum_xgboost" not in out
    assert out["momentum_lightgbm"] is True
    # Non-momentum keys pass through untouched.
    assert out["confidence_ridge"] is True


def test_catboost_active_drops_xgboost_and_lightgbm_alternatives() -> None:
    """Symmetric: CatBoost-active deployment keeps only its own row."""
    health = {
        "momentum_catboost": True,
        "momentum_xgboost": False,
        "momentum_lightgbm": False,
        "risk_rf": True,
    }
    out = _normalize_momentum_health(health, active_backend="catboost")
    assert out["momentum_catboost"] is True
    assert "momentum_xgboost" not in out
    assert "momentum_lightgbm" not in out
    assert out["risk_rf"] is True


def test_xgboost_active_drops_catboost_and_lightgbm_alternatives() -> None:
    """Symmetric: XGBoost-active deployment keeps only its own row."""
    health = {
        "momentum_catboost": False,
        "momentum_xgboost": True,
        "momentum_lightgbm": False,
    }
    out = _normalize_momentum_health(health, active_backend="xgboost")
    assert out["momentum_xgboost"] is True
    assert "momentum_catboost" not in out
    assert "momentum_lightgbm" not in out


# ---------------------------------------------------------------------------
# No-backend case: collapse to a single honest "momentum: False" row
# ---------------------------------------------------------------------------


def test_no_backend_loaded_collapses_to_single_momentum_false() -> None:
    """No cascade backend loaded → drop all three cascade keys and emit a
    single ``momentum: False`` row so the panel surfaces ONE outage entry,
    not THREE that imply broken parallel ensemble."""
    health = {
        "momentum_catboost": False,
        "momentum_xgboost": False,
        "momentum_lightgbm": False,
        "transformer_gate": True,
    }
    out = _normalize_momentum_health(health, active_backend="none")
    # All three cascade keys removed
    assert "momentum_catboost" not in out
    assert "momentum_xgboost" not in out
    assert "momentum_lightgbm" not in out
    # Single honest entry replaces them
    assert out["momentum"] is False
    # Other entries preserved
    assert out["transformer_gate"] is True


# ---------------------------------------------------------------------------
# Defensive: boolean state wins over the active_backend label if they
# disagree (e.g. label says lightgbm but the booleans say xgboost)
# ---------------------------------------------------------------------------


def test_boolean_state_wins_over_mislabeled_active_backend() -> None:
    """If the cascade booleans and the active_backend label disagree, the
    booleans are authoritative (that's the actual loaded state)."""
    health = {
        "momentum_catboost": False,
        "momentum_xgboost": True,   # boolean says xgboost is loaded
        "momentum_lightgbm": False,
    }
    # ...but label claims lightgbm. Function should still pick the True one.
    out = _normalize_momentum_health(health, active_backend="lightgbm")
    assert out.get("momentum_xgboost") is True
    assert "momentum_catboost" not in out
    assert "momentum_lightgbm" not in out


# ---------------------------------------------------------------------------
# Edge cases: missing cascade keys, empty health dict
# ---------------------------------------------------------------------------


def test_no_cascade_keys_in_health_passes_through() -> None:
    """If the input dict has no momentum_* cascade keys at all (e.g. early
    initialization or non-cascade caller), the function is a no-op."""
    health = {"confidence_ridge": True, "transformer_gate": False}
    out = _normalize_momentum_health(health, active_backend="none")
    assert out == health
    # Returned a new dict (not the same object) per the function contract.
    assert out is not health


def test_partial_cascade_keys_only_known_subset_processed() -> None:
    """If only some cascade keys are present (e.g. legacy ge missing the
    catboost key), the function still produces correct output for the
    subset it sees."""
    health = {
        "momentum_xgboost": False,
        "momentum_lightgbm": True,
    }
    out = _normalize_momentum_health(health, active_backend="lightgbm")
    assert out["momentum_lightgbm"] is True
    assert "momentum_xgboost" not in out


def test_empty_health_dict_passes_through() -> None:
    """Empty dict → empty dict, no exceptions."""
    out = _normalize_momentum_health({}, active_backend="none")
    assert out == {}


# ---------------------------------------------------------------------------
# Caller contract: function does not mutate input (engine.py count math
# runs on the original dict BEFORE calling this; mutation would break it)
# ---------------------------------------------------------------------------


def test_input_dict_not_mutated() -> None:
    """``engine.py`` computes ``loaded_count``/``total`` from ``health`` then
    calls ``_normalize_momentum_health(health, ...)`` and re-assigns. If
    this function mutated ``health`` in place, a future refactor that
    re-orders the call site would silently break the count math. The
    function must always return a new dict."""
    health = {
        "momentum_catboost": False,
        "momentum_xgboost": False,
        "momentum_lightgbm": True,
    }
    snapshot = dict(health)
    out = _normalize_momentum_health(health, active_backend="lightgbm")
    assert health == snapshot  # original unchanged
    assert out is not health   # new dict returned
