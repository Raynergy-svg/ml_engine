"""Test Scanner.get_model_health() reports actual loaded models, not a
hardcoded count. Guards against the old TUI display bug (hardcoded 3/3).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_fake_scanner(
    transformer_regime=True,
    transformer_direction=True,
    tcn_volatility=True,
    histgb_baseline=True,
    momentum_catboost=False,
    momentum_lightgbm=True,
    ridge_confidence=True,
    rf_risk=True,
    transformer_gate=False,
    meta_labeler=False,
    tcn_volatility_gate=True,
    momentum_type="lightgbm",
):
    """Build a Scanner stub with only the surface get_model_health() touches."""
    from src.scanner.engine import Scanner

    s = Scanner.__new__(Scanner)

    # Group 1: modular ensemble
    mei = MagicMock()
    mei.regime_model = object() if transformer_regime else None
    mei.transformer = object() if transformer_direction else None
    mei.tcn = object() if tcn_volatility else None
    mei.histgb = object() if histgb_baseline else None
    s._modular_ensemble = mei

    # Group 2: gate evaluator
    ge = MagicMock()
    ge.get_model_health = MagicMock(return_value={
        "momentum_catboost": momentum_catboost,
        "momentum_xgboost": False,
        "momentum_lightgbm": momentum_lightgbm,
        "confidence_ridge": ridge_confidence,
        "risk_rf": rf_risk,
        "transformer_gate": transformer_gate,
        "meta_labeler": meta_labeler,
        "tcn_volatility_gate": tcn_volatility_gate,
    })
    ge._momentum_model_type = momentum_type
    s._gate_evaluator = ge

    return s


def test_get_model_health_reports_typical_production_load(tmp_path, monkeypatch):
    """Typical production: Transformer + TCN + HistGB + LightGBM gates loaded.

    Historical TUI bug reported '4 models' when reality is 8+. Verify the
    real count is returned.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "trained_data" / "models").mkdir(parents=True)
    # Simulate agent_weights.json on disk
    (tmp_path / "trained_data" / "models" / "agent_weights.json").write_text("{}")
    (tmp_path / "trained_data" / "models" / "calibration_params.json").write_text("{}")

    s = _make_fake_scanner()
    health = s.get_model_health()

    assert health["count"] > 4, (
        f"Production load should report more than 4 models, got {health['count']}. "
        f"This guards against the old TUI hardcoded 3/3 counter regression."
    )
    assert health["total"] > 4
    # Specific models we expect loaded in typical prod
    loaded = health["loaded"]
    assert loaded["transformer_regime"] is True
    assert loaded["transformer_direction"] is True
    assert loaded["tcn_volatility"] is True
    assert loaded["confidence_ridge"] is True
    assert loaded["risk_rf"] is True
    assert loaded["momentum_lightgbm"] is True
    assert loaded["tcn_volatility_gate"] is True
    assert loaded["agent_team_weights"] is True
    assert health["momentum_type"] == "lightgbm"


def test_get_model_health_degraded_only_tcn(tmp_path, monkeypatch):
    """Minimum viable: just TCN volatility loaded (scanner required model).

    Count should reflect that nearly everything is missing.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "trained_data" / "models").mkdir(parents=True)

    s = _make_fake_scanner(
        transformer_regime=False,
        transformer_direction=False,
        histgb_baseline=False,
        momentum_lightgbm=False,
        ridge_confidence=False,
        rf_risk=False,
        tcn_volatility_gate=True,
        tcn_volatility=True,
        momentum_type="none",
    )
    health = s.get_model_health()
    assert health["count"] <= 2
    assert health["loaded"]["tcn_volatility"] is True


def test_get_model_health_no_modular_ensemble(tmp_path, monkeypatch):
    """If modular ensemble failed to init, reporting still works (returns False)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "trained_data" / "models").mkdir(parents=True)

    from src.scanner.engine import Scanner
    s = Scanner.__new__(Scanner)
    s._modular_ensemble = None
    s._gate_evaluator = None
    health = s.get_model_health()
    assert health["count"] == 0
    assert health["loaded"]["transformer_regime"] is False
    assert health["loaded"]["momentum_catboost"] is False


def test_momentum_cascade_not_double_counted(tmp_path, monkeypatch):
    """Momentum has 3 cascade slots (catboost/xgb/lgbm) but only one loads.

    The total should reflect 1 momentum slot, not 3.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "trained_data" / "models").mkdir(parents=True)

    s_cb = _make_fake_scanner(
        momentum_catboost=True, momentum_lightgbm=False, momentum_type="catboost"
    )
    s_lgb = _make_fake_scanner(
        momentum_catboost=False, momentum_lightgbm=True, momentum_type="lightgbm"
    )

    h_cb = s_cb.get_model_health()
    h_lgb = s_lgb.get_model_health()

    # Regardless of which cascade option loaded, total should be equal
    assert h_cb["total"] == h_lgb["total"], (
        f"Momentum cascade slots being double-counted: catboost total={h_cb['total']}, "
        f"lightgbm total={h_lgb['total']}"
    )
