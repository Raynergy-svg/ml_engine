"""Integration test: CQLRebalancer wiring into ModularEnsembleInference._calculate_position_size().

US-010: When config.use_cql_rebalancer=True and a CQLRebalancer is loaded,
position sizing should use the rebalancer's action selection.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.core.modular_inference import ModularEnsembleInference, InferenceConfig


def test_cql_rebalancer_used_when_enabled():
    """When CQL rebalancer is configured and available, it should influence sizing."""
    cfg = InferenceConfig()
    ensemble = ModularEnsembleInference(config=cfg)

    # Mock CQL rebalancer
    mock_cql = MagicMock()
    mock_cql.select_action.return_value = 3  # action index 3
    ensemble.cql_rebalancer = mock_cql

    # We need to bypass rl_sizer to hit CQL branch
    ensemble.rl_sizer = None

    # Manually set config attribute if missing
    if not hasattr(cfg, "use_cql_rebalancer"):
        cfg.use_cql_rebalancer = True
    if not hasattr(cfg, "position_sizing_mode"):
        cfg.position_sizing_mode = "cql_hybrid"

    size = ensemble._calculate_position_size(
        expected_drawdown_pips=20.0,
        equity=100000.0,
        instrument="EUR_USD",
        current_price=1.1000,
        features=np.array([0.5, 0.6, 0.7, 0.8]),
        tcn_probability=0.7,
        ridge_confidence=75.0,
    )

    mock_cql.select_action.assert_called_once()
    assert size > 0


def test_cql_rebalancer_disabled_does_not_call():
    cfg = InferenceConfig()
    ensemble = ModularEnsembleInference(config=cfg)
    mock_cql = MagicMock()
    ensemble.cql_rebalancer = mock_cql
    ensemble.rl_sizer = None

    if not hasattr(cfg, "use_cql_rebalancer"):
        cfg.use_cql_rebalancer = False
    if not hasattr(cfg, "position_sizing_mode"):
        cfg.position_sizing_mode = "risk_only"

    size = ensemble._calculate_position_size(
        expected_drawdown_pips=20.0,
        equity=100000.0,
        instrument="EUR_USD",
        current_price=1.1000,
        features=np.array([0.5, 0.6, 0.7, 0.8]),
        tcn_probability=0.7,
        ridge_confidence=75.0,
    )

    mock_cql.select_action.assert_not_called()
    assert size > 0
