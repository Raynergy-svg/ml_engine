"""Integration test: HybridInference + TimesFMAdapter wiring into Scanner._run_inference().

US-008: When config.use_hybrid_inference=True, the engine must branch to
HybridInference.predict() and map its output to the standard tuple contract.
"""
from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.scanner.config import ScannerConfig
from src.scanner.engine import Scanner


@pytest.fixture
def minimal_config():
    cfg = ScannerConfig()
    cfg.use_hybrid_inference = True
    cfg.enable_execution = False
    cfg.min_confidence = 30.0
    cfg.min_tcn_probability = 0.45
    cfg.final_score_threshold = 0.45
    cfg.min_momentum = 0.0
    cfg.max_drawdown_pct = 5.0
    return cfg


def test_hybrid_inference_used_when_enabled(minimal_config):
    """When use_hybrid_inference=True, _run_inference must use HybridInference."""
    scanner = Scanner(minimal_config)

    # Prepare minimal OHLCV data
    df_raw = pd.DataFrame({
        "open": [1.1000] * 150,
        "high": [1.1005] * 150,
        "low": [1.0995] * 150,
        "close": [1.1002] * 150,
        "volume": [1000] * 150,
    })
    df_feat = pd.DataFrame({
        "rsi": [50.0] * 150,
        "adx": [25.0] * 150,
        "atr": [0.001] * 150,
        "returns": [0.0] * 150,
    })

    # Mock HybridInference so we don't need real models
    mock_hybrid = MagicMock()
    mock_hybrid.predict.return_value = {
        "direction": "LONG",
        "confidence": 0.72,
        "regime": "NORMAL",
        "ood_detected": False,
        "weights_used": {"foundation": 0.5, "custom": 0.3},
        "source": "hybrid_ensemble",
    }
    scanner._hybrid_inference = mock_hybrid

    # We need to bypass the _infer_from_ensemble path so the hybrid branch is reached
    # The simplest way is to ensure _modular_ensemble is None so the code falls
    # through to the hybrid branch (which we'll add).
    scanner._modular_ensemble = None
    scanner._init_modular_ensemble = lambda: False
    scanner._ensemble_loaded = False

    direction, confidence, tcn_conf, ridge_conf, gates_passed, vol_regime, succeeded, details = (
        scanner._run_inference(df_raw, df_feat, "EUR_USD")
    )

    mock_hybrid.predict.assert_called_once()
    assert direction == "LONG"
    assert confidence == pytest.approx(0.72, abs=0.01)
    assert succeeded is True
    assert details.get("source") == "hybrid_ensemble"
    assert details.get("ood_detected") is False


def test_hybrid_inference_disabled_does_not_call_hybrid(minimal_config):
    """When use_hybrid_inference=False, HybridInference must not be invoked."""
    minimal_config.use_hybrid_inference = False
    scanner = Scanner(minimal_config)

    df_raw = pd.DataFrame({
        "open": [1.1000] * 50,
        "high": [1.1005] * 50,
        "low": [1.0995] * 50,
        "close": [1.1002] * 50,
        "volume": [1000] * 50,
    })
    df_feat = pd.DataFrame({
        "rsi": [50.0] * 50,
        "adx": [25.0] * 50,
        "atr": [0.001] * 50,
        "returns": [0.0] * 50,
    })

    mock_hybrid = MagicMock()
    scanner._hybrid_inference = mock_hybrid
    scanner._modular_ensemble = None
    scanner._init_modular_ensemble = lambda: False
    scanner._ensemble_loaded = False

    # With hybrid disabled and no ensemble, it will fall through to gate eval
    # which may also fail, so we just verify hybrid.predict is NOT called
    try:
        scanner._run_inference(df_raw, df_feat, "EUR_USD")
    except Exception:
        pass  # expected — no inference backend available

    mock_hybrid.predict.assert_not_called()
