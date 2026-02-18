"""Tests for the unified RL inference integration layer."""

from __future__ import annotations

import numpy as np

from src.core.modular_inference import TradeSignal
from src.core.rl_inference_integration import RLInferenceIntegration
from src.training.rl.config import RLIntegrationConfig


def test_enhance_returns_original_signal_when_disabled() -> None:
    integration = RLInferenceIntegration(config=RLIntegrationConfig(enabled=False))
    signal = TradeSignal(trade=True, direction="long", size=1.0, confidence=0.7)
    enhanced = integration.enhance(signal)
    assert enhanced is signal
    assert enhanced.size == 1.0


def test_enhance_handles_invalid_signal() -> None:
    integration = RLInferenceIntegration(config=RLIntegrationConfig(enabled=False))
    enhanced = integration.enhance(None)
    assert enhanced.trade is False
    assert enhanced.reason == "invalid_signal"


def test_enhance_applies_rl_components(monkeypatch) -> None:
    class FakeSizer:
        def load(self, model_path=None, scaler_path=None):  # noqa: ANN001
            return True

        def get_position_size(self, features, ensemble_prediction, account_equity):  # noqa: ANN001
            return account_equity * 0.02

    class FakeGateOptimizer:
        def load(self):  # noqa: ANN201
            return True

        def get_adjusted_thresholds(self, features, win_rate=0.5, drawdown=0.0):  # noqa: ANN001
            return {
                "min_confidence": 52.0,
                "min_momentum": 0.22,
                "max_drawdown_pct": 0.02,
            }

    class FakeExitTimer:
        def load(self):  # noqa: ANN201
            return True

        def get_exit_decision(self, unrealized_pnl_pips, bars_in_trade, momentum, atr):  # noqa: ANN001
            return 1, 0.9

    monkeypatch.setattr(
        "src.core.rl_inference_integration.RLPositionSizer",
        FakeSizer,
    )
    monkeypatch.setattr(
        "src.core.rl_inference_integration.RLGateOptimizer",
        FakeGateOptimizer,
    )
    monkeypatch.setattr(
        "src.core.rl_inference_integration.OptimalExitRL",
        FakeExitTimer,
    )

    cfg = RLIntegrationConfig(enabled=True)
    cfg.position_sizing.enabled = True
    cfg.gate_optimization.enabled = True
    cfg.exit_timing.enabled = True

    integration = RLInferenceIntegration(config=cfg)
    signal = TradeSignal(
        trade=True,
        direction="long",
        size=1.0,
        confidence=0.68,
        tcn_probability=0.66,
    )

    features = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    enhanced = integration.enhance(
        signal,
        features=features,
        account_equity=10_000.0,
        regime="NORMAL",
        unrealized_pnl_pips=25.0,
        bars_held=4,
        momentum=0.2,
        atr=0.01,
    )

    assert enhanced.size == 200.0
    assert enhanced.rl_thresholds_used is True
    assert enhanced.rl_exit_suggestion == "exit_profit"
    assert enhanced.metadata is not None
    assert enhanced.metadata["rl_enhanced"] is True

