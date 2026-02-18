"""Unified RL inference integration layer for ensemble trading signals."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.core.modular_inference import TradeSignal
from src.training.rl.config import RLIntegrationConfig
from src.training.rl.exit_timer import ExitAction, OptimalExitRL
from src.training.rl.gate_optimizer import RLGateOptimizer
from src.training.rl.position_sizer import RLPositionSizer

logger = logging.getLogger(__name__)


class RLInferenceIntegration:
    """
    Integrates RL components into inference with graceful fallback behavior.

    This class is intentionally thin and uses existing RL components:
    - Position sizing: PPO (`src/training/rl/position_sizer.py`)
    - Gate thresholds: SAC (`src/rl/gate_threshold_env.py`)
    - Exit timing: PPO (`src/rl/optimal_exit_env.py`)
    """

    def __init__(self, config: Optional[RLIntegrationConfig] = None):
        self.config = config or RLIntegrationConfig()
        self.rl_sizer: Optional[RLPositionSizer] = None
        self.rl_gate_optimizer: Optional[RLGateOptimizer] = None
        self.rl_exit_timer: Optional[OptimalExitRL] = None
        self._load_rl_components()

    def _safe(self, fn, fallback):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if self.config.inference.fallback_on_error:
                logger.warning("RL integration fallback on error: %s", exc)
                return fallback
            raise

    def _load_rl_components(self) -> None:
        """Load enabled RL components, preserving graceful degradation."""
        if not self.config.enabled:
            return

        if self.config.position_sizing.enabled:
            def _load_sizer() -> Optional[RLPositionSizer]:
                sizer = RLPositionSizer()
                loaded = sizer.load(
                    model_path=Path(self.config.position_sizing.model_path),
                    scaler_path=Path(self.config.position_sizing.scaler_path),
                )
                return sizer if loaded else None

            self.rl_sizer = self._safe(_load_sizer, None)

        if self.config.gate_optimization.enabled:
            def _load_gates() -> Optional[RLGateOptimizer]:
                optimizer = RLGateOptimizer()
                loaded = optimizer.load()
                return optimizer if loaded else None

            self.rl_gate_optimizer = self._safe(_load_gates, None)

        if self.config.exit_timing.enabled:
            def _load_exits() -> Optional[OptimalExitRL]:
                optimizer = OptimalExitRL()
                loaded = optimizer.load()
                return optimizer if loaded else None

            self.rl_exit_timer = self._safe(_load_exits, None)

    def enhance(
        self,
        signal: Optional[TradeSignal],
        *,
        features: Optional[np.ndarray] = None,
        account_equity: float = 10000.0,
        regime: str = "NORMAL",
        unrealized_pnl_pips: float = 0.0,
        bars_held: int = 0,
        momentum: float = 0.0,
        atr: float = 0.01,
    ) -> TradeSignal:
        """
        Apply RL enhancements to an existing signal.

        Returns input signal unchanged when RL is disabled or unavailable.
        """
        if signal is None:
            return TradeSignal(
                trade=False,
                direction=None,
                size=0.0,
                confidence=0.0,
                reason="invalid_signal",
            )

        if not self.config.enabled:
            return signal

        if (
            self.config.inference.position_size_source == "rl"
            and self.rl_sizer is not None
            and signal.trade
            and features is not None
        ):
            pred = np.array([signal.tcn_probability, signal.confidence], dtype=np.float32)
            size = self.get_position_size(features, pred, account_equity)
            if size > 0:
                signal.size = float(size)

        if self.rl_gate_optimizer is not None and features is not None:
            thresholds = self.get_adaptive_thresholds(features, regime=regime)
            if thresholds:
                signal.rl_thresholds_used = True
                if signal.metadata is None:
                    signal.metadata = {}
                signal.metadata["rl_thresholds"] = thresholds

        if self.rl_exit_timer is not None and signal.trade:
            decision = self.get_exit_decision(
                unrealized_pnl_pips=unrealized_pnl_pips,
                bars_held=bars_held,
                momentum=momentum,
                atr=atr,
            )
            if decision:
                signal.rl_exit_suggestion = decision
                signal.rl_exit_confidence = (
                    self.config.inference.exit_timing_confidence
                )

        if self.config.inference.log_rl_decisions and signal.metadata is not None:
            signal.metadata["rl_enhanced"] = True

        return signal

    def get_position_size(
        self,
        features: np.ndarray,
        ensemble_prediction: np.ndarray,
        account_equity: float,
    ) -> float:
        """Get RL-based position size in dollars, with fallback to configured fixed percentage."""
        if self.rl_sizer is None:
            return account_equity * self.config.position_sizing.fallback_pct

        return self._safe(
            lambda: float(
                self.rl_sizer.get_position_size(
                    features=features,
                    ensemble_prediction=ensemble_prediction,
                    account_equity=account_equity,
                )
            ),
            account_equity * self.config.position_sizing.fallback_pct,
        )

    def get_adaptive_thresholds(
        self,
        features: np.ndarray,
        regime: str = "NORMAL",
    ) -> Dict[str, float]:
        """Get adaptive RL thresholds or static defaults."""
        if self.rl_gate_optimizer is None:
            return dict(self.config.gate_optimization.default_thresholds)

        regime_onehot = {
            "STRONG_TREND": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "WEAK_TREND": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "CHOP": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "MEAN_REVERT": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "BREAKOUT": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "NORMAL": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }.get(regime.upper(), np.array([1.0, 0.0, 0.0], dtype=np.float32))

        gate_obs = np.concatenate(
            [regime_onehot, np.array([0.5, 0.0], dtype=np.float32)]
        ).astype(np.float32)
        _ = features  # reserved for future richer state

        adjusted = self._safe(
            lambda: self.rl_gate_optimizer.get_adjusted_thresholds(
                features=gate_obs, win_rate=0.5, drawdown=0.0
            ),
            {},
        )
        if not adjusted:
            return dict(self.config.gate_optimization.default_thresholds)
        return adjusted

    def get_exit_decision(
        self,
        unrealized_pnl_pips: float,
        bars_held: int,
        momentum: float,
        atr: float,
    ) -> str:
        """Return RL exit decision as: hold, exit_profit, or exit_loss."""
        if self.rl_exit_timer is None:
            return "hold"

        action, _confidence = self._safe(
            lambda: self.rl_exit_timer.get_exit_decision(
                unrealized_pnl_pips=unrealized_pnl_pips,
                bars_in_trade=bars_held,
                momentum=momentum,
                atr=atr,
            ),
            (ExitAction.HOLD, 0.0),
        )
        return {
            ExitAction.HOLD: "hold",
            ExitAction.EXIT_PROFIT: "exit_profit",
            ExitAction.EXIT_LOSS: "exit_loss",
        }.get(int(action), "hold")
