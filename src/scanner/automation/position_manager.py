"""Live position management driven by trade outcome prediction.

US-103: Wire trade_outcome_predictor (US-082) into execution monitoring.
Currently the predictor exists but is never called for live positions.
This module computes post-entry features from live trade context and
routes predictions to position management actions.

Actions:
- HOLD: keep position unchanged (win_prob > 0.55)
- TIGHTEN_SL: move SL closer to entry (win_prob 0.40-0.55)
- TAKE_PARTIAL: close portion at profit (MFE high + win_prob declining)
- EXIT_EARLY: close position (win_prob < 0.30)

Closes the prediction → action feedback loop that was missing since US-082.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "position_management_log.json"

# Action thresholds (from trade_outcome_predictor)
WIN_PROB_HOLD = 0.55  # Above this: HOLD
WIN_PROB_TIGHTEN = 0.40  # Between TIGHTEN and HOLD
WIN_PROB_EXIT = 0.30  # Below this: EXIT_EARLY
MFE_PARTIAL_THRESHOLD = 0.65  # MFE as fraction of TP distance triggers partial

# Position management parameters
SL_TIGHTEN_RATIO = 0.50  # Move SL 50% closer to entry
PARTIAL_CLOSE_RATIO = 0.50  # Close 50% of position on TAKE_PARTIAL
MIN_BARS_BEFORE_ACTION = 3  # Don't act in first 3 bars
MAX_ACTIONS_PER_TRADE = 5  # Circuit breaker: max actions per trade
COOLDOWN_BARS = 5  # Minimum bars between actions on same trade


@dataclass
class PositionContext:
    """Context for evaluating an open position."""

    trade_id: str = ""
    pair: str = ""
    direction: str = "BUY"  # BUY or SELL
    entry_price: float = 0.0
    current_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    lot_size: float = 0.0
    bars_in_trade: int = 0
    mae_pips: float = 0.0  # Maximum adverse excursion
    mfe_pips: float = 0.0  # Maximum favorable excursion
    entry_atr: float = 0.0
    current_atr: float = 0.0
    momentum_since_entry: float = 0.0
    regime: str = "NORMAL"

    def to_features(self) -> Dict[str, float]:
        """Compute the 7 post-entry features for prediction."""
        # Distance to SL as percentage of total SL distance
        sl_distance = abs(self.entry_price - self.sl_price) or 1e-6
        if self.direction == "BUY":
            dist_to_sl = (self.current_price - self.sl_price) / sl_distance
            dist_to_tp = (self.current_price - self.entry_price) / (
                abs(self.tp_price - self.entry_price) or 1e-6
            )
        else:
            dist_to_sl = (self.sl_price - self.current_price) / sl_distance
            dist_to_tp = (self.entry_price - self.current_price) / (
                abs(self.entry_price - self.tp_price) or 1e-6
            )

        # Volatility ratio
        vol_ratio = self.current_atr / max(self.entry_atr, 1e-8)

        return {
            "dist_to_sl_pct": float(np.clip(dist_to_sl, -2.0, 3.0)),
            "dist_to_tp_pct": float(np.clip(dist_to_tp, -1.0, 2.0)),
            "mae_pips": self.mae_pips,
            "mfe_pips": self.mfe_pips,
            "bars_in_trade": float(self.bars_in_trade),
            "momentum_since_entry": self.momentum_since_entry,
            "current_vol_vs_entry_vol": float(np.clip(vol_ratio, 0.1, 10.0)),
        }


@dataclass
class ManagementAction:
    """Recommended action for a position."""

    action: str = "HOLD"  # HOLD, TIGHTEN_SL, TAKE_PARTIAL, EXIT_EARLY
    confidence: float = 0.0
    win_probability: float = 0.5
    new_sl_price: Optional[float] = None  # For TIGHTEN_SL
    close_ratio: float = 0.0  # For TAKE_PARTIAL (0.0 to 1.0)
    reason: str = ""
    features_used: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "win_probability": round(self.win_probability, 4),
            "reason": self.reason,
        }
        if self.new_sl_price is not None:
            result["new_sl_price"] = round(self.new_sl_price, 5)
        if self.close_ratio > 0:
            result["close_ratio"] = round(self.close_ratio, 2)
        return result


class LivePositionManager:
    """Manages open positions using trade outcome prediction.

    Bridges the gap between trade_outcome_predictor (which generates
    predictions) and execution (which manages positions). Computes
    post-entry features from live position context, gets predictions,
    and translates them to actionable position management commands.

    Usage:
        manager = LivePositionManager()
        ctx = PositionContext(trade_id="T001", pair="EUR_USD", ...)
        action = manager.evaluate_position(ctx)
        # action.action = "TIGHTEN_SL", action.new_sl_price = 1.0825
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None,
    ):
        self.persistence_path = persistence_path or _LOG_PATH

        # Predictor (lazy-loaded to avoid circular imports)
        self._predictor = None
        self._predictor_loaded = False

        # Action tracking per trade (prevents overacting)
        self._action_counts: Dict[str, int] = {}
        self._last_action_bar: Dict[str, int] = {}

        # Decision log
        self._decisions: List[Dict[str, Any]] = []
        self._total_decisions: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_position(
        self,
        context: PositionContext,
    ) -> ManagementAction:
        """Evaluate an open position and recommend an action.

        Args:
            context: Live position context with current prices.

        Returns:
            ManagementAction with recommended action and parameters.
        """
        # Guard: minimum bars before any action
        if context.bars_in_trade < MIN_BARS_BEFORE_ACTION:
            return ManagementAction(
                action="HOLD",
                confidence=1.0,
                win_probability=0.5,
                reason=f"Too early ({context.bars_in_trade} < {MIN_BARS_BEFORE_ACTION} bars)",
            )

        # Guard: action count circuit breaker
        action_count = self._action_counts.get(context.trade_id, 0)
        if action_count >= MAX_ACTIONS_PER_TRADE:
            return ManagementAction(
                action="HOLD",
                confidence=1.0,
                win_probability=0.5,
                reason=f"Action limit reached ({action_count}/{MAX_ACTIONS_PER_TRADE})",
            )

        # Guard: cooldown between actions
        last_bar = self._last_action_bar.get(context.trade_id, 0)
        if context.bars_in_trade - last_bar < COOLDOWN_BARS:
            return ManagementAction(
                action="HOLD",
                confidence=0.8,
                win_probability=0.5,
                reason=f"Cooldown active ({context.bars_in_trade - last_bar} < {COOLDOWN_BARS} bars)",
            )

        # Compute post-entry features
        features = context.to_features()

        # Get prediction from trade outcome predictor
        prediction = self._get_prediction(features, context.pair)
        win_prob = prediction.get("win_probability", 0.5)
        is_trained = prediction.get("is_trained", False)

        # If predictor not trained, be conservative
        if not is_trained:
            return ManagementAction(
                action="HOLD",
                confidence=0.3,
                win_probability=0.5,
                reason="Predictor not trained (insufficient historical data)",
                features_used=features,
            )

        # Determine action from win probability
        action = self._decide_action(win_prob, features, context)
        action.features_used = features

        # Track action
        if action.action != "HOLD":
            self._action_counts[context.trade_id] = action_count + 1
            self._last_action_bar[context.trade_id] = context.bars_in_trade

        # Log decision
        self._log_decision(context, action)

        return action

    def cleanup_trade(self, trade_id: str) -> None:
        """Clean up tracking for a closed trade."""
        self._action_counts.pop(trade_id, None)
        self._last_action_bar.pop(trade_id, None)

    def get_status(self) -> Dict[str, Any]:
        """Get position manager status for observability."""
        action_dist: Dict[str, int] = {}
        for d in self._decisions[-50:]:
            act = d.get("action", "HOLD")
            action_dist[act] = action_dist.get(act, 0) + 1

        return {
            "total_decisions": self._total_decisions,
            "active_trades_tracked": len(self._action_counts),
            "recent_action_distribution": action_dist,
            "predictor_loaded": self._predictor_loaded,
            "recent_decisions": self._decisions[-5:],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _decide_action(
        self,
        win_prob: float,
        features: Dict[str, float],
        context: PositionContext,
    ) -> ManagementAction:
        """Route win probability to position management action."""

        # EXIT_EARLY: very low win probability
        if win_prob < WIN_PROB_EXIT:
            return ManagementAction(
                action="EXIT_EARLY",
                confidence=1.0 - win_prob,
                win_probability=win_prob,
                reason=f"Win probability critically low ({win_prob:.2f} < {WIN_PROB_EXIT})",
            )

        # TAKE_PARTIAL: MFE is high but probability declining
        dist_to_tp = features.get("dist_to_tp_pct", 0.0)
        if (
            dist_to_tp >= MFE_PARTIAL_THRESHOLD
            and win_prob < WIN_PROB_HOLD
        ):
            return ManagementAction(
                action="TAKE_PARTIAL",
                confidence=dist_to_tp * (1.0 - win_prob),
                win_probability=win_prob,
                close_ratio=PARTIAL_CLOSE_RATIO,
                reason=f"High MFE ({dist_to_tp:.2f}) with declining win prob ({win_prob:.2f})",
            )

        # TIGHTEN_SL: moderate uncertainty
        if win_prob < WIN_PROB_HOLD:
            new_sl = self._compute_tightened_sl(context)
            return ManagementAction(
                action="TIGHTEN_SL",
                confidence=(WIN_PROB_HOLD - win_prob) / (WIN_PROB_HOLD - WIN_PROB_TIGHTEN),
                win_probability=win_prob,
                new_sl_price=new_sl,
                reason=f"Win probability uncertain ({win_prob:.2f}), tightening SL",
            )

        # HOLD: confident in trade
        return ManagementAction(
            action="HOLD",
            confidence=win_prob,
            win_probability=win_prob,
            reason=f"Win probability favorable ({win_prob:.2f} >= {WIN_PROB_HOLD})",
        )

    def _compute_tightened_sl(self, context: PositionContext) -> float:
        """Compute a tightened stop loss price."""
        if context.direction == "BUY":
            # Move SL up (closer to entry)
            current_distance = context.entry_price - context.sl_price
            tightened_distance = current_distance * (1.0 - SL_TIGHTEN_RATIO)
            new_sl = context.entry_price - tightened_distance
            # Don't move SL above current price
            return min(new_sl, context.current_price - 0.0001)
        else:
            # Move SL down (closer to entry)
            current_distance = context.sl_price - context.entry_price
            tightened_distance = current_distance * (1.0 - SL_TIGHTEN_RATIO)
            new_sl = context.entry_price + tightened_distance
            # Don't move SL below current price
            return max(new_sl, context.current_price + 0.0001)

    def _get_prediction(
        self,
        features: Dict[str, float],
        pair: str,
    ) -> Dict[str, Any]:
        """Get prediction from trade outcome predictor."""
        if not self._predictor_loaded:
            try:
                from src.scanner.automation.trade_outcome_predictor import (
                    TradeOutcomePredictor,
                )
                self._predictor = TradeOutcomePredictor()
                self._predictor_loaded = True
            except Exception as e:
                logger.debug(f"PositionManager: Predictor import failed: {e}")
                return {"win_probability": 0.5, "is_trained": False}

        if self._predictor is None:
            return {"win_probability": 0.5, "is_trained": False}

        try:
            # Call the predictor's predict method
            feature_array = [
                features.get("dist_to_sl_pct", 0.0),
                features.get("dist_to_tp_pct", 0.0),
                features.get("mae_pips", 0.0),
                features.get("mfe_pips", 0.0),
                features.get("bars_in_trade", 0.0),
                features.get("momentum_since_entry", 0.0),
                features.get("current_vol_vs_entry_vol", 1.0),
            ]

            result = self._predictor.predict(pair, feature_array)
            if hasattr(result, "to_dict"):
                return result.to_dict()
            elif isinstance(result, dict):
                return result
            else:
                return {"win_probability": 0.5, "is_trained": False}
        except Exception as e:
            logger.debug(f"PositionManager: Prediction failed: {e}")
            return {"win_probability": 0.5, "is_trained": False}

    def _log_decision(
        self,
        context: PositionContext,
        action: ManagementAction,
    ) -> None:
        """Log a position management decision."""
        self._total_decisions += 1
        entry = {
            "trade_id": context.trade_id,
            "pair": context.pair,
            "bars_in_trade": context.bars_in_trade,
            "action": action.action,
            "win_probability": round(action.win_probability, 4),
            "confidence": round(action.confidence, 4),
            "reason": action.reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if action.new_sl_price is not None:
            entry["new_sl_price"] = round(action.new_sl_price, 5)
        if action.close_ratio > 0:
            entry["close_ratio"] = round(action.close_ratio, 2)

        self._decisions.append(entry)
        if len(self._decisions) > 500:
            self._decisions = self._decisions[-250:]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist position management log."""
        try:
            data = {
                "version": 1,
                "total_decisions": self._total_decisions,
                "decisions": self._decisions[-250:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"PositionManager: Failed to save: {e}")

    def _load_state(self) -> None:
        """Load persisted state."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                return

            self._total_decisions = data.get("total_decisions", 0)
            self._decisions = data.get("decisions", [])

            logger.debug(
                f"PositionManager: Loaded {self._total_decisions} historical decisions"
            )
        except Exception as e:
            logger.debug(f"PositionManager: Failed to load: {e}")
