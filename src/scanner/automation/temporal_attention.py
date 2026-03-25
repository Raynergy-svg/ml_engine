"""Temporal attention for multi-timeframe signal weighting.

US-102: Replace hardcoded lookback windows with learned attention weights
over the time dimension. A lightweight single-head attention mechanism
learns which historical windows (5/20/50/100-bar) matter most for each
pair and regime. Dynamically reweights short-term vs long-term signals.

Based on 2024-2025 Temporal Fusion Transformer research, simplified
to gradient-free online learning suitable for a trading system.

Attention mechanism:
1. Each timeframe has a learnable weight (initialized uniform)
2. After each trade outcome, attention weights are updated:
   - Winning trade: boost timeframes whose signals aligned with direction
   - Losing trade: penalize timeframes whose signals aligned with direction
3. Softmax normalization ensures weights sum to 1.0
4. Regime-conditioned: separate attention profiles per regime
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ATTENTION_PATH = _PROJECT_ROOT / "trained_data" / "temporal_attention.json"

# Default timeframes (in bars)
DEFAULT_TIMEFRAMES = [5, 20, 50, 100]

# Learning parameters
LEARNING_RATE = 0.05  # Step size for attention weight updates
MIN_WEIGHT = 0.05  # Floor for any single timeframe weight
TEMPERATURE = 1.0  # Softmax temperature (lower = sharper)
DECAY_RATE = 0.001  # Slight decay toward uniform each update cycle
UPDATE_CLIP = 0.15  # Maximum single-step weight change


def _softmax(logits: List[float], temperature: float = 1.0) -> List[float]:
    """Temperature-scaled softmax for attention weights."""
    if not logits:
        return []
    scaled = [x / max(temperature, 0.01) for x in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(x - max_val) for x in scaled]
    total = sum(exp_vals)
    if total == 0:
        return [1.0 / len(logits)] * len(logits)
    return [e / total for e in exp_vals]


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""

    timeframe: int = 0  # Lookback window in bars
    direction: float = 0.0  # -1.0 (SELL) to +1.0 (BUY)
    strength: float = 0.0  # Signal strength [0, 1]
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timeframe": self.timeframe,
            "direction": round(self.direction, 4),
            "strength": round(self.strength, 4),
        }


class TemporalAttentionLayer:
    """Learned attention over multi-timeframe signals.

    Maintains per-regime attention logits over timeframes. Updated
    via gradient-free learning from trade outcomes.

    Usage:
        attention = TemporalAttentionLayer()
        signals = [
            TimeframeSignal(timeframe=5, direction=1.0, strength=0.8),
            TimeframeSignal(timeframe=20, direction=1.0, strength=0.6),
            TimeframeSignal(timeframe=50, direction=-1.0, strength=0.3),
            TimeframeSignal(timeframe=100, direction=1.0, strength=0.5),
        ]
        weighted = attention.compute_attention(signals, regime="NORMAL")
        # Returns attention-weighted signal vector
        attention.update_from_outcome("win", signals, regime="NORMAL")
    """

    def __init__(
        self,
        timeframes: Optional[List[int]] = None,
        learning_rate: float = LEARNING_RATE,
        temperature: float = TEMPERATURE,
        persistence_path: Optional[Path] = None,
    ):
        self.timeframes = timeframes or list(DEFAULT_TIMEFRAMES)
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.persistence_path = persistence_path or _ATTENTION_PATH

        # Attention logits: regime → {timeframe: logit}
        # Using logits (pre-softmax) for stable gradient-free updates
        self._logits: Dict[str, Dict[int, float]] = defaultdict(
            lambda: {tf: 0.0 for tf in self.timeframes}
        )

        # Update history for observability
        self._update_count: int = 0
        self._update_log: List[Dict[str, Any]] = []

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_attention(
        self,
        timeframe_signals: List[TimeframeSignal],
        regime: str = "NORMAL",
    ) -> Dict[str, float]:
        """Compute attention-weighted signal vector.

        Args:
            timeframe_signals: Signals from each timeframe.
            regime: Current volatility regime.

        Returns:
            Dict with:
                - "weighted_direction": attention-weighted direction [-1, 1]
                - "weighted_strength": attention-weighted strength [0, 1]
                - "attention_entropy": entropy of attention distribution
                - "dominant_timeframe": timeframe with highest attention
        """
        if not timeframe_signals:
            return {
                "weighted_direction": 0.0,
                "weighted_strength": 0.0,
                "attention_entropy": 0.0,
                "dominant_timeframe": 0,
            }

        # Get attention weights for this regime
        weights = self.get_attention_weights(regime)

        # Compute weighted signals
        weighted_dir = 0.0
        weighted_str = 0.0
        total_weight = 0.0

        for signal in timeframe_signals:
            w = weights.get(signal.timeframe, 1.0 / len(self.timeframes))
            weighted_dir += w * signal.direction * signal.strength
            weighted_str += w * signal.strength
            total_weight += w

        if total_weight > 0:
            weighted_dir /= total_weight
            weighted_str /= total_weight

        # Compute attention entropy
        weight_values = list(weights.values())
        entropy = 0.0
        for w in weight_values:
            if w > 1e-10:
                entropy -= w * math.log(w + 1e-10)

        # Find dominant timeframe
        dominant_tf = max(weights, key=lambda tf: weights[tf]) if weights else 0

        return {
            "weighted_direction": round(float(np.clip(weighted_dir, -1.0, 1.0)), 4),
            "weighted_strength": round(float(np.clip(weighted_str, 0.0, 1.0)), 4),
            "attention_entropy": round(entropy, 4),
            "dominant_timeframe": dominant_tf,
        }

    def get_attention_weights(
        self,
        regime: str = "NORMAL",
    ) -> Dict[int, float]:
        """Get current attention distribution over timeframes.

        Args:
            regime: Volatility regime.

        Returns:
            Dict mapping timeframe → attention weight (sums to ~1.0).
        """
        logits = self._logits[regime]
        logit_list = [logits.get(tf, 0.0) for tf in self.timeframes]
        weights = _softmax(logit_list, self.temperature)

        result = {}
        for i, tf in enumerate(self.timeframes):
            result[tf] = max(MIN_WEIGHT, weights[i])

        # Re-normalize after min-weight floor
        total = sum(result.values())
        if total > 0:
            result = {tf: w / total for tf, w in result.items()}

        return result

    def update_from_outcome(
        self,
        trade_result: str,
        timeframe_signals: List[TimeframeSignal],
        regime: str = "NORMAL",
        trade_direction: str = "BUY",
    ) -> None:
        """Adjust attention weights via gradient-free learning.

        Winning trade: boost timeframes whose signals aligned with direction.
        Losing trade: penalize timeframes whose signals aligned (they led
        to a bad trade) and boost opposing timeframes (they were right).

        Args:
            trade_result: "win", "loss", or "breakeven".
            timeframe_signals: Signals that were active for this trade.
            regime: Regime at time of trade.
            trade_direction: "BUY" or "SELL".
        """
        if trade_result == "breakeven":
            return  # No learning from breakeven

        is_win = trade_result == "win"
        dir_sign = 1.0 if trade_direction == "BUY" else -1.0

        logits = self._logits[regime]
        updates: Dict[int, float] = {}

        for signal in timeframe_signals:
            tf = signal.timeframe
            if tf not in logits:
                logits[tf] = 0.0

            # Alignment: how much this timeframe agreed with trade direction
            alignment = signal.direction * dir_sign * signal.strength

            if is_win:
                # Boost aligned timeframes (they contributed to a win)
                delta = self.learning_rate * alignment
            else:
                # Penalize aligned timeframes (they contributed to a loss)
                # Slightly boost opposing timeframes
                delta = -self.learning_rate * alignment

            # Clip update
            delta = float(np.clip(delta, -UPDATE_CLIP, UPDATE_CLIP))
            logits[tf] += delta
            updates[tf] = delta

        # Apply slight decay toward uniform (prevents extreme specialization)
        for tf in self.timeframes:
            if tf in logits:
                logits[tf] *= (1.0 - DECAY_RATE)

        self._update_count += 1

        # Log update
        self._update_log.append({
            "regime": regime,
            "result": trade_result,
            "direction": trade_direction,
            "updates": {str(k): round(v, 6) for k, v in updates.items()},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._update_log) > 200:
            self._update_log = self._update_log[-100:]

        logger.debug(
            f"TemporalAttention: Updated {regime} from {trade_result} "
            f"({len(updates)} timeframes adjusted)"
        )

    def get_regime_profile(self, regime: str = "NORMAL") -> Dict[str, Any]:
        """Get detailed attention profile for a regime.

        Returns:
            Dict with attention weights, dominant timeframe, entropy.
        """
        weights = self.get_attention_weights(regime)
        logits = dict(self._logits[regime])

        # Entropy
        entropy = 0.0
        for w in weights.values():
            if w > 1e-10:
                entropy -= w * math.log(w + 1e-10)

        max_entropy = math.log(max(len(self.timeframes), 1))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            "regime": regime,
            "weights": {str(k): round(v, 4) for k, v in weights.items()},
            "logits": {str(k): round(v, 4) for k, v in logits.items()},
            "dominant_timeframe": max(weights, key=lambda tf: weights[tf]),
            "entropy": round(entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "is_specialized": normalized_entropy < 0.70,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get attention layer status for observability."""
        regime_profiles = {}
        for regime in self._logits:
            weights = self.get_attention_weights(regime)
            dominant = max(weights, key=lambda tf: weights[tf])
            regime_profiles[regime] = {
                "dominant_timeframe": dominant,
                "dominant_weight": round(weights[dominant], 4),
            }

        return {
            "timeframes": self.timeframes,
            "n_regimes": len(self._logits),
            "total_updates": self._update_count,
            "regime_profiles": regime_profiles,
            "recent_updates": self._update_log[-5:],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist attention weights."""
        try:
            data = {
                "version": 1,
                "timeframes": self.timeframes,
                "logits": {
                    regime: {str(tf): round(logit, 6) for tf, logit in tfs.items()}
                    for regime, tfs in self._logits.items()
                },
                "update_count": self._update_count,
                "update_log": self._update_log[-100:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"TemporalAttention: Failed to save: {e}")

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

            # Restore logits
            for regime, tfs in data.get("logits", {}).items():
                self._logits[regime] = {
                    int(tf): float(logit) for tf, logit in tfs.items()
                }

            self._update_count = data.get("update_count", 0)
            self._update_log = data.get("update_log", [])

            logger.debug(
                f"TemporalAttention: Loaded {len(self._logits)} regime profiles, "
                f"{self._update_count} updates"
            )
        except Exception as e:
            logger.debug(f"TemporalAttention: Failed to load: {e}")
