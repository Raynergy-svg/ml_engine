"""Attention-based dynamic feature weighting.

US-088: Replace static feature importance with a lightweight attention
mechanism. Per scan, compute attention weights over technical indicators —
dynamically prioritize features that matter in current conditions
(e.g., RSI matters more in mean-reversion, trend_strength matters more
in trending). Uses softmax attention without full transformer overhead.

Architecture:
- Query: current regime vector (one-hot encoded)
- Keys: per-feature learned bias vectors
- Values: raw feature values
- Output: attention-weighted feature vector

Regime-conditioned biases provide default attention patterns that adapt
over time as the system observes which features correlate with successful
trades in each regime.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ATTENTION_PATH = _PROJECT_ROOT / "trained_data" / "feature_attention.json"

# Default feature names used in the scanner
DEFAULT_FEATURES = [
    "trend_strength",
    "volatility_percentile",
    "entry_score",
    "atr_pips",
    "rsi",
    "momentum",
    "spread_score",
    "volume_ratio",
    "support_distance",
    "resistance_distance",
]

# Regime-specific default attention biases (before learning)
# Higher bias = more attention to this feature in this regime
REGIME_ATTENTION_BIASES: Dict[str, Dict[str, float]] = {
    "LOW": {
        "trend_strength": 0.3,
        "volatility_percentile": -0.5,
        "entry_score": 0.2,
        "atr_pips": -0.3,
        "rsi": 0.5,         # RSI matters more in low vol (mean reversion)
        "momentum": 0.1,
        "spread_score": 0.4,  # Spreads matter when ATR is small
        "volume_ratio": 0.2,
        "support_distance": 0.4,
        "resistance_distance": 0.4,
    },
    "NORMAL": {
        "trend_strength": 0.5,
        "volatility_percentile": 0.0,
        "entry_score": 0.3,
        "atr_pips": 0.1,
        "rsi": 0.2,
        "momentum": 0.3,
        "spread_score": 0.1,
        "volume_ratio": 0.1,
        "support_distance": 0.2,
        "resistance_distance": 0.2,
    },
    "HIGH": {
        "trend_strength": 0.7,    # Trend matters most in high vol
        "volatility_percentile": 0.3,
        "entry_score": 0.2,
        "atr_pips": 0.4,
        "rsi": -0.2,              # RSI less reliable in strong trends
        "momentum": 0.6,          # Momentum crucial
        "spread_score": -0.1,
        "volume_ratio": 0.3,
        "support_distance": 0.1,
        "resistance_distance": 0.1,
    },
    "EXTREME": {
        "trend_strength": 0.4,
        "volatility_percentile": 0.5,
        "entry_score": 0.1,
        "atr_pips": 0.6,          # ATR crucial for sizing in crisis
        "rsi": -0.4,              # RSI unreliable in extreme vol
        "momentum": 0.3,
        "spread_score": 0.5,      # Spread spikes matter
        "volume_ratio": 0.4,
        "support_distance": -0.2,  # S/R breaks down in extreme vol
        "resistance_distance": -0.2,
    },
}

# Learning rate for attention bias updates
ATTENTION_LEARNING_RATE = 0.02
# Temperature for softmax (lower = sharper attention)
SOFTMAX_TEMPERATURE = 1.0


@dataclass
class AttentionResult:
    """Result of attention-based feature weighting."""

    weighted_features: Dict[str, float] = field(default_factory=dict)
    attention_weights: Dict[str, float] = field(default_factory=dict)
    regime: str = "NORMAL"
    dominant_feature: str = ""
    attention_entropy: float = 0.0  # Higher = more uniform attention

    def to_dict(self) -> dict:
        return {
            "weighted_features": {
                k: round(v, 4) for k, v in self.weighted_features.items()
            },
            "attention_weights": {
                k: round(v, 4) for k, v in self.attention_weights.items()
            },
            "regime": self.regime,
            "dominant_feature": self.dominant_feature,
            "attention_entropy": round(self.attention_entropy, 4),
        }


class FeatureAttentionLayer:
    """Lightweight attention mechanism for dynamic feature weighting.

    Computes softmax attention over features conditioned on the current
    volatility regime. Regime-specific biases provide prior knowledge
    about which features matter when, and these biases update slowly
    based on trade outcomes.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        temperature: float = SOFTMAX_TEMPERATURE,
        learning_rate: float = ATTENTION_LEARNING_RATE,
        persistence_path: Optional[Path] = None,
    ):
        self.feature_names = feature_names or DEFAULT_FEATURES.copy()
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.persistence_path = persistence_path or _ATTENTION_PATH

        # Initialize biases from defaults
        self._biases: Dict[str, Dict[str, float]] = {}
        for regime, biases in REGIME_ATTENTION_BIASES.items():
            self._biases[regime] = {
                f: biases.get(f, 0.0) for f in self.feature_names
            }

        # History for learning
        self._outcome_buffer: List[Dict[str, Any]] = []

        # Load persisted biases
        self._load_biases()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_attention(
        self,
        feature_vector: Dict[str, float],
        regime: str = "NORMAL",
    ) -> AttentionResult:
        """Compute attention-weighted features for the current regime.

        Args:
            feature_vector: {feature_name: value} dict
            regime: Current volatility regime (LOW/NORMAL/HIGH/EXTREME)

        Returns:
            AttentionResult with weighted features and attention weights
        """
        result = AttentionResult(regime=regime)

        # Get regime biases (fall back to NORMAL if unknown)
        biases = self._biases.get(regime, self._biases.get("NORMAL", {}))

        # Compute attention scores (bias + feature magnitude signal)
        features_present = [f for f in self.feature_names if f in feature_vector]
        if not features_present:
            return result

        # Raw attention logits: bias + scaled feature value
        logits: Dict[str, float] = {}
        for f in features_present:
            val = feature_vector[f]
            bias = biases.get(f, 0.0)
            # Feature value contributes to attention (normalized by magnitude)
            magnitude_signal = min(1.0, max(-1.0, val)) * 0.3
            logits[f] = bias + magnitude_signal

        # Softmax over logits
        attention_weights = self._softmax(logits)
        result.attention_weights = attention_weights

        # Apply attention weights to features
        for f in features_present:
            result.weighted_features[f] = feature_vector[f] * attention_weights[f]

        # Compute entropy (information about attention distribution)
        weights_arr = np.array(list(attention_weights.values()))
        weights_arr = weights_arr[weights_arr > 1e-10]  # Avoid log(0)
        result.attention_entropy = float(-np.sum(weights_arr * np.log(weights_arr)))

        # Dominant feature
        if attention_weights:
            result.dominant_feature = max(attention_weights, key=attention_weights.get)

        logger.debug(
            f"FeatureAttention [{regime}]: dominant={result.dominant_feature} "
            f"entropy={result.attention_entropy:.3f} "
            f"weights={', '.join(f'{k}:{v:.2f}' for k, v in sorted(attention_weights.items(), key=lambda x: -x[1])[:3])}"
        )

        return result

    def record_outcome(
        self,
        regime: str,
        feature_vector: Dict[str, float],
        attention_weights: Dict[str, float],
        trade_won: bool,
    ) -> None:
        """Record a trade outcome for attention bias learning.

        When a trade wins, slightly increase biases for features that had
        high attention. When a trade loses, decrease them. This slowly
        adapts the regime-specific biases to local market conditions.

        Args:
            regime: Regime during the trade
            feature_vector: Features at entry time
            attention_weights: Attention weights at entry time
            trade_won: Whether the trade was profitable
        """
        self._outcome_buffer.append({
            "regime": regime,
            "features": feature_vector,
            "weights": attention_weights,
            "won": trade_won,
        })

        # Batch update after 10 outcomes
        if len(self._outcome_buffer) >= 10:
            self._update_biases()

    def get_regime_priorities(self, regime: str) -> Dict[str, float]:
        """Get current attention priorities for a regime (for logging)."""
        biases = self._biases.get(regime, {})
        return dict(sorted(biases.items(), key=lambda x: -x[1]))

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _softmax(self, logits: Dict[str, float]) -> Dict[str, float]:
        """Compute softmax over a dict of logits."""
        if not logits:
            return {}
        vals = np.array(list(logits.values())) / self.temperature
        # Numerical stability: subtract max
        vals = vals - np.max(vals)
        exp_vals = np.exp(vals)
        total = np.sum(exp_vals)
        if total < 1e-10:
            # Uniform fallback
            n = len(logits)
            return {k: 1.0 / n for k in logits}
        probs = exp_vals / total
        return {k: float(p) for k, p in zip(logits.keys(), probs)}

    def _update_biases(self) -> None:
        """Update regime-specific biases from outcome buffer."""
        if not self._outcome_buffer:
            return

        # Group by regime
        regime_outcomes: Dict[str, List] = {}
        for outcome in self._outcome_buffer:
            r = outcome["regime"]
            if r not in regime_outcomes:
                regime_outcomes[r] = []
            regime_outcomes[r].append(outcome)

        for regime, outcomes in regime_outcomes.items():
            if regime not in self._biases:
                continue

            for outcome in outcomes:
                weights = outcome["weights"]
                won = outcome["won"]
                direction = 1.0 if won else -1.0

                for feature, weight in weights.items():
                    if feature in self._biases[regime]:
                        # Adjust bias proportional to attention weight and outcome
                        delta = direction * weight * self.learning_rate
                        self._biases[regime][feature] += delta
                        # Clamp biases to reasonable range
                        self._biases[regime][feature] = max(
                            -2.0, min(2.0, self._biases[regime][feature])
                        )

        self._outcome_buffer.clear()
        logger.debug(f"FeatureAttention: Updated biases from {sum(len(v) for v in regime_outcomes.values())} outcomes")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_biases(self) -> None:
        """Persist learned attention biases."""
        try:
            # Flush any remaining outcomes
            if self._outcome_buffer:
                self._update_biases()

            data = {
                "version": 1,
                "biases": self._biases,
                "feature_names": self.feature_names,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"FeatureAttention: Failed to save biases: {e}")

    def _load_biases(self) -> None:
        """Load persisted attention biases."""
        if not self.persistence_path.exists():
            return
        try:
            data = json.loads(self.persistence_path.read_text())
            loaded_biases = data.get("biases", {})
            # Merge with defaults (preserve new features, keep learned values)
            for regime in self._biases:
                if regime in loaded_biases:
                    for feature in self._biases[regime]:
                        if feature in loaded_biases[regime]:
                            self._biases[regime][feature] = loaded_biases[regime][feature]
            logger.debug("FeatureAttention: Loaded persisted biases")
        except Exception as e:
            logger.debug(f"FeatureAttention: Failed to load biases: {e}")
