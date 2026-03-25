"""Temporal attention agent consensus feedback loop.

US-108: Create bidirectional feedback between agent consensus quality
and temporal attention weights. After agent votes are collected, compute
per-timeframe agreement quality. Feed this back to temporal_attention
logit updates so the system learns which timeframes produce the most
consistent agent consensus.

Approach:
1. Receive agent votes and their associated timeframe signals
2. Compute agreement quality per timeframe (how much agents agree)
3. Feed quality scores to temporal_attention for logit adjustment
4. Track convergence over time
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "attention_feedback_log.json"

# Feedback parameters
QUALITY_LR = 0.03  # Learning rate for attention logit updates
MIN_AGENTS_FOR_QUALITY = 3  # Minimum agents voting to compute quality
CONVERGENCE_WINDOW = 20  # Rolling window to detect convergence
CONVERGENCE_THRESHOLD = 0.05  # Max std of quality scores = converged


class AttentionFeedbackLoop:
    """Bidirectional feedback between agent consensus and temporal attention.

    Connects the agent voting pipeline (which produces per-timeframe
    consensus) with the temporal attention layer (which weights timeframes).
    High-consensus timeframes get attention boosts; low-consensus get penalized.

    Usage:
        feedback = AttentionFeedbackLoop()
        quality = feedback.compute_timeframe_quality(agent_votes, tf_signals)
        feedback.update_attention(quality, regime="NORMAL")
    """

    def __init__(
        self,
        learning_rate: float = QUALITY_LR,
        persistence_path: Optional[Path] = None,
    ):
        self.learning_rate = learning_rate
        self.persistence_path = persistence_path or _LOG_PATH

        # Temporal attention reference (lazy-loaded)
        self._attention = None
        self._attention_loaded = False

        # Quality history per regime per timeframe
        self._quality_history: Dict[str, Dict[str, List[float]]] = {}

        # Feedback log
        self._feedback_log: List[Dict[str, Any]] = []
        self._total_updates: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_timeframe_quality(
        self,
        agent_votes: Dict[str, str],
        timeframe_signals: Dict[int, Dict[str, float]],
    ) -> Dict[int, float]:
        """Compute per-timeframe agreement quality from agent votes.

        Agent votes represent directional consensus. For each timeframe,
        measure how much the agents that used that timeframe's signal
        agree with each other.

        Args:
            agent_votes: {agent_name: "BUY"/"SELL"/"HOLD"} votes.
            timeframe_signals: {timeframe: {"direction": float, "strength": float}}.

        Returns:
            Dict mapping timeframe → quality score [0.0, 1.0].
        """
        if len(agent_votes) < MIN_AGENTS_FOR_QUALITY:
            return {}

        # Count vote distribution
        buy_count = sum(1 for v in agent_votes.values() if v == "BUY")
        sell_count = sum(1 for v in agent_votes.values() if v == "SELL")
        total_votes = len(agent_votes)

        # Overall consensus: how much agents agree
        consensus = abs(buy_count - sell_count) / max(total_votes, 1)

        quality_scores: Dict[int, float] = {}

        for tf, signal in timeframe_signals.items():
            direction = signal.get("direction", 0.0)
            strength = signal.get("strength", 0.0)

            # Quality = consensus * alignment with timeframe signal
            # If agents agree AND their direction matches timeframe signal → high quality
            majority_dir = 1.0 if buy_count > sell_count else -1.0
            alignment = majority_dir * direction  # +1 if same direction

            # Quality score: consensus-weighted alignment with signal strength
            quality = consensus * max(0.0, alignment) * strength

            quality_scores[tf] = round(float(np.clip(quality, 0.0, 1.0)), 4)

        return quality_scores

    def update_attention(
        self,
        quality_scores: Dict[int, float],
        regime: str = "NORMAL",
    ) -> None:
        """Feed quality scores back to temporal attention logits.

        High-quality timeframes get logit boosts. Low-quality get penalties.

        Args:
            quality_scores: {timeframe: quality} from compute_timeframe_quality.
            regime: Current regime.
        """
        if not quality_scores:
            return

        self._ensure_attention_loaded()

        # Update temporal attention logits
        if self._attention is not None:
            try:
                logits = self._attention._logits[regime]
                for tf, quality in quality_scores.items():
                    if tf not in logits:
                        logits[tf] = 0.0

                    # Boost high-quality timeframes, penalize low-quality
                    delta = self.learning_rate * (quality - 0.5)  # Centered at 0.5
                    delta = float(np.clip(delta, -0.10, 0.10))
                    logits[tf] += delta

            except Exception as e:
                logger.debug(f"AttentionFeedback: Logit update failed: {e}")

        # Track quality history
        if regime not in self._quality_history:
            self._quality_history[regime] = {}

        for tf, quality in quality_scores.items():
            tf_key = str(tf)
            if tf_key not in self._quality_history[regime]:
                self._quality_history[regime][tf_key] = []
            self._quality_history[regime][tf_key].append(quality)
            # Keep rolling window
            if len(self._quality_history[regime][tf_key]) > CONVERGENCE_WINDOW * 2:
                self._quality_history[regime][tf_key] = (
                    self._quality_history[regime][tf_key][-CONVERGENCE_WINDOW:]
                )

        self._total_updates += 1

        # Log feedback
        self._feedback_log.append({
            "regime": regime,
            "quality_scores": {str(k): v for k, v in quality_scores.items()},
            "converged": self.is_converged(regime),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._feedback_log) > 300:
            self._feedback_log = self._feedback_log[-150:]

    def is_converged(self, regime: str = "NORMAL") -> bool:
        """Check if attention weights have converged for a regime.

        Convergence = quality scores have low variance over recent window.

        Args:
            regime: Regime to check.

        Returns:
            True if attention weights appear converged.
        """
        if regime not in self._quality_history:
            return False

        for tf_key, history in self._quality_history[regime].items():
            if len(history) < CONVERGENCE_WINDOW:
                return False
            recent = history[-CONVERGENCE_WINDOW:]
            if np.std(recent) > CONVERGENCE_THRESHOLD:
                return False

        return True

    def get_convergence_report(self) -> Dict[str, Any]:
        """Get convergence status for all regimes."""
        report: Dict[str, Any] = {}
        for regime in self._quality_history:
            tf_stats = {}
            for tf_key, history in self._quality_history[regime].items():
                if history:
                    recent = history[-CONVERGENCE_WINDOW:]
                    tf_stats[tf_key] = {
                        "mean": round(float(np.mean(recent)), 4),
                        "std": round(float(np.std(recent)), 4),
                        "n_samples": len(history),
                    }
            report[regime] = {
                "converged": self.is_converged(regime),
                "timeframes": tf_stats,
            }
        return report

    def get_status(self) -> Dict[str, Any]:
        """Get feedback loop status for observability."""
        return {
            "total_updates": self._total_updates,
            "attention_loaded": self._attention_loaded,
            "regimes_tracked": list(self._quality_history.keys()),
            "convergence": {
                r: self.is_converged(r) for r in self._quality_history
            },
            "recent_feedback": self._feedback_log[-5:],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_attention_loaded(self) -> None:
        """Lazy-load temporal attention layer."""
        if self._attention_loaded:
            return
        try:
            from src.scanner.automation.temporal_attention import TemporalAttentionLayer
            self._attention = TemporalAttentionLayer()
            self._attention_loaded = True
        except Exception as e:
            logger.debug(f"AttentionFeedback: Attention import failed: {e}")
            self._attention_loaded = True  # Don't retry

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist feedback state."""
        try:
            data = {
                "version": 1,
                "total_updates": self._total_updates,
                "quality_history": {
                    regime: {
                        tf: scores[-CONVERGENCE_WINDOW:]
                        for tf, scores in tfs.items()
                    }
                    for regime, tfs in self._quality_history.items()
                },
                "feedback_log": self._feedback_log[-150:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"AttentionFeedback: Failed to save: {e}")

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

            if isinstance(data, dict):
                self._total_updates = data.get("total_updates", 0)
                self._quality_history = data.get("quality_history", {})
                self._feedback_log = data.get("feedback_log", [])
        except Exception as e:
            logger.debug(f"AttentionFeedback: Failed to load: {e}")
