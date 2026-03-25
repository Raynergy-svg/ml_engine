"""Model bandit routing for pair-specific ensemble selection.

US-106: Activate model_bandit (US-093) routing in the inference pipeline.
Currently all 3 models (TCN/Ridge/RF) always run and average. Route to
the best-performing model per pair/regime using bandit selection, with
1 fallback. Reduces compute by ~33% and improves signal quality.

Approach:
1. Query model_bandit for arm probabilities per pair/regime
2. Select primary model (highest probability) + fallback (second highest)
3. Blend predictions with primary-weighted average
4. Record outcome back to bandit for reward update
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "model_routing_log.json"

# Routing parameters
PRIMARY_WEIGHT = 0.70  # Primary model gets 70% weight in blend
FALLBACK_WEIGHT = 0.30  # Fallback gets 30%
MIN_BANDIT_CONFIDENCE = 0.40  # If bandit probability < this, use all models
AVAILABLE_MODELS = ["TCN", "Ridge", "RF"]


@dataclass
class ModelSelection:
    """Result of model routing decision."""

    primary_model: str = ""
    fallback_model: str = ""
    primary_probability: float = 0.0
    fallback_probability: float = 0.0
    strategy: str = "ROUTED"  # ROUTED or ALL_MODELS
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "primary_probability": round(self.primary_probability, 4),
            "fallback_probability": round(self.fallback_probability, 4),
            "strategy": self.strategy,
            "reason": self.reason,
        }


class ModelRouter:
    """Routes inference to best-performing model per pair/regime.

    Bridges model_bandit (which tracks arm statistics) and the inference
    pipeline (which runs models). Selects primary + fallback model,
    blends their predictions, and records outcomes.

    Usage:
        router = ModelRouter()
        selection = router.select_models("EUR_USD", "NORMAL")
        # selection.primary_model = "TCN", selection.fallback_model = "Ridge"
        blended = router.blend_predictions(0.72, 0.58, 0.70)
        # blended = 0.678
        router.record_outcome("EUR_USD", "NORMAL", "TCN", pnl_pips=12.5)
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None,
    ):
        self.persistence_path = persistence_path or _LOG_PATH

        # Bandit reference (lazy-loaded)
        self._bandit = None
        self._bandit_loaded = False

        # Routing history
        self._decisions: List[Dict[str, Any]] = []
        self._total_routed: int = 0
        self._model_selection_counts: Dict[str, int] = {m: 0 for m in AVAILABLE_MODELS}

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_models(
        self,
        pair: str,
        regime: str = "NORMAL",
    ) -> ModelSelection:
        """Select primary and fallback models for a pair/regime.

        Args:
            pair: Currency pair.
            regime: Current volatility regime.

        Returns:
            ModelSelection with primary and fallback model names.
        """
        self._ensure_bandit_loaded()
        self._total_routed += 1

        # Get arm probabilities from bandit
        probs = self._get_bandit_probabilities(pair, regime)

        if not probs or max(probs.values()) < MIN_BANDIT_CONFIDENCE:
            # Bandit not confident enough — use all models
            return ModelSelection(
                primary_model="ALL",
                fallback_model="ALL",
                primary_probability=1.0 / len(AVAILABLE_MODELS),
                fallback_probability=1.0 / len(AVAILABLE_MODELS),
                strategy="ALL_MODELS",
                reason=f"Bandit confidence too low (max={max(probs.values()) if probs else 0:.2f})",
            )

        # Sort by probability
        sorted_models = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_models[0]
        fallback = sorted_models[1] if len(sorted_models) > 1 else sorted_models[0]

        # Update selection counts
        self._model_selection_counts[primary[0]] = (
            self._model_selection_counts.get(primary[0], 0) + 1
        )

        selection = ModelSelection(
            primary_model=primary[0],
            fallback_model=fallback[0],
            primary_probability=primary[1],
            fallback_probability=fallback[1],
            strategy="ROUTED",
            reason=f"Bandit selected {primary[0]} (p={primary[1]:.3f})",
        )

        # Log decision
        self._decisions.append({
            "pair": pair,
            "regime": regime,
            "selection": selection.to_dict(),
            "all_probabilities": {k: round(v, 4) for k, v in probs.items()},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._decisions) > 300:
            self._decisions = self._decisions[-150:]

        return selection

    def blend_predictions(
        self,
        primary_pred: float,
        fallback_pred: float,
        primary_weight: float = PRIMARY_WEIGHT,
    ) -> float:
        """Blend primary and fallback model predictions.

        Args:
            primary_pred: Primary model confidence.
            fallback_pred: Fallback model confidence.
            primary_weight: Weight for primary model.

        Returns:
            Blended confidence score.
        """
        fallback_weight = 1.0 - primary_weight
        blended = primary_weight * primary_pred + fallback_weight * fallback_pred
        return round(float(np.clip(blended, 0.0, 1.0)), 4)

    def record_outcome(
        self,
        pair: str,
        regime: str,
        selected_model: str,
        pnl_pips: float,
    ) -> None:
        """Record trade outcome for bandit reward update.

        Args:
            pair: Currency pair.
            regime: Regime at time of selection.
            selected_model: Which model was primary.
            pnl_pips: Trade P/L in pips.
        """
        self._ensure_bandit_loaded()

        if self._bandit is not None:
            try:
                # Bandit normalizes P/L internally via tanh squash
                self._bandit.update_reward(
                    selected_model,
                    pnl_pips,
                )
                logger.debug(
                    f"ModelRouter: Recorded {selected_model} outcome "
                    f"(pnl={pnl_pips:.1f} pips)"
                )
            except Exception as e:
                logger.debug(f"ModelRouter: Failed to record reward: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get router status for observability."""
        return {
            "total_routed": self._total_routed,
            "model_selection_counts": dict(self._model_selection_counts),
            "bandit_loaded": self._bandit_loaded,
            "recent_decisions": self._decisions[-5:],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_bandit_loaded(self) -> None:
        """Lazy-load the model bandit."""
        if self._bandit_loaded:
            return
        try:
            from src.scanner.automation.model_bandit import ModelBanditSelector
            self._bandit = ModelBanditSelector()
            self._bandit_loaded = True
        except Exception as e:
            logger.debug(f"ModelRouter: Bandit import failed: {e}")
            self._bandit_loaded = True  # Don't retry

    def _get_bandit_probabilities(
        self,
        pair: str,
        regime: str,
    ) -> Dict[str, float]:
        """Get arm probabilities from bandit."""
        if self._bandit is None:
            # Return uniform if no bandit
            return {m: 1.0 / len(AVAILABLE_MODELS) for m in AVAILABLE_MODELS}

        try:
            probs = self._bandit.get_model_probabilities()
            if isinstance(probs, dict):
                return probs
        except Exception as e:
            logger.debug(f"ModelRouter: Failed to get bandit probs: {e}")

        return {m: 1.0 / len(AVAILABLE_MODELS) for m in AVAILABLE_MODELS}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist routing state."""
        try:
            data = {
                "version": 1,
                "total_routed": self._total_routed,
                "model_selection_counts": dict(self._model_selection_counts),
                "decisions": self._decisions[-150:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"ModelRouter: Failed to save: {e}")

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
                self._total_routed = data.get("total_routed", 0)
                self._model_selection_counts = data.get("model_selection_counts", {})
                self._decisions = data.get("decisions", [])
        except Exception as e:
            logger.debug(f"ModelRouter: Failed to load: {e}")
