"""Hybrid Inference with Out-of-Distribution Fallback Safety.

US-008: Ensembles foundation model + custom CNN-Transformer + XGBoost,
with OOD detection via input reconstruction error.

When the foundation model is >2σ out-of-distribution, the ensemble
automatically falls back to the custom model (which is more robust
on the training distribution).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.sota_core.raw_sequence_model import RawSequenceModel, ModelConfig

logger = logging.getLogger(__name__)

# Optional XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    xgb = None  # type: ignore[assignment]


@dataclass
class HybridInferenceConfig:
    """Configuration for hybrid inference."""

    seq_len: int = 128
    confidence_threshold: float = 0.55
    # OOD detection
    ood_threshold_sigma: float = 2.0
    enable_ood_fallback: bool = True
    # Weights (sum to 1.0)
    weight_foundation: float = 0.5
    weight_custom: float = 0.3
    weight_xgboost: float = 0.2
    # Fallback
    fallback_to_custom_on_error: bool = True


class HybridInference:
    """Ensemble inference with OOD-aware fallback.

    Combines:
      1. Foundation model (TimesFM or similar) — best on-distribution
      2. Custom CNN-Transformer — trained on your specific data
      3. XGBoost — fast, interpretable, good on tabular features

    OOD detection: compute reconstruction error of input vs expected
    distribution. If error > threshold, reduce foundation weight to 0
    and renormalize.
    """

    def __init__(self, config: Optional[HybridInferenceConfig] = None):
        self.cfg = config or HybridInferenceConfig()
        self._foundation: Any = None
        self._custom: Optional[RawSequenceModel] = None
        self._xgb: Any = None
        self._normalization_stats: Optional[Dict[str, Any]] = None

        # Load custom model (always available as baseline)
        self._load_custom()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_custom(self) -> None:
        """Load the custom CNN-Transformer."""
        try:
            self._custom = RawSequenceModel(ModelConfig(seq_len=self.cfg.seq_len))
            # Attempt to load weights if they exist
            custom_path = Path("trained_data/models/sota_finetuned/sota_model.keras")
            if custom_path.exists():
                ok = self._custom.load(str(custom_path))
                if ok:
                    logger.info("HybridInference: custom model loaded")
                else:
                    logger.warning("HybridInference: custom model load returned False")
            else:
                logger.debug("HybridInference: custom model weights not found at %s", custom_path)
        except Exception as exc:
            logger.warning("HybridInference: failed to load custom model: %s", exc)
            self._custom = None

    def register_foundation(self, model: Any) -> None:
        """Register a foundation model (e.g., TimesFMAdapter)."""
        self._foundation = model
        logger.info("HybridInference: foundation model registered")

    def register_xgboost(self, model: Any) -> None:
        """Register an XGBoost model."""
        self._xgb = model
        logger.info("HybridInference: XGBoost model registered")

    def set_normalization_stats(self, stats: Dict[str, Any]) -> None:
        """Set per-channel mean/std for OOD detection."""
        self._normalization_stats = stats

    # ------------------------------------------------------------------
    # OOD Detection
    # ------------------------------------------------------------------
    def _compute_ood_score(self, tensor: np.ndarray) -> float:
        """Compute reconstruction-based OOD score.

        For now, uses a simple z-score of input values against
        stored normalization stats.  Higher = more OOD.
        """
        if self._normalization_stats is None:
            return 0.0  # no stats = assume in-distribution
        means = self._normalization_stats.get("mean")
        stds = self._normalization_stats.get("std")
        if means is None or stds is None:
            return 0.0
        z = (tensor - means) / (stds + 1e-8)
        # Max absolute z-score across channels as OOD metric
        return float(np.max(np.abs(z)))

    def _is_ood(self, tensor: np.ndarray) -> bool:
        """Return True if input is out-of-distribution."""
        if not self.cfg.enable_ood_fallback:
            return False
        score = self._compute_ood_score(tensor)
        return score > self.cfg.ood_threshold_sigma

    # ------------------------------------------------------------------
    # Ensemble prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        pair: str,
        df_raw: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run hybrid ensemble and return signal dict.

        Returns:
            {"direction": str, "confidence": float, "regime": str,
             "ood_detected": bool, "weights_used": dict, "source": str}
        """
        if df_raw is None or df_raw.empty:
            return {
                "direction": "HOLD",
                "confidence": 0.0,
                "regime": "UNKNOWN",
                "ood_detected": False,
                "weights_used": {},
                "source": "no_data",
            }

        # Build tensor for OOD check
        tensor = RawSequenceModel.df_to_tensor(df_raw, self.cfg.seq_len)
        if tensor is None:
            return {
                "direction": "HOLD",
                "confidence": 0.0,
                "regime": "UNKNOWN",
                "ood_detected": False,
                "weights_used": {},
                "source": "insufficient_data",
            }

        ood = self._is_ood(tensor)

        # Get predictions from each model
        preds: List[Tuple[float, float]] = []  # (direction_prob, weight)
        weights: Dict[str, float] = {}

        # Foundation model
        if self._foundation is not None and not ood:
            try:
                f_pred = self._foundation.predict(pair, df_raw, **kwargs)
                f_conf = float(f_pred.get("confidence", 0.5))
                preds.append((f_conf, self.cfg.weight_foundation))
                weights["foundation"] = self.cfg.weight_foundation
            except Exception as exc:
                logger.debug("Foundation model predict failed: %s", exc)

        # Custom model
        if self._custom is not None:
            try:
                c_pred = self._custom.model.predict(tensor, verbose=0)
                c_conf = float(c_pred[0][0]) if hasattr(c_pred, "shape") else 0.5
                w = self.cfg.weight_custom if not ood else 0.6
                preds.append((c_conf, w))
                weights["custom"] = w
            except Exception as exc:
                logger.debug("Custom model predict failed: %s", exc)

        # XGBoost
        if self._xgb is not None:
            try:
                # Flatten last seq_len bars into feature vector
                features = tensor.reshape(1, -1)
                x_pred = self._xgb.predict_proba(features)[0]
                x_conf = float(x_pred[1]) if len(x_pred) > 1 else 0.5
                preds.append((x_conf, self.cfg.weight_xgboost))
                weights["xgboost"] = self.cfg.weight_xgboost
            except Exception as exc:
                logger.debug("XGBoost predict failed: %s", exc)

        if not preds:
            return {
                "direction": "HOLD",
                "confidence": 0.0,
                "regime": "UNKNOWN",
                "ood_detected": ood,
                "weights_used": weights,
                "source": "all_models_failed",
            }

        # Weighted average of probabilities
        total_weight = sum(w for _, w in preds)
        if total_weight <= 0:
            return {
                "direction": "HOLD",
                "confidence": 0.0,
                "regime": "UNKNOWN",
                "ood_detected": ood,
                "weights_used": weights,
                "source": "zero_weight",
            }

        ensemble_prob = sum(p * w for p, w in preds) / total_weight

        # Direction
        if ensemble_prob >= self.cfg.confidence_threshold:
            direction = "LONG"
        elif ensemble_prob <= (1.0 - self.cfg.confidence_threshold):
            direction = "SHORT"
        else:
            direction = "HOLD"

        # Regime (from custom model if available)
        regime = "NORMAL"
        if self._custom is not None:
            try:
                _, regime_probs = self._custom.model.predict(tensor, verbose=0)
                regime_idx = int(np.argmax(regime_probs[0]))
                regime_names = ["LOW", "NORMAL", "HIGH", "EXTREME"]
                regime = regime_names[regime_idx] if 0 <= regime_idx < 4 else "UNKNOWN"
            except Exception:
                pass

        return {
            "direction": direction,
            "confidence": ensemble_prob,
            "regime": regime,
            "ood_detected": ood,
            "weights_used": weights,
            "source": "hybrid_ensemble",
        }
