"""
Inference wrapper for the SOTA raw-sequence model.

Exposes a ModularEnsembleInference-compatible interface so the scanner
engine can drop it in without changing orchestration code.

    from src.sota_core import SOTAInference, SOTAInferenceConfig
    inference = SOTAInference(config)
    signal = inference.predict(pair="EUR_USD", df_raw=df)

CRIT-1 FIX: Model is loaded ONCE at instantiation time.  If the load fails,
the failure is terminal — _load_attempted prevents retries on every predict()
call.  This avoids grinding the scanner to a halt in watch mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.core.modular_inference import TradeSignal
from .raw_sequence_model import RawSequenceModel, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class SOTAInferenceConfig:
    """Configuration for SOTA inference."""

    model_path: str = "trained_data/models/sota_finetuned/sota_model.keras"
    seq_len: int = 128
    confidence_threshold: float = 0.55
    regime_filter: Optional[str] = None  # e.g. "EXTREME" to block trades in extreme vol
    fallback_to_hold: bool = True
    # Backward-compat fields that the scanner expects
    min_tcn_probability: float = 0.50
    min_confidence: float = 0.55
    max_drawdown_pct: float = 0.025
    max_uncertainty_std: float = 0.15
    use_tcn_volatility_filter: bool = True
    min_volatility_regime: int = 0
    tcn_required: bool = True
    final_score_threshold: float = 0.55
    log_rl_decisions: bool = False
    enforce_model_contracts: bool = False
    require_direction_model: bool = True
    require_volatility_model: bool = False
    position_size_source: str = "heuristic"
    exit_timing_confidence: float = 0.7


class SOTAInference:
    """Drop-in replacement for ModularEnsembleInference.

    Loads the SOTA raw-sequence model once at __init__ time.  If the load
    fails, the instance is permanently in fallback mode — predict() returns
    HOLD signals without retrying disk I/O.

    MED-2: Confidence threshold is adaptive.  Starts wide (0.40/0.60) and
    narrows toward the configured value as calibration history grows.
    """

    REGIME_NAMES = ["LOW", "NORMAL", "HIGH", "EXTREME"]
    _WARMUP_SAMPLES = 500  # threshold stays wide until this many predictions

    def __init__(self, config: Optional[SOTAInferenceConfig] = None):
        self.cfg = config or SOTAInferenceConfig()
        self.model = RawSequenceModel(ModelConfig(seq_len=self.cfg.seq_len))
        self._loaded = False
        self._load_attempted = False
        self._model_version = "unknown"
        # MED-2: calibration history for adaptive threshold
        self._prediction_history: list[float] = []
        # CRIT-1 FIX: eager load at construction time; failures are terminal.
        self._attempt_load()

    def _attempt_load(self) -> None:
        """Try to load the fine-tuned model once.  Never retries."""
        if self._load_attempted:
            return
        self._load_attempted = True
        ok = self.model.load(self.cfg.model_path)
        if ok:
            self._loaded = True
            self._model_version = "sota_v1"
            logger.info("SOTAInference: model loaded from %s", self.cfg.model_path)
        else:
            logger.warning(
                "SOTAInference: model load FAILED from %s — "
                "permanently in fallback mode (HOLD signals only)",
                self.cfg.model_path,
            )

    def load_models(self) -> bool:
        """Backward-compat entry point.  Idempotent no-op after __init__."""
        self._attempt_load()
        return self._loaded

    def predict(
        self,
        pair: str,
        df_raw: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> TradeSignal:
        """Run inference and return a TradeSignal.

        CRIT-1: If model never loaded, returns HOLD immediately with no
        disk I/O.  Safe to call in tight watch-mode loops.
        """
        if not self._loaded:
            logger.debug("SOTAInference predict skipped: model not loaded")
            return self._make_hold_signal("model_not_loaded")

        if df_raw is None or df_raw.empty:
            return self._make_hold_signal("no_data")

        # CRIT-2 FIX: pass normalization_stats path so df_to_tensor can
        # use fixed per-pair stats instead of a rolling window.
        stats_path = kwargs.get("normalization_stats_path") or self._guess_stats_path(pair)
        tensor = RawSequenceModel.df_to_tensor(
            df_raw, self.cfg.seq_len, normalization_stats_path=stats_path
        )
        if tensor is None:
            return self._make_hold_signal("insufficient_data")

        # Forward pass
        try:
            dir_prob, regime_probs = self.model.model.predict(tensor, verbose=0)
        except Exception as e:
            logger.warning("SOTAInference predict failed: %s", e)
            return self._make_hold_signal("prediction_error")

        direction_prob = float(dir_prob[0, 0])
        regime_idx = int(np.argmax(regime_probs[0]))
        regime_name = self.REGIME_NAMES[regime_idx] if 0 <= regime_idx < 4 else "UNKNOWN"

        # MED-2: Adaptive confidence threshold
        # During warmup, use a wide threshold so the model generates
        # signals and builds calibration history.  After warmup, use the
        # configured threshold.
        self._prediction_history.append(direction_prob)
        if len(self._prediction_history) < self._WARMUP_SAMPLES:
            # Linearly interpolate from wide (0.40) to configured threshold
            warmup_frac = len(self._prediction_history) / self._WARMUP_SAMPLES
            adaptive_thresh = 0.40 + (self.cfg.confidence_threshold - 0.40) * warmup_frac
        else:
            adaptive_thresh = self.cfg.confidence_threshold

        if direction_prob >= adaptive_thresh:
            direction = "LONG"
            confidence = direction_prob
        elif direction_prob <= (1.0 - adaptive_thresh):
            direction = "SHORT"
            confidence = 1.0 - direction_prob
        else:
            direction = "HOLD"
            confidence = max(direction_prob, 1.0 - direction_prob)

        # Regime filter
        if self.cfg.regime_filter and regime_name == self.cfg.regime_filter:
            direction = "HOLD"
            reason = f"regime_filter_{regime_name.lower()}"
        else:
            reason = "sota_signal"

        signal = TradeSignal(
            trade=direction in ("LONG", "SHORT"),
            direction=direction,
            size=0.0,
            confidence=confidence,
            tcn_probability=direction_prob,  # backward-compat
            reason=reason,
            metadata={
                "model": "sota_raw_sequence",
                "model_version": self._model_version,
                "direction_probability": direction_prob,
                "regime": regime_name,
                "regime_probs": regime_probs[0].tolist(),
                "confidence_threshold": self.cfg.confidence_threshold,
            },
        )
        return signal

    def predict_regime_only(
        self,
        pair: str,
        df_raw: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> Tuple[str, np.ndarray]:
        """Run inference and return ONLY the regime classification.

        Returns (regime_name, regime_probs).  If model never loaded or
        data insufficient, returns ("UNKNOWN", uniform).  No direction
        computation, no adaptive threshold, no TradeSignal wrapping.
        """
        if not self._loaded or df_raw is None or df_raw.empty:
            return ("UNKNOWN", np.full(4, 0.25, dtype=np.float32))

        stats_path = kwargs.get("normalization_stats_path") or self._guess_stats_path(pair)
        tensor = RawSequenceModel.df_to_tensor(
            df_raw, self.cfg.seq_len, normalization_stats_path=stats_path
        )
        if tensor is None:
            return ("UNKNOWN", np.full(4, 0.25, dtype=np.float32))

        try:
            _, regime_probs = self.model.model.predict(tensor, verbose=0)
        except Exception as e:
            logger.warning("SOTAInference predict_regime_only failed: %s", e)
            return ("UNKNOWN", np.full(4, 0.25, dtype=np.float32))

        regime_idx = int(np.argmax(regime_probs[0]))
        regime_name = (
            self.REGIME_NAMES[regime_idx]
            if 0 <= regime_idx < 4
            else "UNKNOWN"
        )
        return (regime_name, regime_probs[0])

    def _make_hold_signal(self, reason: str) -> TradeSignal:
        return TradeSignal(
            trade=False,
            direction="HOLD",
            size=0.0,
            confidence=0.0,
            tcn_probability=0.5,
            reason=reason,
            metadata={"model": "sota_raw_sequence", "model_version": self._model_version},
        )

    def _guess_stats_path(self, pair: str) -> Optional[str]:
        """Guess the normalization-stats path for a pair."""
        candidate = f"trained_data/models/{pair}/normalization_stats.json"
        if Path(candidate).exists():
            return candidate
        return None

    @property
    def tcn(self) -> Optional[Any]:
        """Backward-compat property: return self if loaded so scanner sees a 'direction model'."""
        return self if self._loaded else None
