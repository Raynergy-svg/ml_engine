"""
Volatility Filter Module.

Provides volatility regime detection using TCN models and ATR thresholds.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Volatility regime constants
REGIME_LOW = 0
REGIME_NORMAL = 1
REGIME_HIGH = 2
REGIME_EXTREME = 3

REGIME_NAMES = ["LOW", "NORMAL", "HIGH", "EXTREME"]


def _predict_with_named_input_if_needed(model: object, batch: np.ndarray) -> np.ndarray:
    """Match single-input Keras models by name to avoid input-structure warnings."""
    model_inputs = getattr(model, "inputs", None)
    if isinstance(model_inputs, list) and len(model_inputs) == 1:
        input_name = getattr(model_inputs[0], "name", None)
        if isinstance(input_name, str) and input_name:
            candidates = [input_name.split(":", 1)[0]]
            if candidates[0] != input_name:
                candidates.append(input_name)
            for candidate in candidates:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=".*structure of `inputs` doesn't match the expected structure.*",
                            category=UserWarning,
                        )
                        return model.predict({candidate: batch}, verbose=0)
                except (TypeError, ValueError, KeyError):
                    continue
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*structure of `inputs` doesn't match the expected structure.*",
            category=UserWarning,
        )
        return model.predict(batch, verbose=0)


@dataclass
class VolatilityResult:
    """Result of volatility check.

    Attributes:
        regime: Volatility regime (0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME)
        regime_name: Human-readable regime name
        confidence: Model confidence in regime prediction
        trade_allowed: Whether trading is allowed in this regime
        atr_pips: Current ATR in pips
        atr_percentile: ATR percentile vs historical
        reason: Explanation for trade_allowed decision
    """
    regime: int = REGIME_NORMAL
    regime_name: str = "NORMAL"
    confidence: float = 0.5
    trade_allowed: bool = True
    atr_pips: float = 0.0
    atr_percentile: float = 0.5
    reason: Optional[str] = None


class VolatilityFilter:
    """Filters trades based on volatility regime.

    Uses a combination of:
    1. TCN model for regime prediction (if available)
    2. ATR-based thresholds (fallback)

    Only allows trades in HIGH or EXTREME volatility regimes
    to ensure sufficient price movement for profitable trades.

    Regimes:
    - LOW (0): Very quiet market, skip trades
    - NORMAL (1): Average volatility, skip trades
    - HIGH (2): Good trading conditions, allow trades
    - EXTREME (3): Very volatile, allow trades (with caution)
    """

    def __init__(
        self,
        min_atr_pips: float = 8.0,
        model_dir: Optional[Path] = None,
        use_tcn: bool = True,
    ):
        """Initialize volatility filter.

        Args:
            min_atr_pips: Minimum ATR in pips to allow trading
            model_dir: Path to model artifacts
            use_tcn: Whether to use TCN model (if available)
        """
        self.min_atr_pips = min_atr_pips
        self.model_dir = model_dir or Path("trained_data/models")
        self.use_tcn = use_tcn

        # Lazy-loaded TCN model
        self._tcn_model = None
        self._tcn_scaler = None
        self._tcn_loaded = False

        # Cached modular ensemble reference
        self._modular_ensemble = None

    def _load_tcn_model(self) -> bool:
        """Load TCN volatility regime model.

        Returns:
            True if model loaded successfully
        """
        if self._tcn_loaded:
            return self._tcn_model is not None

        self._tcn_loaded = True

        tcn_path = self.model_dir / "tcn_volatility_regime.keras"
        meta_path = self.model_dir / "tcn_volatility_regime.meta.pkl"

        if not tcn_path.exists():
            logger.debug("TCN volatility model not found")
            return False

        try:
            import pickle
            from src.utils.keras_model_loader import load_keras_model

            self._tcn_model, load_meta = load_keras_model(
                str(tcn_path),
                compile=False,
            )

            if meta_path.exists():
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    self._tcn_scaler = meta.get('scaler')
                    logger.debug(f"TCN volatility model loaded: {meta.get('metrics', {})}")

            logger.info(
                "✓ TCN Volatility Regime filter loaded via %s",
                load_meta.get("approach_used"),
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to load TCN volatility model: {e}")
            return False

    def _init_modular_ensemble(self) -> bool:
        """Initialize modular ensemble for volatility check.

        Returns:
            True if ensemble available
        """
        if self._modular_ensemble is not None:
            return True

        try:
            from src.core.modular_inference import ModularEnsembleInference

            self._modular_ensemble = ModularEnsembleInference(
                model_dir=str(self.model_dir),
            )
            return self._modular_ensemble.use_tcn_volatility_filter

        except ImportError:
            logger.debug("ModularEnsembleInference not available")
            return False
        except Exception as e:
            logger.debug(f"Failed to init modular ensemble: {e}")
            return False

    def check(
        self,
        df: pd.DataFrame,
        pair: str,
        pip_value: float = 0.0001,
    ) -> VolatilityResult:
        """Check volatility regime for a pair.

        Args:
            df: DataFrame with features (must have 'atr' column)
            pair: Instrument name
            pip_value: Pip value for pair (0.01 for JPY pairs)

        Returns:
            VolatilityResult with regime and trade decision
        """
        # Calculate ATR in pips
        atr = df["atr"].iloc[-1] if "atr" in df.columns else 0.0
        atr_pips = atr / pip_value if pip_value > 0 else 0.0

        # Calculate ATR percentile
        atr_percentile = 0.5
        if "atr" in df.columns:
            atr_series = df["atr"].dropna()
            if len(atr_series) > 20:
                atr_percentile = float((atr_series < atr).mean())

        # Try TCN model first
        if self.use_tcn:
            tcn_result = self._check_with_tcn(df)
            if tcn_result is not None:
                tcn_result.atr_pips = atr_pips
                tcn_result.atr_percentile = atr_percentile
                return tcn_result

            # Try modular ensemble's TCN filter
            ensemble_result = self._check_with_ensemble(df)
            if ensemble_result is not None:
                ensemble_result.atr_pips = atr_pips
                ensemble_result.atr_percentile = atr_percentile
                return ensemble_result

        # Fallback to ATR-based regime detection
        return self._check_with_atr(atr_pips, atr_percentile)

    def _check_with_tcn(self, df: pd.DataFrame) -> Optional[VolatilityResult]:
        """Check volatility using TCN model.

        Args:
            df: DataFrame with features

        Returns:
            VolatilityResult or None if TCN unavailable
        """
        if not self._load_tcn_model():
            return None

        try:
            # Prepare features for TCN
            # Expected features: normalized technical indicators
            feature_cols = [
                'returns', 'volatility', 'atr', 'rsi', 'macd',
                'adx', 'bb_width', 'momentum'
            ]

            available_cols = [c for c in feature_cols if c in df.columns]
            if len(available_cols) < 4:
                return None

            X = df[available_cols].tail(60).values

            if self._tcn_scaler is not None:
                X = self._tcn_scaler.transform(X)

            X = X.reshape(1, X.shape[0], X.shape[1])

            # Predict regime
            probs = _predict_with_named_input_if_needed(self._tcn_model, X)[0]
            regime = int(np.argmax(probs))
            confidence = float(probs[regime])

            # Only allow trades in HIGH or EXTREME
            trade_allowed = regime >= REGIME_HIGH

            reason = f"TCN regime: {REGIME_NAMES[regime]}"
            if not trade_allowed:
                reason += " (low volatility, skipping)"

            return VolatilityResult(
                regime=regime,
                regime_name=REGIME_NAMES[regime],
                confidence=confidence,
                trade_allowed=trade_allowed,
                reason=reason,
            )

        except Exception as e:
            logger.debug(f"TCN volatility check failed: {e}")
            return None

    def _check_with_ensemble(self, df: pd.DataFrame) -> Optional[VolatilityResult]:
        """Check volatility using modular ensemble's TCN filter.

        Args:
            df: DataFrame with features

        Returns:
            VolatilityResult or None if unavailable
        """
        if not self._init_modular_ensemble():
            return None

        if not self._modular_ensemble.use_tcn_volatility_filter:
            return None

        try:
            regime, confidence = self._modular_ensemble.get_volatility_regime(df)

            trade_allowed = regime >= REGIME_HIGH

            reason = f"Ensemble TCN: {REGIME_NAMES[regime]}"
            if not trade_allowed:
                reason += " (insufficient volatility)"

            return VolatilityResult(
                regime=regime,
                regime_name=REGIME_NAMES[regime],
                confidence=confidence,
                trade_allowed=trade_allowed,
                reason=reason,
            )

        except Exception as e:
            logger.debug(f"Ensemble volatility check failed: {e}")
            return None

    def _check_with_atr(
        self,
        atr_pips: float,
        atr_percentile: float,
    ) -> VolatilityResult:
        """Check volatility using ATR thresholds.

        Args:
            atr_pips: Current ATR in pips
            atr_percentile: ATR percentile vs historical

        Returns:
            VolatilityResult based on ATR analysis
        """
        # Determine regime from ATR percentile
        if atr_percentile < 0.25:
            regime = REGIME_LOW
        elif atr_percentile < 0.50:
            regime = REGIME_NORMAL
        elif atr_percentile < 0.80:
            regime = REGIME_HIGH
        else:
            regime = REGIME_EXTREME

        # Check minimum ATR threshold
        trade_allowed = atr_pips >= self.min_atr_pips and regime >= REGIME_HIGH

        reason = f"ATR: {atr_pips:.1f} pips ({atr_percentile:.0%} percentile)"
        if atr_pips < self.min_atr_pips:
            reason += f" (below {self.min_atr_pips} pip minimum)"
        elif regime < REGIME_HIGH:
            reason += " (low volatility regime)"

        return VolatilityResult(
            regime=regime,
            regime_name=REGIME_NAMES[regime],
            confidence=0.7,  # Fixed confidence for ATR-based
            trade_allowed=trade_allowed,
            atr_pips=atr_pips,
            atr_percentile=atr_percentile,
            reason=reason,
        )

    def filter_results(
        self,
        results: list,
        data_cache: dict,
        pip_values: dict,
    ) -> list:
        """Filter results by volatility regime.

        Args:
            results: List of PairAnalysis results
            data_cache: Dict mapping pair -> DataFrame
            pip_values: Dict mapping pair -> pip value

        Returns:
            Filtered list with volatility info added
        """
        filtered = []

        for result in results:
            pair = result.pair

            if pair not in data_cache:
                filtered.append(result)
                continue

            df = data_cache[pair]
            pip_value = pip_values.get(pair, 0.0001)

            vol_result = self.check(df, pair, pip_value)

            # Update result with volatility info
            result.volatility_regime = vol_result.regime_name
            result.atr_pips = vol_result.atr_pips

            # Skip if volatility too low
            if not vol_result.trade_allowed:
                result.gates_passed = False
                result.volatility_gate_passed = False
                if not result.error:
                    result.error = vol_result.reason
            else:
                result.volatility_gate_passed = True

            filtered.append(result)

        return filtered
