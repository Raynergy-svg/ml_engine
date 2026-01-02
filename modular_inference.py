"""
Modular Ensemble Inference Pipeline.

Combines predictions from 4 independent models using gated logic:
1. TCN/Transformer gives direction (long/short)
2. Ridge provides confidence score (0-100)
3. XGBoost checks momentum (fresh or accelerating?)
4. Random Forest assesses risk (expected drawdown)

Trade only if ALL gates pass:
- Ridge confidence > 75
- XGBoost momentum fresh OR accelerating
- RF expected drawdown < stop loss

Position sized for 2% max risk.

IMPORTANT: Uses NORMALIZED features (returns, z-scores, ratios) that are
instrument-agnostic. Models trained on GBP_USD work on USD_JPY, EUR_USD, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Import normalized feature computation from data loaders
from modular_data_loaders import compute_normalized_features, get_normalized_feature_names

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference gates.
    
    NOTE: All thresholds use NORMALIZED values (percentages, not pips)
    for instrument-agnostic inference.
    """
    # Confidence gate
    min_confidence: float = 75.0  # 0-100 scale
    
    # Momentum gate
    min_momentum: float = 0.3  # 0-1 scale
    require_fresh_or_accel: bool = True  # Must have either fresh momentum or acceleration
    
    # Risk gate (NORMALIZED - percentage based, not pips)
    # max_drawdown_pct: 0.005 = 0.5% max expected drawdown
    # This is instrument-agnostic and works across all pairs
    max_drawdown_pct: float = 0.01  # 1% max expected drawdown (normalized)
    max_streak_prob: float = 0.3  # Max acceptable streak continuation probability
    
    # Legacy pip-based (kept for backward compatibility)
    max_drawdown_pips: float = 55.0  # Deprecated, use max_drawdown_pct
    
    # Position sizing
    risk_per_trade_pct: float = 0.02  # 2% risk per trade
    account_equity: float = 10000.0  # Default equity for sizing
    pip_value: float = 1.0  # Value per pip (varies by pair)


@dataclass
class TradeSignal:
    """Result of inference pipeline."""
    trade: bool
    direction: Optional[str]  # 'long' or 'short' or None
    size: float  # Position size (lots or units)
    confidence: float
    
    # Gate results
    tcn_direction: Optional[int]
    tcn_probability: float
    ridge_confidence: float
    xgb_momentum: float
    xgb_acceleration: bool
    rf_drawdown_pips: float
    rf_streak_prob: float
    
    # Gate status
    confidence_gate_passed: bool
    momentum_gate_passed: bool
    risk_gate_passed: bool
    
    # Rejection reason if no trade
    reason: Optional[str]


class ModularEnsembleInference:
    """
    Inference pipeline for modular ensemble.
    
    Loads 4 independent models and combines their predictions using gated logic.
    No shared processing - each model sees its own feature subset.
    """
    
    def __init__(
        self,
        model_dir: str = "trained_data/models",
        config: Optional[InferenceConfig] = None,
    ):
        self.model_dir = Path(model_dir)
        self.config = config or InferenceConfig()
        
        self.tcn = None
        self.xgb = None
        self.rf = None
        self.ridge = None
        
        self._loaded = False
    
    def load_models(self) -> None:
        """Load all 4 models from disk."""
        from modular_trainers import TCNTrainer, TransformerDirectionTrainer, XGBoostTrainer, RandomForestTrainer, RidgeTrainer
        
        logger.info("Loading modular ensemble models...")
        
        # Direction model (try Transformer first, fall back to TCN)
        transformer_path = self.model_dir / "transformer_direction.keras"
        tcn_path = self.model_dir / "tcn_direction.keras"
        
        if transformer_path.exists():
            self.tcn = TransformerDirectionTrainer()
            self.tcn.load(str(transformer_path))
            logger.info("✓ Transformer direction model loaded")
        elif tcn_path.exists():
            self.tcn = TCNTrainer()
            self.tcn.load(str(tcn_path))
            logger.info("✓ TCN direction model loaded (legacy)")
        else:
            logger.warning(f"Direction model not found at {transformer_path} or {tcn_path}")
        
        # XGBoost
        xgb_path = self.model_dir / "xgb_momentum.pkl"
        if xgb_path.exists():
            self.xgb = XGBoostTrainer()
            self.xgb.load(str(xgb_path))
            logger.info("✓ XGBoost loaded")
        else:
            logger.warning(f"XGBoost model not found at {xgb_path}")
        
        # Random Forest
        rf_path = self.model_dir / "rf_risk.pkl"
        if rf_path.exists():
            self.rf = RandomForestTrainer()
            self.rf.load(str(rf_path))
            logger.info("✓ Random Forest loaded")
        else:
            logger.warning(f"Random Forest model not found at {rf_path}")
        
        # Ridge
        ridge_path = self.model_dir / "ridge_confidence.pkl"
        if ridge_path.exists():
            self.ridge = RidgeTrainer()
            self.ridge.load(str(ridge_path))
            logger.info("✓ Ridge loaded")
        else:
            logger.warning(f"Ridge model not found at {ridge_path}")
        
        self._loaded = True
        logger.info("Modular ensemble loaded.")
    
    def _find_features_by_pattern(self, df: pd.DataFrame, patterns: List[str]) -> List[str]:
        """Find features by partial name matching."""
        found = []
        for col in df.columns:
            col_lower = col.lower()
            for pattern in patterns:
                if pattern.lower() in col_lower:
                    if col not in found:
                        found.append(col)
                    break
        return found
    
    def _extract_features_by_names(
        self,
        df: pd.DataFrame,
        feature_names: Optional[List[str]],
        fallback_preferred: List[str],
        fallback_patterns: List[str],
        exclude: List[str] = None,
    ) -> np.ndarray:
        """
        Extract features using saved feature names from training.
        Falls back to pattern matching if feature_names is not available.
        """
        exclude = exclude or ['open', 'high', 'low', 'close', 'volume', 'time', 'timestamp']
        
        # If we have saved feature names from training, use them
        if feature_names:
            available = [f for f in feature_names if f in df.columns]
            if len(available) == len(feature_names):
                return df[available].values.astype(np.float32)
            # Some features missing, fall through to fallback
        
        # Fallback: Try exact matches first
        available = [f for f in fallback_preferred if f in df.columns]
        
        # If not enough, try pattern matching
        if len(available) < 5:
            pattern_features = self._find_features_by_pattern(df, fallback_patterns)
            for f in pattern_features:
                if f not in available and f not in exclude:
                    available.append(f)
        
        # If still not enough, use any numeric columns
        if len(available) < 5:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                if col not in available and col not in exclude:
                    available.append(col)
        
        if not available:
            # Last resort: use all numeric columns
            available = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return df[available].values.astype(np.float32)
    
    def _add_directional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add computed directional features to dataframe for inference.
        Must match the features computed during training.
        """
        df = df.copy()
        
        # SMA crossover: 1 if sma_5 > sma_20, else 0
        if 'sma_5' in df.columns and 'sma_20' in df.columns:
            df['sma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(np.float32)
        
        # MACD crossover: 1 if macd > signal, else 0
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(np.float32)
        
        # Higher high count: How many of last 10 bars made higher highs
        if 'high' in df.columns:
            highs = df['high'].values
            hh_count = np.zeros(len(highs), dtype=np.float32)
            for i in range(10, len(highs)):
                count = 0
                for j in range(1, 10):
                    if highs[i-j] > highs[i-j-1]:
                        count += 1
                hh_count[i] = count / 9.0  # Normalize to 0-1
            df['higher_high_count'] = hh_count
        
        # Lower low count: How many of last 10 bars made lower lows
        if 'low' in df.columns:
            lows = df['low'].values
            ll_count = np.zeros(len(lows), dtype=np.float32)
            for i in range(10, len(lows)):
                count = 0
                for j in range(1, 10):
                    if lows[i-j] < lows[i-j-1]:
                        count += 1
                ll_count[i] = count / 9.0  # Normalize to 0-1
            df['lower_low_count'] = ll_count
        
        # Volume direction: Are up bars getting more volume?
        if 'volume' in df.columns and 'close' in df.columns:
            close = df['close'].values
            volume = df['volume'].values
            vol_dir = np.zeros(len(close), dtype=np.float32)
            for i in range(10, len(close)):
                up_vol = 0.0
                down_vol = 0.0
                for j in range(10):
                    if close[i-j] > close[i-j-1]:
                        up_vol += volume[i-j]
                    else:
                        down_vol += volume[i-j]
                total = up_vol + down_vol
                vol_dir[i] = up_vol / max(total, 1e-8) if total > 0 else 0.5
            df['volume_direction'] = vol_dir
        
        # Trend direction: Combined signal from multiple indicators
        trend_components = []
        if 'sma_cross_5_20' in df.columns:
            trend_components.append(df['sma_cross_5_20'].values)
        if 'macd_cross' in df.columns:
            trend_components.append(df['macd_cross'].values)
        if 'rsi' in df.columns:
            trend_components.append((df['rsi'].values > 50).astype(np.float32))
        
        if trend_components:
            df['trend_direction'] = np.mean(trend_components, axis=0)
        
        return df
    
    def _extract_tcn_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for direction model (TCN or Transformer) using saved feature names."""
        # NORMALIZED features for instrument-agnostic inference
        normalized_features = get_normalized_feature_names()['direction']
        
        # Legacy fallback features
        legacy_fallback = [
            'adx', 'macd', 'macd_signal', 'macd_hist',
            'rsi', 'stoch_k', 'stoch_d',
            'returns', 'momentum_10', 'roc_5', 'roc_10',
            'volatility_20', 'sma_cross_5_20', 'macd_cross',
        ]
        
        # Combine: prefer normalized, then legacy
        fallback_preferred = normalized_features + legacy_fallback
        fallback_patterns = ['return', 'zscore', 'ratio', 'norm', 'pct_rank', 'cross', 'rsi', 'macd']
        
        # Use saved feature names from training if available
        feature_names = getattr(self.tcn, 'feature_names', None) if self.tcn else None
        return self._extract_features_by_names(df, feature_names, fallback_preferred, fallback_patterns)
    
    def _extract_xgb_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for XGBoost model using saved feature names."""
        # NORMALIZED features for instrument-agnostic inference
        normalized_features = get_normalized_feature_names()['momentum']
        
        # Legacy fallback features
        legacy_fallback = [
            'returns', 'roc_5', 'roc_10', 'roc_20',
            'high_low_ratio', 'momentum_10', 'macd', 'macd_hist',
            'stoch_k', 'stoch_d', 'mfi',
        ]
        
        # Combine: prefer normalized, then legacy
        fallback_preferred = normalized_features + legacy_fallback
        fallback_patterns = ['return', 'atr_pct', 'volatility', 'volume_ratio', 'macd_norm', 'rsi_norm']
        
        # Use saved feature names from training if available
        feature_names = getattr(self.xgb, 'feature_names', None) if self.xgb else None
        return self._extract_features_by_names(df, feature_names, fallback_preferred, fallback_patterns)
    
    def _extract_rf_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for Random Forest model using saved feature names."""
        # NORMALIZED features for instrument-agnostic inference
        normalized_features = get_normalized_feature_names()['risk']
        
        # Legacy fallback features
        legacy_fallback = [
            'atr', 'volatility_5', 'volatility_10', 'volatility_20',
            'high_low_ratio', 'bb_width_20',
            'returns', 'momentum_10',
        ]
        
        # Combine: prefer normalized, then legacy
        fallback_preferred = normalized_features + legacy_fallback
        fallback_patterns = ['atr_pct', 'volatility', 'tr_pct', 'hl_range', 'zscore', 'return']
        
        # Use saved feature names from training if available
        feature_names = getattr(self.rf, 'feature_names', None) if self.rf else None
        return self._extract_features_by_names(df, feature_names, fallback_preferred, fallback_patterns)
    
    def _extract_ridge_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for Ridge model using saved feature names."""
        # NORMALIZED features for instrument-agnostic inference
        normalized_features = get_normalized_feature_names()['confidence']
        
        # Legacy fallback features
        legacy_fallback = [
            'volatility_5', 'volatility_10', 'volatility_20',
            'atr', 'bb_width_20', 'bb_position_20',
            'adx', 'returns',
        ]
        
        # Combine: prefer normalized, then legacy
        fallback_preferred = normalized_features + legacy_fallback
        fallback_patterns = ['atr_pct', 'volatility', 'volume_ratio', 'sma_ratio', 'return', 'zscore']
        
        # Use saved feature names from training if available
        feature_names = getattr(self.ridge, 'feature_names', None) if self.ridge else None
        return self._extract_features_by_names(df, feature_names, fallback_preferred, fallback_patterns)
    
    def _calculate_position_size(
        self,
        expected_drawdown_pips: float,
        equity: Optional[float] = None,
    ) -> float:
        """
        Calculate position size for target risk percentage.
        
        Formula: size = (equity * risk_pct) / (drawdown_pips * pip_value)
        """
        equity = equity or self.config.account_equity
        risk_amount = equity * self.config.risk_per_trade_pct
        
        if expected_drawdown_pips <= 0:
            expected_drawdown_pips = 10.0  # Default minimum
        
        size = risk_amount / (expected_drawdown_pips * self.config.pip_value)
        
        # Clamp to reasonable range
        size = max(0.01, min(10.0, size))
        
        return round(size, 2)
    
    def predict(
        self,
        df: pd.DataFrame,
        equity: Optional[float] = None,
    ) -> TradeSignal:
        """
        Run inference through all 4 models and apply gates.
        
        IMPORTANT: Computes normalized features first for instrument-agnostic inference.
        Models trained on GBP_USD will work on USD_JPY, EUR_USD, etc.
        
        Args:
            df: DataFrame with features (must have all required columns)
            equity: Account equity for position sizing
        
        Returns:
            TradeSignal with trade decision and all model outputs
        """
        if not self._loaded:
            self.load_models()
        
        # FIRST: Compute normalized features for instrument-agnostic inference
        # This is the key to making models work across different currency pairs
        if 'returns_1' not in df.columns:
            df = compute_normalized_features(df)
        
        # Initialize with defaults
        tcn_direction = None
        tcn_probability = 0.5
        ridge_confidence = 0.0
        xgb_momentum = 0.0
        xgb_acceleration = False
        rf_drawdown_pips = 100.0
        rf_streak_prob = 1.0
        
        # Get predictions from each model
        try:
            if self.tcn is not None:
                tcn_features = self._extract_tcn_features(df)
                tcn_pred = self.tcn.predict(tcn_features)
                tcn_direction = tcn_pred['direction']
                tcn_probability = tcn_pred['probability']
        except Exception as e:
            logger.warning(f"TCN prediction failed: {e}")
        
        try:
            if self.ridge is not None:
                ridge_features = self._extract_ridge_features(df)
                ridge_pred = self.ridge.predict(ridge_features)
                ridge_confidence = ridge_pred['confidence']
        except Exception as e:
            logger.warning(f"Ridge prediction failed: {e}")
        
        try:
            if self.xgb is not None:
                xgb_features = self._extract_xgb_features(df)
                # DEBUG: Log feature stats on first prediction
                if not hasattr(self, '_xgb_debug_logged'):
                    logger.info(f"XGBoost features shape: {xgb_features.shape}")
                    if len(xgb_features) > 0:
                        logger.info(f"XGBoost feature stats - min:{xgb_features.min():.2f} max:{xgb_features.max():.2f} mean:{xgb_features.mean():.2f}")
                    self._xgb_debug_logged = True
                xgb_pred = self.xgb.predict(xgb_features)
                xgb_momentum = xgb_pred['momentum']
                xgb_acceleration = xgb_pred['acceleration']
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
        
        try:
            if self.rf is not None:
                rf_features = self._extract_rf_features(df)
                rf_pred = self.rf.predict(rf_features)
                # Use percentage-based drawdown (instrument-agnostic)
                rf_drawdown_pct = rf_pred.get('expected_drawdown_pct', rf_pred.get('expected_drawdown_pips', 0) / 10000)
                rf_drawdown_pips = rf_pred.get('expected_drawdown_pips', rf_drawdown_pct * 10000)  # For display
                rf_streak_prob = rf_pred['streak_prob']
        except Exception as e:
            logger.warning(f"Random Forest prediction failed: {e}")
            rf_drawdown_pct = 1.0  # Default to high risk on error
        
        # Apply gates
        confidence_gate_passed = ridge_confidence >= self.config.min_confidence
        
        momentum_fresh = xgb_momentum >= self.config.min_momentum
        momentum_gate_passed = momentum_fresh or xgb_acceleration
        
        # Use percentage-based risk gate (instrument-agnostic)
        risk_gate_passed = (
            rf_drawdown_pct <= self.config.max_drawdown_pct and
            rf_streak_prob <= self.config.max_streak_prob
        )
        
        # Determine if we trade
        all_gates_passed = (
            tcn_direction is not None and
            confidence_gate_passed and
            momentum_gate_passed and
            risk_gate_passed
        )
        
        # Build rejection reason
        reason = None
        if not all_gates_passed:
            reasons = []
            if tcn_direction is None:
                reasons.append("no_direction")
            if not confidence_gate_passed:
                reasons.append(f"low_confidence({ridge_confidence:.0f}<{self.config.min_confidence})")
            if not momentum_gate_passed:
                reasons.append(f"dead_momentum({xgb_momentum:.2f})")
            if not risk_gate_passed:
                if rf_drawdown_pct > self.config.max_drawdown_pct:
                    reasons.append(f"high_drawdown({rf_drawdown_pct:.2%}>{self.config.max_drawdown_pct:.2%})")
                if rf_streak_prob > self.config.max_streak_prob:
                    reasons.append(f"streak_risk({rf_streak_prob:.2f}>{self.config.max_streak_prob})")
            reason = ", ".join(reasons)
        
        # Calculate position size
        size = 0.0
        if all_gates_passed:
            size = self._calculate_position_size(rf_drawdown_pips, equity)
        
        # Determine direction string
        direction_str = None
        if all_gates_passed and tcn_direction is not None:
            direction_str = 'long' if tcn_direction == 1 else 'short'
        
        return TradeSignal(
            trade=all_gates_passed,
            direction=direction_str,
            size=size,
            confidence=ridge_confidence,
            tcn_direction=tcn_direction,
            tcn_probability=tcn_probability,
            ridge_confidence=ridge_confidence,
            xgb_momentum=xgb_momentum,
            xgb_acceleration=xgb_acceleration,
            rf_drawdown_pips=rf_drawdown_pips,
            rf_streak_prob=rf_streak_prob,
            confidence_gate_passed=confidence_gate_passed,
            momentum_gate_passed=momentum_gate_passed,
            risk_gate_passed=risk_gate_passed,
            reason=reason,
        )
    
    def predict_verbose(
        self,
        df: pd.DataFrame,
        equity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run inference with verbose output for logging/display.
        
        Returns dict with all details formatted for display.
        """
        signal = self.predict(df, equity)
        
        # Format gate checks
        gate_checks = []
        
        # TCN direction
        if signal.tcn_direction is not None:
            dir_str = "LONG" if signal.tcn_direction == 1 else "SHORT"
            gate_checks.append(f"TCN: {dir_str} (prob={signal.tcn_probability:.2f})")
        else:
            gate_checks.append("TCN: NO SIGNAL")
        
        # Ridge confidence
        status = "✓" if signal.confidence_gate_passed else "✗"
        gate_checks.append(f"Ridge: {signal.ridge_confidence:.0f}/100 {status}")
        
        # XGBoost momentum
        status = "✓" if signal.momentum_gate_passed else "✗"
        accel_str = "accel=true" if signal.xgb_acceleration else "accel=false"
        gate_checks.append(f"XGBoost: momentum={signal.xgb_momentum:.2f}, {accel_str} {status}")
        
        # RF risk
        status = "✓" if signal.risk_gate_passed else "✗"
        gate_checks.append(f"RF: drawdown={signal.rf_drawdown_pips:.1f}pips, streak={signal.rf_streak_prob:.2f} {status}")
        
        # Final decision
        if signal.trade:
            decision = f"→ TRADE: {signal.direction.upper()}, size={signal.size} lots"
        else:
            decision = f"→ NO TRADE: {signal.reason}"
        
        return {
            'trade': signal.trade,
            'direction': signal.direction,
            'size': signal.size,
            'gate_checks': gate_checks,
            'decision': decision,
            'raw_signal': signal,
        }


def run_inference_test():
    """Quick test of inference pipeline."""
    import pandas as pd
    import numpy as np
    
    # Create dummy data
    n = 100
    df = pd.DataFrame({
        'close': np.cumsum(np.random.randn(n) * 0.01) + 150,
        'high': np.cumsum(np.random.randn(n) * 0.01) + 150.1,
        'low': np.cumsum(np.random.randn(n) * 0.01) + 149.9,
        'volume': np.random.randint(1000, 10000, n),
        'returns': np.random.randn(n) * 0.01,
        'volatility_5': np.abs(np.random.randn(n)) * 0.01,
        'volatility_10': np.abs(np.random.randn(n)) * 0.01,
        'volatility_20': np.abs(np.random.randn(n)) * 0.01,
        'atr': np.abs(np.random.randn(n)) * 0.5,
        'rsi': np.random.uniform(30, 70, n),
        'momentum_10': np.random.randn(n) * 0.01,
        'macd': np.random.randn(n) * 0.001,
        'macd_hist': np.random.randn(n) * 0.001,
        'obv': np.cumsum(np.random.randint(-1000, 1000, n)),
        'mfi': np.random.uniform(20, 80, n),
        'adx': np.random.uniform(15, 40, n),
    })
    
    # Test inference
    ensemble = ModularEnsembleInference()
    
    # Check if models exist (check for both Transformer and TCN)
    model_dir = Path("trained_data/models")
    if not (model_dir / "transformer_direction.keras").exists() and not (model_dir / "tcn_direction.keras").exists():
        print("Models not found. Train first with: buddy train --model-type ensemble")
        return
    
    result = ensemble.predict_verbose(df)
    
    print("\n" + "="*60)
    print("MODULAR ENSEMBLE INFERENCE TEST")
    print("="*60)
    for check in result['gate_checks']:
        print(check)
    print("-"*60)
    print(result['decision'])
    print("="*60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_inference_test()

