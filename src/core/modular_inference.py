"""
Modular Ensemble Inference Pipeline.

Supports TWO modes:

1. DIRECTION MODE (legacy):
   - Transformer/TCN gives direction (long/short)
   - Trade only if all gates pass

2. REGIME MODE (new):
   - Transformer classifies market regime (trend/chop/mean_revert)
   - TREND: Let XGBoost/Ridge/RF decide direction via momentum
   - CHOP: Skip trading entirely
   - MEAN_REVERT: Fade 2-bar momentum
   
Gates (both modes):
- Ridge confidence > 75
- XGBoost momentum fresh OR accelerating
- RF expected drawdown < threshold

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
from .modular_data_loaders import compute_normalized_features, get_normalized_feature_names

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference gates.
    
    Thresholds calibrated for gate models trained on:
    - Confidence: ADX-based trend strength (0-100)
    - Momentum: Percentile-normalized (median=0.3, P90=0.7)
    - Risk: ATR-based expected drawdown (typically 0.5-3%)
    """
    # Confidence gate - ADX-based, 50+ is strong trend
    min_confidence: float = 45.0  # 0-100 scale
    
    # Momentum gate - median momentum is 0.3, so 0.15 catches bottom 30%
    min_momentum: float = 0.15  # 0-1 scale
    require_fresh_or_accel: bool = True
    
    # Risk gate - ATR-based drawdown (2x ATR, typically 0.5-3%)
    max_drawdown_pct: float = 0.025  # 2.5% max expected drawdown
    max_streak_prob: float = 0.6  # 60% max streak continuation
    
    # Legacy pip-based (kept for backward compatibility)
    max_drawdown_pips: float = 250.0  # ~2.5% for majors
    
    # Permissive mode: Ignore failing gates from sklearn models when version mismatch detected
    # When True, only Transformer direction is used for decision
    permissive_mode: bool = False
    
    # Position sizing - RISK-BASED (not arbitrary lots!)
    risk_per_trade_pct: float = 0.01  # 1% risk per trade (conservative)
    account_equity: float = 10000.0  # Default equity for sizing
    pip_value: float = 10.0  # ~$10 per pip per standard lot
    
    # LIQUIDITY LIMITS - Maximum lots by pair to avoid market impact
    # Large orders cause slippage that destroys edge
    max_lots_by_pair: dict = None  # Set in __post_init__
    
    def __post_init__(self):
        if self.max_lots_by_pair is None:
            self.max_lots_by_pair = {
                'EUR_USD': 2.0,   # Most liquid - 2 lots max
                'USD_JPY': 1.5,   # Very liquid
                'GBP_USD': 1.0,   # Liquid
                'USD_CHF': 1.0,
                'AUD_USD': 1.0,
                'USD_CAD': 1.0,
                'NZD_USD': 0.5,   # Less liquid
                'GBP_JPY': 0.5,   # Cross - illiquid
                'EUR_GBP': 0.5,
                'EUR_JPY': 0.5,
                'DEFAULT': 0.5,   # Unknown pairs
            }


@dataclass
class TradeSignal:
    """Result of inference pipeline."""
    trade: bool
    direction: Optional[str]  # 'long' or 'short' or None
    size: float  # Position size (lots or units)
    confidence: float
    
    # Regime results (new)
    regime: Optional[str] = None  # 'trend', 'chop', 'mean_revert', or None
    regime_confidence: float = 0.0
    
    # Gate results
    tcn_direction: Optional[int] = None
    tcn_probability: float = 0.5
    ridge_confidence: float = 0.0
    xgb_momentum: float = 0.0
    xgb_acceleration: bool = False
    rf_drawdown_pips: float = 0.0
    rf_streak_prob: float = 0.0
    
    # Hybrid voting (HistGB + Transformer)
    histgb_direction: Optional[int] = None
    histgb_probability: float = 0.5
    models_agree: bool = True  # True if Transformer and HistGB agree
    
    # Gate status
    confidence_gate_passed: bool = False
    momentum_gate_passed: bool = False
    risk_gate_passed: bool = False
    regime_gate_passed: bool = True  # True if not in CHOP regime
    
    # Rejection reason if no trade
    reason: Optional[str] = None


class ModularEnsembleInference:
    """
    Inference pipeline for modular ensemble.
    
    Loads 4 independent models and combines their predictions using gated logic.
    No shared processing - each model sees its own feature subset.
    
    Supports three modes:
    - DIRECTION MODE: Transformer/TCN predicts direction directly
    - REGIME MODE: Transformer classifies regime, direction derived from momentum
    - HYBRID MODE: Transformer + HistGB voting for higher confidence trades
    
    Hybrid voting logic:
    - If both models agree: trade with full confidence
    - If models disagree in low-vol regime: use HistGB (more stable)
    - If models disagree in high-vol regime: use Transformer (better at trends)
    """
    
    def __init__(
        self,
        model_dir: str = "trained_data/models",
        config: Optional[InferenceConfig] = None,
    ):
        self.model_dir = Path(model_dir)
        self.config = config or InferenceConfig()
        
        self.tcn = None  # Direction model (legacy)
        self.histgb = None  # HistGB baseline for hybrid voting
        self.regime_model = None  # Regime classifier (new)
        self.xgb = None
        self.rf = None
        self.ridge = None
        
        self.use_regime = False  # Will be set during load_models
        self.use_hybrid = False  # Enable HistGB voting
        self._loaded = False
    
    def load_models(self) -> None:
        """Load all 4 models from disk."""
        from modular_trainers import (
            TCNTrainer, TransformerDirectionTrainer, TransformerRegimeTrainer,
            XGBoostTrainer, RandomForestTrainer, RidgeTrainer,
            HistGradientBoostingDirectionTrainer
        )
        
        logger.info("Loading modular ensemble models...")
        
        # Check for regime model first (new mode)
        regime_path = self.model_dir / "transformer_regime.keras"
        transformer_path = self.model_dir / "transformer_direction.keras"
        tcn_path = self.model_dir / "tcn_direction.keras"
        histgb_path = self.model_dir / "histgb_direction.pkl"
        
        if regime_path.exists():
            # REGIME MODE
            self.regime_model = TransformerRegimeTrainer()
            self.regime_model.load(str(regime_path))
            self.use_regime = True
            logger.info("✓ Transformer REGIME model loaded (3-class: trend/chop/mean_revert)")
        elif transformer_path.exists():
            # DIRECTION MODE (Transformer)
            self.tcn = TransformerDirectionTrainer()
            self.tcn.load(str(transformer_path))
            self.use_regime = False
            logger.info("✓ Transformer direction model loaded")
        elif tcn_path.exists():
            # DIRECTION MODE (TCN legacy)
            self.tcn = TCNTrainer()
            self.tcn.load(str(tcn_path))
            self.use_regime = False
            logger.info("✓ TCN direction model loaded (legacy)")
        else:
            logger.warning(f"No direction/regime model found at {regime_path}, {transformer_path}, or {tcn_path}")
        
        # Load HistGB for hybrid voting (if available)
        if histgb_path.exists():
            self.histgb = HistGradientBoostingDirectionTrainer()
            self.histgb.load(str(histgb_path))
            self.use_hybrid = True
            logger.info("✓ HistGB baseline loaded (hybrid voting enabled)")
        else:
            self.use_hybrid = False
            logger.info("ℹ HistGB not found - single-model mode (train with --train-histgb to enable hybrid)")
        
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
        
        # Auto-detect sklearn version mismatch and enable permissive mode
        self._check_sklearn_version_mismatch()
        
        self._loaded = True
        logger.info("Modular ensemble loaded.")
    
    def _check_sklearn_version_mismatch(self) -> None:
        """Check if sklearn models were trained with different version and enable permissive mode."""
        import sklearn
        current_version = sklearn.__version__
        
        # Check for explicit version mismatch OR poor model quality in model metadata
        meta_path = self.model_dir / "modular_ensemble.meta.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            
            # Check sklearn version mismatch
            trained_sklearn = meta.get('sklearn_version')
            if trained_sklearn and trained_sklearn != current_version:
                major_trained = trained_sklearn.split('.')[0:2]
                major_current = current_version.split('.')[0:2]
                
                if major_trained != major_current:
                    logger.warning(
                        f"⚠️ sklearn version mismatch: trained={trained_sklearn}, current={current_version}. "
                        f"Enabling permissive mode (sklearn gates bypassed)."
                    )
                    self.config.permissive_mode = True
                    return
            
            # Check for poor RF model quality (drawdown MAE > 50% means useless)
            results = meta.get('results', {})
            rf_results = results.get('random_forest', {})
            drawdown_mae_bps = rf_results.get('drawdown_mae_bps', 0)
            
            if drawdown_mae_bps > 5000:  # > 50% MAE = unreliable
                logger.warning(
                    f"⚠️ RF model has high error (MAE={drawdown_mae_bps/100:.1f}%). "
                    f"Enabling permissive mode (risk gate bypassed)."
                )
                self.config.permissive_mode = True

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
        
        CRITICAL: If feature_names is provided, we MUST return features in that
        exact order with the exact count. Missing features are filled with 0.0
        to maintain compatibility with the trained model's scaler and weights.
        
        Falls back to pattern matching only if feature_names is not available.
        """
        exclude = exclude or ['open', 'high', 'low', 'close', 'volume', 'time', 'timestamp']
        
        # If we have saved feature names from training, use them with proper ordering
        if feature_names:
            # Build array with exact feature count and order
            n_features = len(feature_names)
            n_rows = len(df)
            result = np.zeros((n_rows, n_features), dtype=np.float32)
            
            missing_features = []
            for i, fname in enumerate(feature_names):
                if fname in df.columns:
                    result[:, i] = df[fname].values.astype(np.float32)
                else:
                    missing_features.append(fname)
            
            if missing_features and len(missing_features) < len(feature_names) // 2:
                # Log missing features but continue (fill with 0)
                logger.debug(f"Missing {len(missing_features)} features (filled with 0): {missing_features[:5]}...")
            elif missing_features:
                logger.warning(f"Many features missing ({len(missing_features)}/{n_features}). "
                              f"First 10: {missing_features[:10]}")
            
            return result
        
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
    
    def _extract_regime_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for regime classification model."""
        # Features that describe market state (regime indicators)
        regime_features = [
            # Trend strength
            'adx', 'trend_strength',
            # Volatility state
            'atr_pct_14', 'atr_pct_20', 'volatility_10', 'volatility_20',
            # Mean reversion signals
            'zscore_20', 'zscore_50', 'rsi', 'rsi_norm',
            'bb_position_20', 'pct_rank_20', 'pct_rank_50',
            # Momentum consistency
            'returns_1', 'returns_5', 'returns_10', 'returns_20',
            # Crossover state
            'sma_cross_5_20', 'macd_cross',
            # Volume context
            'volume_ratio_10', 'volume_zscore',
        ]
        
        fallback_patterns = ['adx', 'rsi', 'zscore', 'returns', 'volatility', 'atr_pct', 'pct_rank']
        
        # Use saved feature names from training if available
        feature_names = getattr(self.regime_model, 'feature_names', None) if self.regime_model else None
        return self._extract_features_by_names(df, feature_names, regime_features, fallback_patterns)
    
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
    
    def _compute_confidence_direct(self, df: pd.DataFrame) -> float:
        """
        Compute confidence score directly from indicators (0-100 scale).
        
        This bypasses the ElasticNet model and uses the formula directly,
        which is faster and avoids learning a synthetic target.
        
        Formula weights:
        - ADX (40%): Trend strength - higher = more confident
        - Volatility (20%): Moderate vol = high confidence
        - RSI (15%): Not extreme = high confidence
        - BB Position (10%): Position within Bollinger Bands
        - Volume (15%): Above average volume = confirmation
        """
        import numpy as np
        
        # Get the last row of data
        row = df.iloc[-1]
        
        # Extract indicators with safe defaults
        adx = row.get('adx', 25.0) if 'adx' in df.columns else 25.0
        rsi = row.get('rsi', 50.0) if 'rsi' in df.columns else 50.0
        atr_pct = row.get('atr_pct_14', 0.01) if 'atr_pct_14' in df.columns else 0.01
        bb_pos = row.get('bb_position_20', 0.5) if 'bb_position_20' in df.columns else 0.5
        volume_ratio = row.get('volume_ratio_20', 1.0) if 'volume_ratio_20' in df.columns else 1.0
        
        # Handle NaN values
        adx = float(adx) if not np.isnan(adx) else 25.0
        rsi = float(rsi) if not np.isnan(rsi) else 50.0
        atr_pct = float(atr_pct) if not np.isnan(atr_pct) else 0.01
        bb_pos = float(bb_pos) if not np.isnan(bb_pos) else 0.5
        volume_ratio = float(volume_ratio) if not np.isnan(volume_ratio) else 1.0
        
        # ADX score (40%) - higher ADX = more confident trend
        # Use percentile-based scaling: 15-35 is typical range
        adx_normalized = (adx - 15) / 20  # 15->0, 35->1
        adx_score = np.clip(adx_normalized * 0.5 + 0.25, 0.0, 1.0)
        
        # Volatility score (20%) - moderate vol is ideal
        if atr_pct < 0.005:  # Too low - no movement
            vol_score = 0.5
        elif atr_pct > 0.02:  # Too high - chaotic
            vol_score = max(0.0, 1.0 - (atr_pct - 0.02) / 0.02)
        else:  # Sweet spot
            vol_score = 0.8 + 0.2 * (1.0 - abs(atr_pct - 0.01) / 0.01)
        vol_score = np.clip(vol_score, 0.0, 1.0)
        
        # RSI score (15%) - not extreme = high confidence
        rsi_distance = abs(rsi - 50)
        rsi_score = max(0.0, 1.0 - rsi_distance / 30.0)
        
        # BB position score (10%) - extremes can be good for reversals
        if bb_pos < 0.2 or bb_pos > 0.8:  # Near bands
            bb_score = 0.7
        else:  # Middle zone
            bb_score = 0.5 + 0.5 * (1.0 - abs(bb_pos - 0.5) * 2)
        bb_score = np.clip(bb_score, 0.0, 1.0)
        
        # Volume confirmation (15%) - above average = confirmation
        vol_conf_score = np.clip((volume_ratio - 0.5) / 1.0, 0.0, 1.0)
        
        # Combine with weights
        raw_conf = (
            adx_score * 0.40 +
            vol_score * 0.20 +
            rsi_score * 0.15 +
            bb_score * 0.10 +
            vol_conf_score * 0.15
        )
        
        # Map to 15-95 range (never fully 0 or 100)
        confidence = 15 + raw_conf * 80
        
        return float(confidence)
    
    def _calculate_position_size(
        self,
        expected_drawdown_pips: float,
        equity: Optional[float] = None,
        instrument: Optional[str] = None,
    ) -> float:
        """
        Calculate position size for target risk percentage with liquidity limits.
        
        Formula: size = (equity * risk_pct) / (stop_loss_pips * pip_value)
        
        CRITICAL: Large positions cause slippage that destroys edge!
        - 10 lots on EUR_USD = 6+ pips slippage
        - 1 lot on EUR_USD = <1 pip slippage
        
        Args:
            expected_drawdown_pips: Stop loss distance in pips
            equity: Account equity (default: config value)
            instrument: Trading pair for liquidity limit lookup
        
        Returns:
            Position size in lots, capped by liquidity limits
        """
        equity = equity or self.config.account_equity
        risk_amount = equity * self.config.risk_per_trade_pct
        
        # Minimum stop loss to prevent oversizing
        if expected_drawdown_pips <= 5.0:
            expected_drawdown_pips = 10.0  # Minimum 10 pip stop
        
        # Risk-based position size: lots = risk_$ / (pips * pip_value)
        # pip_value ~= $10 per pip per standard lot for most pairs
        size_lots = risk_amount / (expected_drawdown_pips * self.config.pip_value)
        
        # Apply liquidity limit for instrument
        max_lots = self.config.max_lots_by_pair.get(
            instrument, 
            self.config.max_lots_by_pair.get('DEFAULT', 0.5)
        ) if instrument else 1.0
        
        # Hard cap at liquidity limit
        size_lots = min(size_lots, max_lots)
        
        # Minimum position size
        size_lots = max(0.01, size_lots)
        
        return round(size_lots, 2)
    
    def predict(
        self,
        df: pd.DataFrame,
        equity: Optional[float] = None,
        instrument: Optional[str] = None,
    ) -> TradeSignal:
        """
        Run inference through all models and apply gates.
        
        REGIME MODE:
        - Transformer classifies regime (trend/chop/mean_revert)
        - TREND: Direction from XGBoost momentum sign
        - CHOP: Skip trading entirely
        - MEAN_REVERT: Fade 2-bar momentum
        
        DIRECTION MODE (legacy):
        - Transformer/TCN predicts direction directly
        
        IMPORTANT: Computes normalized features first for instrument-agnostic inference.
        
        Args:
            df: DataFrame with features (must have all required columns)
            equity: Account equity for position sizing
            instrument: Trading pair (e.g., 'EUR_USD') for liquidity limits
        
        Returns:
            TradeSignal with trade decision and all model outputs
        """
        if not self._loaded:
            self.load_models()
        
        # Store instrument for position sizing
        self._current_instrument = instrument
        
        # FIRST: Compute normalized features for instrument-agnostic inference
        if 'returns_1' not in df.columns:
            df = compute_normalized_features(df)
        
        # Initialize with defaults
        regime = None
        regime_confidence = 0.0
        tcn_direction = None
        tcn_probability = 0.5
        ridge_confidence = 0.0
        xgb_momentum = 0.0
        xgb_acceleration = False
        rf_drawdown_pips = 100.0
        rf_drawdown_pct = 1.0
        rf_streak_prob = 1.0
        
        # === GET REGIME OR DIRECTION ===
        if self.use_regime and self.regime_model is not None:
            # REGIME MODE
            try:
                regime_features = self._extract_regime_features(df)
                regime_pred = self.regime_model.predict(regime_features)
                regime = regime_pred['regime_name']  # 'trend', 'chop', 'mean_revert'
                regime_confidence = regime_pred['confidence']
                logger.debug(f"Regime: {regime} (confidence={regime_confidence:.2f})")
            except Exception as e:
                logger.warning(f"Regime prediction failed: {e}")
                regime = 'chop'  # Default to skip on error
        else:
            # DIRECTION MODE (legacy)
            try:
                if self.tcn is not None:
                    tcn_features = self._extract_tcn_features(df)
                    tcn_pred = self.tcn.predict(tcn_features)
                    tcn_direction = tcn_pred['direction']
                    tcn_probability = tcn_pred['probability']
            except Exception as e:
                logger.warning(f"TCN prediction failed: {e}")
        
        # === HYBRID VOTING (HistGB + Transformer) ===
        histgb_direction = None
        histgb_probability = 0.5
        models_agree = True
        
        if self.use_hybrid and self.histgb is not None and not self.use_regime:
            try:
                histgb_features = self._extract_tcn_features(df)  # Same features as Transformer
                histgb_pred = self.histgb.predict(histgb_features)
                histgb_direction = histgb_pred['direction']
                histgb_probability = histgb_pred['probability']
                
                # Calculate confidence (distance from 0.5)
                tcn_confidence = abs(tcn_probability - 0.5) * 2  # 0-1 scale
                histgb_confidence = abs(histgb_probability - 0.5) * 2  # 0-1 scale
                
                # Check if models agree
                if tcn_direction is not None and histgb_direction is not None:
                    models_agree = (tcn_direction == histgb_direction)
                    
                    if models_agree:
                        # Both agree - boost confidence
                        tcn_probability = (tcn_probability + histgb_probability) / 2
                        logger.debug(f"Hybrid: AGREE (Transformer={tcn_direction}, HistGB={histgb_direction})")
                    else:
                        # Models disagree - use confidence-weighted decision
                        # Only trust HistGB if it's significantly more confident
                        atr_pct = df['atr_pct_14'].iloc[-1] if 'atr_pct_14' in df.columns else 0.01
                        
                        # NEW: Consider confidence, not just volatility
                        # HistGB wins only if: (1) higher confidence AND (2) low volatility
                        if atr_pct < 0.005 and histgb_confidence > tcn_confidence * 1.2:
                            # Low-vol AND HistGB is 20%+ more confident - trust HistGB
                            tcn_direction = histgb_direction
                            tcn_probability = histgb_probability
                            logger.debug(f"Hybrid: DISAGREE, low-vol + HistGB more confident -> HistGB ({histgb_direction})")
                        else:
                            # Trust Transformer (primary model), reduce confidence due to disagreement
                            tcn_probability = tcn_probability * 0.8
                            logger.debug(f"Hybrid: DISAGREE -> Transformer ({tcn_direction}), conf={tcn_confidence:.2f} vs HistGB={histgb_confidence:.2f}")
            except Exception as e:
                logger.warning(f"HistGB prediction failed: {e}")
        
        # === GET SUPPORTING MODEL PREDICTIONS ===
        try:
            # OPTION A: Use direct formula instead of Ridge model (faster, more reliable)
            # This computes confidence from indicators directly rather than learning it
            ridge_confidence = self._compute_confidence_direct(df)
            logger.debug(f"Direct confidence formula: {ridge_confidence:.1f}")
        except Exception as e:
            logger.warning(f"Direct confidence calculation failed: {e}, falling back to Ridge")
            try:
                if self.ridge is not None:
                    ridge_features = self._extract_ridge_features(df)
                    ridge_pred = self.ridge.predict(ridge_features)
                    ridge_confidence = ridge_pred['confidence']
            except Exception as e2:
                logger.warning(f"Ridge prediction also failed: {e2}")
                ridge_confidence = 50.0  # Neutral default
        
        try:
            if self.xgb is not None:
                xgb_features = self._extract_xgb_features(df)
                xgb_pred = self.xgb.predict(xgb_features)
                xgb_momentum = xgb_pred['momentum']
                xgb_acceleration = xgb_pred['acceleration']
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
        
        try:
            if self.rf is not None:
                rf_features = self._extract_rf_features(df)
                rf_pred = self.rf.predict(rf_features)
                rf_drawdown_pct = rf_pred.get('expected_drawdown_pct', rf_pred.get('expected_drawdown_pips', 0) / 10000)
                rf_drawdown_pips = rf_pred.get('expected_drawdown_pips', rf_drawdown_pct * 10000)
                rf_streak_prob = rf_pred['streak_prob']
        except Exception as e:
            logger.warning(f"Random Forest prediction failed: {e}")
        
        # === APPLY GATES ===
        # SMART GATING: Transformer probability is the primary signal
        # Gate models provide confirmation, but TCN confidence can override
        
        if self.config.permissive_mode:
            confidence_gate_passed = True
            momentum_gate_passed = True
            risk_gate_passed = True
            logger.debug("Permissive mode: sklearn gates bypassed")
        else:
            # TCN confidence = how far from 0.5 (uncertain)
            # 0.5 -> 0%, 0.6 -> 20%, 0.7 -> 40%, 0.8 -> 60%, 0.9 -> 80%, 1.0 -> 100%
            tcn_confidence = abs(tcn_probability - 0.5) * 200
            
            # Strong TCN signal (>55% or <45% probability) can override weak gate models
            tcn_strong = abs(tcn_probability - 0.5) > 0.05  # >55% or <45%
            tcn_very_strong = abs(tcn_probability - 0.5) > 0.15  # >65% or <35%
            
            # CONFIDENCE GATE: Use TCN-derived confidence OR Ridge, whichever is higher
            effective_confidence = max(ridge_confidence, tcn_confidence)
            confidence_gate_passed = effective_confidence >= self.config.min_confidence
            
            # MOMENTUM GATE: Pass if any of these are true:
            # 1. XGBoost momentum is fresh (above threshold)
            # 2. XGBoost detects acceleration
            # 3. TCN is strong (override weak momentum readings)
            momentum_fresh = xgb_momentum >= self.config.min_momentum
            momentum_gate_passed = momentum_fresh or xgb_acceleration or tcn_strong
            
            # RISK GATE: More lenient when TCN is confident
            # The Transformer sees patterns the simple gate models might miss
            if tcn_very_strong:
                # Very confident TCN - relax risk thresholds significantly
                risk_gate_passed = (
                    rf_drawdown_pct <= self.config.max_drawdown_pct * 2.0 and
                    rf_streak_prob <= self.config.max_streak_prob * 2.0
                )
            elif tcn_strong:
                # Strong TCN - relax risk thresholds moderately
                risk_gate_passed = (
                    rf_drawdown_pct <= self.config.max_drawdown_pct * 1.5 and
                    rf_streak_prob <= self.config.max_streak_prob * 1.5
                )
            else:
                # Weak TCN - use strict thresholds
                risk_gate_passed = (
                    rf_drawdown_pct <= self.config.max_drawdown_pct and
                    rf_streak_prob <= self.config.max_streak_prob
                )
        
        # === DETERMINE DIRECTION AND TRADE DECISION ===
        direction_str = None
        regime_gate_passed = True
        
        if self.use_regime:
            # REGIME-BASED DIRECTION LOGIC
            if regime == 'chop':
                # CHOP: Skip trading entirely
                regime_gate_passed = False
                direction_str = None
            elif regime == 'trend':
                # TREND: Direction from recent momentum sign
                # Use 2-bar return to determine trend direction
                if 'returns_2' in df.columns:
                    recent_return = df['returns_2'].iloc[-1]
                elif 'returns_1' in df.columns:
                    recent_return = df['returns_1'].iloc[-1]
                else:
                    recent_return = 0
                
                # Follow the trend
                if recent_return > 0:
                    direction_str = 'long'
                    tcn_direction = 1
                else:
                    direction_str = 'short'
                    tcn_direction = 0
                tcn_probability = regime_confidence
            elif regime == 'mean_revert':
                # MEAN REVERT: Fade 2-bar momentum
                if 'returns_2' in df.columns:
                    recent_return = df['returns_2'].iloc[-1]
                elif 'returns_1' in df.columns:
                    recent_return = df['returns_1'].iloc[-1]
                else:
                    recent_return = 0
                
                # Fade (opposite of recent move)
                if recent_return > 0:
                    direction_str = 'short'  # Fade the up move
                    tcn_direction = 0
                else:
                    direction_str = 'long'  # Fade the down move
                    tcn_direction = 1
                tcn_probability = regime_confidence
            
            # All gates for regime mode
            all_gates_passed = (
                regime_gate_passed and
                direction_str is not None and
                confidence_gate_passed and
                momentum_gate_passed and
                risk_gate_passed
            )
        else:
            # DIRECTION MODE (legacy)
            all_gates_passed = (
                tcn_direction is not None and
                confidence_gate_passed and
                momentum_gate_passed and
                risk_gate_passed
            )
            if all_gates_passed and tcn_direction is not None:
                direction_str = 'long' if tcn_direction == 1 else 'short'
        
        # === BUILD REJECTION REASON ===
        reason = None
        if not all_gates_passed:
            reasons = []
            if self.use_regime:
                if not regime_gate_passed:
                    reasons.append(f"regime=CHOP (skip)")
                if direction_str is None and regime != 'chop':
                    reasons.append("no_direction")
            else:
                if tcn_direction is None:
                    reasons.append("no_direction")
            if not confidence_gate_passed:
                reasons.append(f"low_confidence({ridge_confidence:.0f}<{self.config.min_confidence})")
            if not momentum_gate_passed:
                reasons.append(f"dead_momentum({xgb_momentum:.2f})")
            if not risk_gate_passed:
                if rf_drawdown_pct > self.config.max_drawdown_pct:
                    reasons.append(f"high_drawdown({rf_drawdown_pct:.2%})")
                if rf_streak_prob > self.config.max_streak_prob:
                    reasons.append(f"streak_risk({rf_streak_prob:.2f})")
            reason = ", ".join(reasons)
        
        # === CALCULATE POSITION SIZE ===
        size = 0.0
        if all_gates_passed:
            size = self._calculate_position_size(
                rf_drawdown_pips, 
                equity,
                instrument=getattr(self, '_current_instrument', None)
            )
        
        return TradeSignal(
            trade=all_gates_passed,
            direction=direction_str,
            size=size,
            confidence=ridge_confidence,
            regime=regime,
            regime_confidence=regime_confidence,
            tcn_direction=tcn_direction,
            tcn_probability=tcn_probability,
            ridge_confidence=ridge_confidence,
            xgb_momentum=xgb_momentum,
            xgb_acceleration=xgb_acceleration,
            rf_drawdown_pips=rf_drawdown_pips,
            rf_streak_prob=rf_streak_prob,
            histgb_direction=histgb_direction,
            histgb_probability=histgb_probability,
            models_agree=models_agree,
            confidence_gate_passed=confidence_gate_passed,
            momentum_gate_passed=momentum_gate_passed,
            risk_gate_passed=risk_gate_passed,
            regime_gate_passed=regime_gate_passed,
            reason=reason,
        )
    
    def predict_verbose(
        self,
        df: pd.DataFrame,
        equity: Optional[float] = None,
        instrument: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run inference with verbose output for logging/display.
        
        Returns dict with all details formatted for display.
        """
        signal = self.predict(df, equity, instrument=instrument)
        
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

