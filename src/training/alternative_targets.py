"""
Alternative Prediction Targets for FX Trading.

Instead of predicting direction (which is ~50% random in efficient markets),
we predict more actionable targets:

1. VOLATILITY REGIME - When to trade (high vol = opportunity)
2. OPTIMAL ENTRY TIMING - When price will move significantly
3. RISK-ADJUSTED RETURN - Expected return per unit of risk
4. TREND STRENGTH - How strong/reliable is the current trend
5. REVERSAL PROBABILITY - Likelihood of trend reversal

These targets are more predictable than direction and help with:
- Position sizing (volatility regime)
- Entry timing (optimal entry)
- Trade filtering (trend strength)
- Risk management (reversal probability)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """Market volatility regimes."""
    LOW = 0      # Ranging, low opportunity
    NORMAL = 1   # Average conditions
    HIGH = 2     # High opportunity, larger moves
    EXTREME = 3  # Crisis/event, high risk


class TrendStrength(Enum):
    """Trend strength categories."""
    NONE = 0       # No trend (ranging)
    WEAK = 1       # Weak trend
    MODERATE = 2   # Moderate trend
    STRONG = 3     # Strong trend


@dataclass
class AlternativeTargetConfig:
    """Configuration for alternative targets."""
    
    # Volatility regime thresholds (ATR percentiles)
    vol_low_threshold: float = 0.25
    vol_high_threshold: float = 0.75
    vol_extreme_threshold: float = 0.95
    
    # Optimal entry (significant move threshold in ATR units)
    significant_move_atr: float = 1.5
    entry_lookahead: int = 12  # Bars to look ahead
    
    # Trend strength (ADX thresholds)
    trend_weak_threshold: float = 20
    trend_moderate_threshold: float = 30
    trend_strong_threshold: float = 45
    
    # Reversal detection
    reversal_lookback: int = 20
    reversal_threshold: float = 0.5  # % retracement


class AlternativeTargetGenerator:
    """
    Generates alternative prediction targets that are more predictable
    than simple direction.
    """
    
    def __init__(self, config: Optional[AlternativeTargetConfig] = None):
        self.config = config or AlternativeTargetConfig()
    
    def generate_all_targets(
        self,
        df: pd.DataFrame,
        include_direction: bool = True,
    ) -> pd.DataFrame:
        """
        Generate all alternative targets.
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            include_direction: Also include direction target
        
        Returns:
            DataFrame with additional target columns
        """
        df = df.copy()
        
        # 1. Volatility Regime
        df = self._add_volatility_regime(df)
        
        # 2. Optimal Entry Timing
        df = self._add_optimal_entry(df)
        
        # 3. Risk-Adjusted Return
        df = self._add_risk_adjusted_return(df)
        
        # 4. Trend Strength
        df = self._add_trend_strength(df)
        
        # 5. Reversal Probability
        df = self._add_reversal_probability(df)
        
        # 6. Move Magnitude (regression target)
        df = self._add_move_magnitude(df)
        
        # 7. Optimal Trade Type
        df = self._add_optimal_trade_type(df)
        
        # Original direction if requested
        if include_direction:
            df = self._add_direction(df)
        
        return df
    
    def _add_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime classification."""
        if "atr" not in df.columns:
            # Calculate ATR if not present
            high = df["high"]
            low = df["low"]
            close = df["close"].shift(1)
            tr = pd.concat([
                high - low,
                (high - close).abs(),
                (low - close).abs()
            ], axis=1).max(axis=1)
            df["atr"] = tr.rolling(14).mean()
        
        # Calculate ATR percentile (rolling)
        atr_percentile = df["atr"].rolling(window=100, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        # Classify regime
        conditions = [
            atr_percentile < self.config.vol_low_threshold,
            atr_percentile < self.config.vol_high_threshold,
            atr_percentile < self.config.vol_extreme_threshold,
            atr_percentile >= self.config.vol_extreme_threshold,
        ]
        choices = [
            VolatilityRegime.LOW.value,
            VolatilityRegime.NORMAL.value,
            VolatilityRegime.HIGH.value,
            VolatilityRegime.EXTREME.value,
        ]
        
        df["target_volatility_regime"] = np.select(conditions, choices, default=1)
        df["target_volatility_percentile"] = atr_percentile
        
        # Binary: is high volatility (tradeable)?
        df["target_is_high_vol"] = (atr_percentile >= self.config.vol_high_threshold).astype(float)
        
        return df
    
    def _add_optimal_entry(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add optimal entry timing target."""
        lookahead = self.config.entry_lookahead
        atr_mult = self.config.significant_move_atr
        
        # Calculate future price range
        future_high = df["high"].shift(-lookahead).rolling(lookahead).max()
        future_low = df["low"].shift(-lookahead).rolling(lookahead).min()
        current_close = df["close"]
        
        # Maximum move in either direction
        up_move = (future_high - current_close) / df["atr"]
        down_move = (current_close - future_low) / df["atr"]
        max_move = pd.concat([up_move, down_move], axis=1).max(axis=1)
        
        # Binary: will there be a significant move?
        df["target_significant_move"] = (max_move >= atr_mult).astype(float)
        
        # Regression: expected move magnitude (in ATR units)
        df["target_move_magnitude_atr"] = max_move.clip(0, 5)  # Cap at 5 ATR
        
        # Which direction will the significant move be?
        df["target_move_direction"] = np.where(
            up_move > down_move,
            1.0,  # Up
            0.0   # Down
        )
        
        # Combined: significant move AND direction
        df["target_optimal_long"] = ((max_move >= atr_mult) & (up_move > down_move)).astype(float)
        df["target_optimal_short"] = ((max_move >= atr_mult) & (down_move > up_move)).astype(float)
        
        return df
    
    def _add_risk_adjusted_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-adjusted return target."""
        lookahead = self.config.entry_lookahead
        
        # Future return
        future_return = df["close"].pct_change(lookahead).shift(-lookahead)
        
        # Risk (realized volatility)
        realized_vol = df["close"].pct_change().rolling(20).std()
        
        # Risk-adjusted return (Sharpe-like)
        df["target_risk_adjusted_return"] = (future_return / realized_vol).clip(-3, 3)
        
        # Binary: positive risk-adjusted return?
        df["target_positive_rar"] = (df["target_risk_adjusted_return"] > 0).astype(float)
        
        return df
    
    def _add_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend strength classification."""
        # Use ADX if available
        if "adx" in df.columns:
            adx = df["adx"]
        else:
            # Calculate simplified ADX
            high = df["high"]
            low = df["low"]
            close = df["close"]
            
            plus_dm = high.diff().clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)
            
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
            adx = dx.rolling(14).mean()
        
        # Classify trend strength
        conditions = [
            adx < self.config.trend_weak_threshold,
            adx < self.config.trend_moderate_threshold,
            adx < self.config.trend_strong_threshold,
            adx >= self.config.trend_strong_threshold,
        ]
        choices = [
            TrendStrength.NONE.value,
            TrendStrength.WEAK.value,
            TrendStrength.MODERATE.value,
            TrendStrength.STRONG.value,
        ]
        
        df["target_trend_strength"] = np.select(conditions, choices, default=1)
        df["target_adx"] = adx
        
        # Binary: is trending (tradeable)?
        df["target_is_trending"] = (adx >= self.config.trend_weak_threshold).astype(float)
        df["target_is_strong_trend"] = (adx >= self.config.trend_moderate_threshold).astype(float)
        
        return df
    
    def _add_reversal_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add reversal probability target."""
        lookback = self.config.reversal_lookback
        threshold = self.config.reversal_threshold
        
        # Calculate swing high/low
        rolling_high = df["high"].rolling(lookback).max()
        rolling_low = df["low"].rolling(lookback).min()
        
        # Current position in range
        range_size = rolling_high - rolling_low
        position_in_range = (df["close"] - rolling_low) / (range_size + 1e-10)
        
        # Reversal indicators
        # Near top of range = potential bearish reversal
        near_top = position_in_range > (1 - threshold)
        # Near bottom of range = potential bullish reversal
        near_bottom = position_in_range < threshold
        
        # RSI extremes (if available)
        if "rsi" in df.columns:
            rsi_overbought = df["rsi"] > 70
            rsi_oversold = df["rsi"] < 30
        else:
            rsi_overbought = pd.Series(False, index=df.index)
            rsi_oversold = pd.Series(False, index=df.index)
        
        # Combined reversal probability
        df["target_bearish_reversal_prob"] = (
            near_top.astype(float) * 0.5 +
            rsi_overbought.astype(float) * 0.5
        )
        df["target_bullish_reversal_prob"] = (
            near_bottom.astype(float) * 0.5 +
            rsi_oversold.astype(float) * 0.5
        )
        
        # Binary: reversal likely?
        df["target_reversal_likely"] = (
            (df["target_bearish_reversal_prob"] > 0.5) |
            (df["target_bullish_reversal_prob"] > 0.5)
        ).astype(float)
        
        return df
    
    def _add_move_magnitude(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add move magnitude regression target."""
        lookahead = self.config.entry_lookahead
        
        # Absolute return (direction-agnostic)
        future_return = df["close"].pct_change(lookahead).shift(-lookahead).abs()
        
        # Normalize by recent volatility
        recent_vol = df["close"].pct_change().rolling(20).std()
        normalized_move = future_return / (recent_vol + 1e-10)
        
        df["target_move_magnitude"] = normalized_move.clip(0, 5)
        
        # Bucketize for classification
        conditions = [
            normalized_move < 0.5,
            normalized_move < 1.0,
            normalized_move < 2.0,
            normalized_move >= 2.0,
        ]
        choices = [0, 1, 2, 3]  # Small, Normal, Large, Very Large
        df["target_move_bucket"] = np.select(conditions, choices, default=1)
        
        return df
    
    def _add_optimal_trade_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add optimal trade type (trend follow vs mean revert)."""
        # Trend following works when: strong trend, high vol
        # Mean reversion works when: ranging, low vol
        
        is_trending = df.get("target_is_trending", pd.Series(0.5, index=df.index))
        is_high_vol = df.get("target_is_high_vol", pd.Series(0.5, index=df.index))
        
        # Score for trend following (0-1)
        trend_follow_score = (is_trending * 0.6 + is_high_vol * 0.4)
        
        # Score for mean reversion (0-1)
        mean_revert_score = ((1 - is_trending) * 0.6 + (1 - is_high_vol) * 0.4)
        
        df["target_trend_follow_score"] = trend_follow_score
        df["target_mean_revert_score"] = mean_revert_score
        
        # Optimal strategy
        df["target_optimal_strategy"] = np.where(
            trend_follow_score > mean_revert_score,
            1,  # Trend follow
            0   # Mean revert
        )
        
        return df
    
    def _add_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple direction target."""
        lookahead = self.config.entry_lookahead
        future_return = df["close"].pct_change(lookahead).shift(-lookahead)
        df["target_direction"] = (future_return > 0).astype(float)
        return df
    
    def get_target_columns(self) -> List[str]:
        """Get list of all target column names."""
        return [
            # Volatility
            "target_volatility_regime",
            "target_volatility_percentile",
            "target_is_high_vol",
            # Entry timing
            "target_significant_move",
            "target_move_magnitude_atr",
            "target_move_direction",
            "target_optimal_long",
            "target_optimal_short",
            # Risk-adjusted
            "target_risk_adjusted_return",
            "target_positive_rar",
            # Trend
            "target_trend_strength",
            "target_adx",
            "target_is_trending",
            "target_is_strong_trend",
            # Reversal
            "target_bearish_reversal_prob",
            "target_bullish_reversal_prob",
            "target_reversal_likely",
            # Move magnitude
            "target_move_magnitude",
            "target_move_bucket",
            # Trade type
            "target_trend_follow_score",
            "target_mean_revert_score",
            "target_optimal_strategy",
            # Direction
            "target_direction",
        ]
    
    def get_classification_targets(self) -> List[str]:
        """Get binary/categorical classification targets."""
        return [
            "target_is_high_vol",
            "target_significant_move",
            "target_optimal_long",
            "target_optimal_short",
            "target_positive_rar",
            "target_is_trending",
            "target_is_strong_trend",
            "target_reversal_likely",
            "target_optimal_strategy",
            "target_direction",
        ]
    
    def get_regression_targets(self) -> List[str]:
        """Get continuous regression targets."""
        return [
            "target_volatility_percentile",
            "target_move_magnitude_atr",
            "target_risk_adjusted_return",
            "target_adx",
            "target_bearish_reversal_prob",
            "target_bullish_reversal_prob",
            "target_move_magnitude",
            "target_trend_follow_score",
            "target_mean_revert_score",
        ]


def analyze_target_predictability(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Analyze which targets are most predictable given the features.
    
    Uses simple correlation and mutual information to estimate
    predictability before training expensive models.
    
    Args:
        df: DataFrame with features and targets
        feature_cols: List of feature column names
        target_cols: List of target column names (auto-detect if None)
    
    Returns:
        DataFrame with predictability scores for each target
    """
    if target_cols is None:
        target_cols = [c for c in df.columns if c.startswith("target_")]
    
    results = []
    
    for target in target_cols:
        if target not in df.columns:
            continue
        
        y = df[target].dropna()
        X = df.loc[y.index, feature_cols].dropna()
        
        # Align indices
        common_idx = y.index.intersection(X.index)
        if len(common_idx) < 100:
            continue
        
        y = y.loc[common_idx]
        X = X.loc[common_idx]
        
        # Calculate correlations with each feature
        correlations = X.corrwith(y).abs()
        
        # Metrics
        max_corr = correlations.max()
        mean_corr = correlations.mean()
        top_5_mean = correlations.nlargest(5).mean()
        
        # Target statistics
        is_binary = y.nunique() <= 2
        class_balance = y.mean() if is_binary else None
        
        results.append({
            "target": target,
            "max_correlation": max_corr,
            "mean_correlation": mean_corr,
            "top_5_correlation": top_5_mean,
            "is_binary": is_binary,
            "class_balance": class_balance,
            "n_samples": len(y),
            "predictability_score": top_5_mean * 2,  # Simple heuristic
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("predictability_score", ascending=False)
    
    return results_df


if __name__ == "__main__":
    # Quick test
    print("Alternative Targets Test")
    print("-" * 50)
    
    # Generate dummy data
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        "open": 100 + np.cumsum(np.random.randn(n) * 0.5),
        "high": 100 + np.cumsum(np.random.randn(n) * 0.5) + np.abs(np.random.randn(n)),
        "low": 100 + np.cumsum(np.random.randn(n) * 0.5) - np.abs(np.random.randn(n)),
        "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
        "volume": np.random.randint(1000, 10000, n),
        "rsi": 50 + np.random.randn(n) * 15,
        "adx": 25 + np.random.randn(n) * 10,
    })
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    
    # Generate targets
    generator = AlternativeTargetGenerator()
    df_with_targets = generator.generate_all_targets(df)
    
    print(f"Generated {len(generator.get_target_columns())} targets")
    print("\nTarget columns:")
    for col in generator.get_target_columns():
        if col in df_with_targets.columns:
            print(f"  {col}: {df_with_targets[col].dropna().mean():.3f} (mean)")
    
    # Analyze predictability
    feature_cols = ["open", "high", "low", "close", "volume", "rsi", "adx"]
    predictability = analyze_target_predictability(df_with_targets, feature_cols)
    
    print("\nTarget Predictability (top 10):")
    print(predictability.head(10).to_string(index=False))
