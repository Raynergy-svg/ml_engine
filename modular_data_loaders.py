"""
Modular Data Loaders for Specialized Ensemble Models.

Each loader prepares features and targets for ONE specific model:
- TCN: Direction prediction from volatility regimes
- XGBoost: Momentum analysis from lagged returns
- Random Forest: Risk assessment (drawdown, streak probability)
- Ridge: Confidence scoring from variance and volume

All models use the SAME temporal split (70/20/10) on the SAME candles,
but each sees DIFFERENT features. No shared gradients, no joint loss.

IMPORTANT: All features are INSTRUMENT-AGNOSTIC (normalized/relative).
This allows models trained on one pair (e.g., GBP_USD) to work on others (e.g., USD_JPY).
We use:
- Returns (percentage changes) instead of raw prices
- Z-scores (standard deviations from mean) instead of absolute values
- Ratios and percentiles instead of raw values
- Normalized indicators (0-1 scale or z-score)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# INSTRUMENT-AGNOSTIC FEATURE COMPUTATION
# =============================================================================

def compute_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute instrument-agnostic normalized features from OHLCV data.
    
    All features are relative/normalized so models work across any instrument:
    - Returns instead of raw prices
    - Z-scores instead of absolute values
    - Ratios and percentiles instead of raw values
    
    This function should be called BEFORE the individual data loaders.
    """
    df = df.copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))
    
    n = len(close)
    
    # =================================================================
    # 1. RETURNS (percentage changes) - Core normalized features
    # =================================================================
    
    # Simple returns
    df['returns_1'] = np.concatenate([[0], np.diff(close) / np.maximum(close[:-1], 1e-10)])
    
    # Multi-period returns
    for period in [2, 3, 5, 10, 20]:
        returns = np.zeros(n)
        for i in range(period, n):
            returns[i] = (close[i] - close[i-period]) / np.maximum(close[i-period], 1e-10)
        df[f'returns_{period}'] = returns
    
    # Log returns (more stable for compounding)
    df['log_returns_1'] = np.concatenate([[0], np.log(np.maximum(close[1:], 1e-10) / np.maximum(close[:-1], 1e-10))])
    
    # =================================================================
    # 2. VOLATILITY (normalized) - ATR ratio, not absolute ATR
    # =================================================================
    
    # True Range as percentage of price
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        ) / np.maximum(close[i-1], 1e-10)
    df['tr_pct'] = tr
    
    # ATR as percentage (rolling average of TR%)
    for period in [5, 10, 14, 20]:
        col_name = f'atr_pct_{period}'
        if col_name not in df.columns:
            atr_pct = np.zeros(n)
            for i in range(period, n):
                atr_pct[i] = np.mean(tr[i-period:i])
            df[col_name] = atr_pct
    
    # Volatility (std of returns)
    for period in [5, 10, 20]:
        vol = np.zeros(n)
        returns = df['returns_1'].values
        for i in range(period, n):
            vol[i] = np.std(returns[i-period:i])
        df[f'volatility_{period}'] = vol
    
    # =================================================================
    # 3. Z-SCORES - Price position relative to recent history
    # =================================================================
    
    for period in [10, 20, 50]:
        zscore = np.zeros(n)
        for i in range(period, n):
            window = close[i-period:i]
            mean = np.mean(window)
            std = np.std(window)
            if std > 1e-10:
                zscore[i] = (close[i] - mean) / std
            else:
                zscore[i] = 0
        df[f'zscore_{period}'] = np.clip(zscore, -4, 4)  # Clip extreme values
    
    # =================================================================
    # 4. PERCENTILE RANKS - Where is price in recent range?
    # =================================================================
    
    for period in [10, 20, 50]:
        pct_rank = np.zeros(n)
        for i in range(period, n):
            window = close[i-period:i+1]
            rank = np.sum(window <= close[i]) / len(window)
            pct_rank[i] = rank
        df[f'pct_rank_{period}'] = pct_rank
    
    # =================================================================
    # 5. PRICE STRUCTURE (normalized ratios)
    # =================================================================
    
    # High-Low range as percentage
    df['hl_range_pct'] = (high - low) / np.maximum(close, 1e-10)
    
    # Body size as percentage (open-close range)
    df['body_pct'] = (close - open_) / np.maximum(close, 1e-10)
    
    # Upper/lower wick ratios
    body_high = np.maximum(close, open_)
    body_low = np.minimum(close, open_)
    hl_range = high - low + 1e-10
    df['upper_wick_ratio'] = (high - body_high) / hl_range
    df['lower_wick_ratio'] = (body_low - low) / hl_range
    df['body_ratio'] = np.abs(close - open_) / hl_range
    
    # =================================================================
    # 6. MOMENTUM INDICATORS (already normalized 0-100 or -1 to 1)
    # =================================================================
    
    # RSI (0-100) - keep if exists, otherwise compute
    if 'rsi' not in df.columns:
        rsi = np.full(n, 50.0)
        for i in range(14, n):
            gains = []
            losses = []
            for j in range(1, 15):
                change = close[i-14+j] - close[i-14+j-1]
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(abs(change))
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100 if avg_gain > 0 else 50
        df['rsi'] = rsi
    
    # Normalize RSI to 0-1
    df['rsi_norm'] = df['rsi'].values / 100.0
    
    # Stochastic %K (0-100) - keep if exists, otherwise compute
    if 'stoch_k' not in df.columns:
        stoch_k = np.full(n, 50.0)
        for i in range(14, n):
            lowest = np.min(low[i-14:i+1])
            highest = np.max(high[i-14:i+1])
            if highest - lowest > 1e-10:
                stoch_k[i] = 100 * (close[i] - lowest) / (highest - lowest)
        df['stoch_k'] = stoch_k
    df['stoch_k_norm'] = df['stoch_k'].values / 100.0
    
    # =================================================================
    # 7. TREND INDICATORS (normalized)
    # =================================================================
    
    # SMA ratios (price relative to moving average)
    for period in [5, 10, 20, 50]:
        sma = np.zeros(n)
        for i in range(period, n):
            sma[i] = np.mean(close[i-period:i])
        sma_ratio = np.zeros(n)
        for i in range(period, n):
            if sma[i] > 1e-10:
                sma_ratio[i] = (close[i] - sma[i]) / sma[i]
        df[f'sma_ratio_{period}'] = sma_ratio
    
    # EMA ratios
    for period in [12, 26]:
        ema = np.zeros(n)
        alpha = 2 / (period + 1)
        ema[0] = close[0]
        for i in range(1, n):
            ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
        ema_ratio = np.zeros(n)
        for i in range(period, n):
            if ema[i] > 1e-10:
                ema_ratio[i] = (close[i] - ema[i]) / ema[i]
        df[f'ema_ratio_{period}'] = ema_ratio
    
    # MACD (normalized by price)
    ema12 = np.zeros(n)
    ema26 = np.zeros(n)
    alpha12 = 2 / 13
    alpha26 = 2 / 27
    ema12[0] = close[0]
    ema26[0] = close[0]
    for i in range(1, n):
        ema12[i] = alpha12 * close[i] + (1 - alpha12) * ema12[i-1]
        ema26[i] = alpha26 * close[i] + (1 - alpha26) * ema26[i-1]
    macd_line = ema12 - ema26
    df['macd_norm'] = macd_line / np.maximum(close, 1e-10)  # Normalize by price
    
    # MACD signal
    signal = np.zeros(n)
    alpha9 = 2 / 10
    signal[0] = macd_line[0]
    for i in range(1, n):
        signal[i] = alpha9 * macd_line[i] + (1 - alpha9) * signal[i-1]
    df['macd_signal_norm'] = signal / np.maximum(close, 1e-10)
    df['macd_hist_norm'] = (macd_line - signal) / np.maximum(close, 1e-10)
    
    # =================================================================
    # 8. VOLUME INDICATORS (normalized)
    # =================================================================
    
    # Volume ratio (current vs average)
    for period in [5, 10, 20]:
        vol_avg = np.zeros(n)
        for i in range(period, n):
            vol_avg[i] = np.mean(volume[i-period:i])
        vol_ratio = np.zeros(n)
        for i in range(period, n):
            if vol_avg[i] > 1e-10:
                vol_ratio[i] = volume[i] / vol_avg[i]
        df[f'volume_ratio_{period}'] = np.clip(vol_ratio, 0, 5)  # Clip extreme spikes
    
    # Volume z-score
    vol_zscore = np.zeros(n)
    for i in range(20, n):
        window = volume[i-20:i]
        mean = np.mean(window)
        std = np.std(window)
        if std > 1e-10:
            vol_zscore[i] = (volume[i] - mean) / std
    df['volume_zscore'] = np.clip(vol_zscore, -4, 4)
    
    # =================================================================
    # 9. CROSSOVER SIGNALS (binary 0/1)
    # =================================================================
    
    # SMA crossovers
    if 'sma_ratio_5' in df.columns and 'sma_ratio_20' in df.columns:
        df['sma_cross_5_20'] = (df['sma_ratio_5'] > df['sma_ratio_20']).astype(np.float32)
    
    # MACD crossover
    df['macd_cross'] = (df['macd_norm'] > df['macd_signal_norm']).astype(np.float32)
    
    # RSI overbought/oversold
    df['rsi_oversold'] = (df['rsi'] < 30).astype(np.float32)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(np.float32)
    
    # =================================================================
    # 10. HIGHER HIGHS / LOWER LOWS (normalized counts)
    # =================================================================
    
    hh_count = np.zeros(n, dtype=np.float32)
    ll_count = np.zeros(n, dtype=np.float32)
    for i in range(10, n):
        hh = 0
        ll = 0
        for j in range(1, 10):
            if high[i-j] > high[i-j-1]:
                hh += 1
            if low[i-j] < low[i-j-1]:
                ll += 1
        hh_count[i] = hh / 9.0
        ll_count[i] = ll / 9.0
    df['higher_high_ratio'] = hh_count
    df['lower_low_ratio'] = ll_count
    
    logger.info(f"Computed {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'time']])} normalized features")
    
    return df


def get_normalized_feature_names() -> Dict[str, List[str]]:
    """
    Return lists of normalized feature names for each model type.
    These features are instrument-agnostic and work across any currency pair.
    """
    return {
        'direction': [
            # Returns (core direction signal)
            'returns_1', 'returns_2', 'returns_3', 'returns_5', 'returns_10',
            # Z-scores (relative position)
            'zscore_10', 'zscore_20', 'zscore_50',
            # Percentile ranks
            'pct_rank_10', 'pct_rank_20',
            # Momentum indicators
            'rsi_norm', 'stoch_k_norm',
            # Trend indicators
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20',
            'ema_ratio_12', 'ema_ratio_26',
            'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
            # Crossovers
            'sma_cross_5_20', 'macd_cross',
            # Structure
            'higher_high_ratio', 'lower_low_ratio',
        ],
        'momentum': [
            # Returns (momentum is about rate of change)
            'returns_1', 'returns_2', 'returns_3', 'returns_5', 'returns_10', 'returns_20',
            'log_returns_1',
            # Volatility (momentum context)
            'atr_pct_5', 'atr_pct_10', 'atr_pct_14',
            'volatility_5', 'volatility_10',
            # Momentum indicators
            'rsi_norm', 'stoch_k_norm',
            'macd_norm', 'macd_hist_norm',
            # Volume (confirms momentum)
            'volume_ratio_5', 'volume_ratio_10', 'volume_zscore',
        ],
        'risk': [
            # Volatility (core risk measure)
            'atr_pct_5', 'atr_pct_10', 'atr_pct_14', 'atr_pct_20',
            'volatility_5', 'volatility_10', 'volatility_20',
            'tr_pct',
            # Price structure (risk from gaps/wicks)
            'hl_range_pct', 'upper_wick_ratio', 'lower_wick_ratio',
            # Z-scores (extreme positions = higher risk)
            'zscore_10', 'zscore_20',
            # Returns (recent moves indicate risk)
            'returns_1', 'returns_5', 'returns_10',
            # Volume (high volume = potential volatility)
            'volume_ratio_10', 'volume_zscore',
        ],
        'confidence': [
            # Volatility (low vol = higher confidence)
            'atr_pct_10', 'atr_pct_20',
            'volatility_10', 'volatility_20',
            # Trend clarity (clear trend = higher confidence)
            'sma_ratio_20', 'ema_ratio_26',
            'macd_norm',
            # RSI (mid-range = unclear, extremes = clearer)
            'rsi_norm',
            # Z-score (extreme = clearer signal)
            'zscore_20',
            # Volume (high volume confirms signal)
            'volume_ratio_10', 'volume_ratio_20',
            # Returns consistency
            'returns_5', 'returns_10', 'returns_20',
        ],
    }


def temporal_split(
    n_samples: int,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    test_frac: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create chronological train/val/test indices (no shuffle, no overlap).
    
    Args:
        n_samples: Total number of samples
        train_frac: Fraction for training (default 0.7)
        val_frac: Fraction for validation (default 0.2)
        test_frac: Fraction for test (default 0.1)
    
    Returns:
        Tuple of (train_idx, val_idx, test_idx) numpy arrays
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 0.001, "Fractions must sum to 1"
    
    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))
    
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n_samples)
    
    return train_idx, val_idx, test_idx


def _ensure_features_exist(df: pd.DataFrame, features: List[str]) -> List[str]:
    """Filter features to only those that exist in the dataframe."""
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.warning(f"Missing features: {missing}")
    return available


def _find_features_by_pattern(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    """
    Find features by partial name matching.
    More flexible than exact matching - finds any column containing the pattern.
    """
    found = []
    for col in df.columns:
        col_lower = col.lower()
        for pattern in patterns:
            if pattern.lower() in col_lower:
                if col not in found:
                    found.append(col)
                break
    return found


def _select_features_for_task(
    df: pd.DataFrame,
    preferred_features: List[str],
    fallback_patterns: List[str],
    min_features: int = 5,
    exclude_cols: List[str] = None,
) -> List[str]:
    """
    Select features for a task, with fallbacks.
    
    1. First try exact matches from preferred_features
    2. If not enough, try pattern matching with fallback_patterns
    3. If still not enough, use any available numeric columns
    """
    exclude_cols = exclude_cols or ['open', 'high', 'low', 'close', 'volume', 'time', 'timestamp']
    
    # Try exact matches first
    features = _ensure_features_exist(df, preferred_features)
    
    # If not enough, try pattern matching
    if len(features) < min_features:
        pattern_features = _find_features_by_pattern(df, fallback_patterns)
        for f in pattern_features:
            if f not in features and f not in exclude_cols:
                features.append(f)
    
    # If still not enough, use any numeric columns
    if len(features) < min_features:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in features and col not in exclude_cols:
                features.append(col)
            if len(features) >= min_features:
                break
    
    return features


# =============================================================================
# REGIME DATA LOADER - Market Regime Classification (Trend/Chop/Mean-Revert)
# =============================================================================

def load_regime_data(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    lookback: int = 20,  # Bars to look back for regime detection
    lookahead: int = 12,  # Bars ahead to confirm regime
) -> Dict[str, np.ndarray]:
    """
    Load data for market regime classification (3 classes).
    
    Regimes:
    - 0 = TREND: Strong directional movement (ADX high, consistent direction)
    - 1 = CHOP: Sideways, no clear direction (low ADX, high reversals)
    - 2 = MEAN_REVERT: Overextended, likely to reverse (RSI extreme, z-score extreme)
    
    The Transformer becomes a "bouncer" - it tells you WHAT KIND of market you're in,
    not which direction to trade. Direction decisions are delegated to other models
    based on regime:
    - TREND: Let XGBoost/Ridge/RF decide direction
    - CHOP: Skip trading entirely
    - MEAN_REVERT: Fade 2-bar momentum
    
    Features: NORMALIZED regime indicators (instrument-agnostic)
    Target: 3-class regime (0=trend, 1=chop, 2=mean_revert)
    """
    logger.info(f"Loading regime data (lookback={lookback}, lookahead={lookahead})...")
    
    # Compute normalized features if not already present
    if 'returns_1' not in df.columns:
        df = compute_normalized_features(df)
    
    # REGIME FEATURES - indicators that describe market state
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
    
    # Get available features
    features = _ensure_features_exist(df, regime_features)
    
    # Fallback if needed
    if len(features) < 10:
        fallback = ['adx', 'rsi', 'atr', 'volatility', 'zscore', 'returns', 'bb_position']
        pattern_features = _find_features_by_pattern(df, fallback)
        for f in pattern_features:
            if f not in features and f not in ['open', 'high', 'low', 'close', 'volume', 'time']:
                features.append(f)
    
    if len(features) < 5:
        raise ValueError(f"Regime model needs at least 5 features, got {len(features)}")
    
    logger.info(f"Regime features: {features[:10]}{'...' if len(features) > 10 else ''} ({len(features)} total)")
    
    # Extract feature matrix
    X = df[features].values.astype(np.float32)
    
    # Create regime labels
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(close)
    
    # Get ADX and RSI if available (key regime indicators)
    adx = df['adx'].values if 'adx' in df.columns else np.full(n, 25.0)
    rsi = df['rsi'].values if 'rsi' in df.columns else np.full(n, 50.0)
    zscore = df['zscore_20'].values if 'zscore_20' in df.columns else np.zeros(n)
    
    # Initialize labels
    y = np.full(n, 1, dtype=np.int32)  # Default: CHOP (safest default)
    
    n_trend = 0
    n_chop = 0
    n_mean_revert = 0
    
    for i in range(lookback, n - lookahead):
        # === REGIME DETECTION LOGIC ===
        
        # Look at recent price action
        recent_close = close[i-lookback:i+1]
        recent_high = high[i-lookback:i+1]
        recent_low = low[i-lookback:i+1]
        
        # Calculate metrics
        price_range = (recent_high.max() - recent_low.min()) / close[i] if close[i] > 0 else 0
        
        # Directional consistency: how often did price move in same direction?
        returns = np.diff(recent_close) / recent_close[:-1]
        returns = returns[~np.isnan(returns)]
        if len(returns) > 0:
            up_ratio = (returns > 0).mean()
            consistency = max(up_ratio, 1 - up_ratio)  # 0.5 = random, 1.0 = perfectly consistent
        else:
            consistency = 0.5
        
        # Current indicators
        current_adx = adx[i] if not np.isnan(adx[i]) else 25.0
        current_rsi = rsi[i] if not np.isnan(rsi[i]) else 50.0
        current_zscore = zscore[i] if not np.isnan(zscore[i]) else 0.0
        
        # === CLASSIFICATION RULES ===
        
        # MEAN REVERT: Overextended (RSI extreme OR z-score extreme)
        rsi_extreme = current_rsi < 25 or current_rsi > 75
        zscore_extreme = abs(current_zscore) > 2.0
        
        if rsi_extreme or zscore_extreme:
            # Confirm with lookahead: did price actually revert?
            future_close = close[i + lookahead]
            current_close = close[i]
            future_return = (future_close - current_close) / current_close if current_close > 0 else 0
            
            # Mean revert confirmed if price moved opposite to extension
            if current_rsi > 75 and future_return < -0.001:  # Overbought -> went down
                y[i] = 2  # MEAN_REVERT
                n_mean_revert += 1
                continue
            elif current_rsi < 25 and future_return > 0.001:  # Oversold -> went up
                y[i] = 2  # MEAN_REVERT
                n_mean_revert += 1
                continue
            elif current_zscore > 2.0 and future_return < -0.001:
                y[i] = 2  # MEAN_REVERT
                n_mean_revert += 1
                continue
            elif current_zscore < -2.0 and future_return > 0.001:
                y[i] = 2  # MEAN_REVERT
                n_mean_revert += 1
                continue
        
        # TREND: Strong ADX + directional consistency
        if current_adx > 25 and consistency > 0.6:
            # Confirm with lookahead: did trend continue?
            future_close = close[i + lookahead]
            current_close = close[i]
            future_return = (future_close - current_close) / current_close if current_close > 0 else 0
            
            # Trend confirmed if price moved significantly in either direction
            if abs(future_return) > 0.002:  # 0.2% move confirms trend
                y[i] = 0  # TREND
                n_trend += 1
                continue
        
        # CHOP: Everything else (low ADX, inconsistent, no clear signal)
        y[i] = 1  # CHOP
        n_chop += 1
    
    # Trim edges (no lookback/lookahead data)
    valid_start = lookback
    valid_end = n - lookahead
    X = X[valid_start:valid_end]
    y = y[valid_start:valid_end]
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Temporal split
    train_idx, val_idx, test_idx = temporal_split(len(X), *split)
    
    # Label statistics
    total = n_trend + n_chop + n_mean_revert
    label_stats = {
        'n_trend': n_trend,
        'n_chop': n_chop,
        'n_mean_revert': n_mean_revert,
        'trend_rate': n_trend / max(total, 1),
        'chop_rate': n_chop / max(total, 1),
        'mean_revert_rate': n_mean_revert / max(total, 1),
        'lookback': lookback,
        'lookahead': lookahead,
    }
    
    logger.info(f"Regime labels: {n_trend} trend ({label_stats['trend_rate']:.1%}), "
                f"{n_chop} chop ({label_stats['chop_rate']:.1%}), "
                f"{n_mean_revert} mean_revert ({label_stats['mean_revert_rate']:.1%})")
    
    result = {
        'X_train': X[train_idx],
        'y_train': y[train_idx],
        'X_val': X[val_idx],
        'y_val': y[val_idx],
        'X_test': X[test_idx],
        'y_test': y[test_idx],
        'feature_names': features,
        'label_stats': label_stats,
        'class_names': ['trend', 'chop', 'mean_revert'],
    }
    
    logger.info(f"Regime data: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, features={len(features)}")
    return result


# =============================================================================
# DIRECTION DATA LOADER - Direction Prediction (for TCN/Transformer)
# =============================================================================

def load_direction_data(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    lookahead: int = 6,
    threshold: float = 0.005,  # 0.5% minimum move to label as clear signal
) -> Dict[str, np.ndarray]:
    """
    Load data for direction prediction model (TCN or Transformer).
    
    Uses threshold-based labeling to filter out noise:
    - Only labels moves >= threshold as clear signals (weight=1.0)
    - Moves below threshold are marked unclear (label=0.5, weight=0.0)
    
    Features: NORMALIZED directional indicators (instrument-agnostic)
    Target: Binary direction (1=up, 0=down) over next `lookahead` bars
    
    Args:
        df: DataFrame with OHLCV and features
        split: (train_frac, val_frac, test_frac)
        lookahead: Number of bars ahead for direction label
        threshold: Minimum price change (as fraction) to assign clear label
    
    Returns:
        Dict with X_train, y_train, w_train (weights), X_val, y_val, w_val, 
        X_test, y_test, w_test, feature_names, label_stats
    """
    logger.info(f"Loading direction data (threshold={threshold:.3%}, lookahead={lookahead})...")
    
    # Compute normalized features if not already present
    if 'returns_1' not in df.columns:
        df = compute_normalized_features(df)
    
    # USE ALL AVAILABLE NUMERIC FEATURES (not just hardcoded 24!)
    # This allows the model to leverage all 186+ features from feature engineering
    exclude_cols = {'open', 'high', 'low', 'close', 'volume', 'time', 'timestamp', 'date', 
                    'target', 'label', 'direction', 'y', 'target_direction'}
    
    # Get all numeric columns that aren't excluded
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in all_numeric if col.lower() not in exclude_cols and col not in exclude_cols]
    
    # Remove any columns that look like targets/labels
    features = [f for f in features if not any(x in f.lower() for x in ['target', 'label', 'future', 'forward'])]
    
    # =========================================================================
    # SMART FEATURE SELECTION: Keep top ~60 uncorrelated features
    # =========================================================================
    # VARIANCE-BASED FEATURE SELECTION (no look-ahead bias)
    # Select features with high variance and remove redundant ones
    max_features = 80  # Use more features since we're not overfitting to target
    correlation_threshold = 0.80  # Remove features correlated > 80%
    
    if len(features) > max_features:
        logger.info(f"Selecting top {max_features} uncorrelated features from {len(features)}...")
        
        # Build numeric feature matrix
        feature_matrix = df[features].values.astype(np.float64)
        
        # Score features by VARIANCE (normalized) - no target leakage
        # High variance = potentially informative, avoids near-constant features
        feature_scores = {}
        for idx, f in enumerate(features):
            try:
                f_values = feature_matrix[:, idx]
                valid_mask = np.isfinite(f_values)
                if valid_mask.sum() > 100:
                    vals = f_values[valid_mask]
                    # Coefficient of variation (normalized variance)
                    mean_val = np.abs(np.mean(vals))
                    std_val = np.std(vals)
                    if mean_val > 1e-10:
                        # Use CoV for scale-invariant scoring
                        feature_scores[f] = std_val / mean_val
                    else:
                        # If mean near zero, just use std
                        feature_scores[f] = std_val
            except Exception:
                pass
        
        # Priority features that are known to be useful for direction prediction
        priority_features = [
            'macd_norm', 'macd_signal_norm', 'macd_hist_momentum',
            'rsi_norm', 'rsi_momentum', 'stoch_k_norm', 'stoch_d_norm',
            'adx', 'atr_pct_10', 'atr_pct_20',
            'sma_ratio_10', 'sma_ratio_20', 'ema_ratio_12', 'ema_ratio_26',
            'bb_position', 'bb_width_norm',
            'obv', 'volume_ratio_10', 'volume_sma_20',
            'returns_1', 'returns_5', 'returns_10', 'returns_20',
            'volatility_10', 'volatility_20'
        ]
        
        # Start with priority features that exist and have good variance
        selected = []
        for pf in priority_features:
            if pf in feature_scores and len(selected) < max_features // 2:
                selected.append(pf)
        
        # Sort remaining by variance score
        remaining = [f for f in feature_scores.keys() if f not in selected]
        sorted_remaining = sorted(remaining, key=lambda x: feature_scores[x], reverse=True)
        
        # Rebuild feature matrix with selected + sorted remaining
        all_candidates = selected + sorted_remaining
        candidate_indices = [features.index(f) for f in all_candidates if f in features]
        candidate_matrix = feature_matrix[:, candidate_indices]
        candidate_features = [all_candidates[i] for i in range(len(all_candidates)) if all_candidates[i] in features]
        
        # Remove highly correlated features
        final_selected = []
        for i, f in enumerate(candidate_features):
            if len(final_selected) >= max_features:
                break
            
            # Check correlation with already selected features
            is_redundant = False
            f_values = candidate_matrix[:, i]
            
            for sel_f in final_selected:
                sel_idx = candidate_features.index(sel_f)
                sel_values = candidate_matrix[:, sel_idx]
                
                # Calculate correlation
                valid = np.isfinite(f_values) & np.isfinite(sel_values)
                if valid.sum() > 100:
                    corr = np.abs(np.corrcoef(f_values[valid], sel_values[valid])[0, 1])
                    if np.isfinite(corr) and corr > correlation_threshold:
                        is_redundant = True
                        break
            
            if not is_redundant:
                final_selected.append(f)
        
        features = final_selected
        logger.info(f"Selected {len(features)} uncorrelated features (variance-based, no target leakage)")
    
    logger.info(f"Direction features: {features[:10]}{'...' if len(features) > 10 else ''} ({len(features)} total)")
    
    # Extract feature matrix
    X = df[features].values.astype(np.float32)
    
    # Create direction labels with THRESHOLD FILTERING
    close = df['close'].values
    n = len(close)
    y = np.full(n, 0.5, dtype=np.float32)  # Default: unclear
    weights = np.zeros(n, dtype=np.float32)  # Default: excluded
    
    n_clear_up = 0
    n_clear_down = 0
    n_unclear = 0
    
    # Handle threshold=0 case: include ALL samples with simple up/down labeling
    use_all_samples = (threshold <= 0)
    
    for i in range(n - lookahead):
        future_close = close[i + lookahead]
        current_close = close[i]
        
        if current_close <= 0:
            continue
        
        pct_change = (future_close - current_close) / current_close
        
        if use_all_samples or abs(pct_change) >= threshold:
            # Label based on direction (with small threshold to handle float precision)
            if pct_change > 1e-10:
                y[i] = 1.0  # UP
                weights[i] = 1.0
                n_clear_up += 1
            elif pct_change < -1e-10:
                y[i] = 0.0  # DOWN
                weights[i] = 1.0
                n_clear_down += 1
            else:
                # Exactly zero change - label as unclear
                y[i] = 0.5
                weights[i] = 0.0
                n_unclear += 1
        else:
            # Unclear signal - move too small (noise)
            y[i] = 0.5
            weights[i] = 0.0  # Exclude from training
            n_unclear += 1
    
    # Drop last `lookahead` rows (no future data)
    X = X[:-lookahead]
    y = y[:-lookahead]
    weights = weights[:-lookahead]
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Temporal split (BEFORE scaling to avoid data leakage)
    train_idx, val_idx, test_idx = temporal_split(len(X), *split)
    
    # =========================================================================
    # FEATURE SCALING: Fit on train, apply to all (prevents data leakage)
    # =========================================================================
    from sklearn.preprocessing import RobustScaler
    
    scaler = RobustScaler()  # Robust to outliers (better for financial data)
    
    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(X[train_idx])
    X_val_scaled = scaler.transform(X[val_idx])
    X_test_scaled = scaler.transform(X[test_idx])
    
    # Clip extreme values after scaling (prevents numerical issues)
    clip_value = 10.0  # Clip to [-10, 10] after robust scaling
    X_train_scaled = np.clip(X_train_scaled, -clip_value, clip_value)
    X_val_scaled = np.clip(X_val_scaled, -clip_value, clip_value)
    X_test_scaled = np.clip(X_test_scaled, -clip_value, clip_value)
    
    # Remove constant features (zero variance after scaling)
    feature_stds = np.std(X_train_scaled, axis=0)
    valid_features = feature_stds > 1e-6
    n_removed = np.sum(~valid_features)
    if n_removed > 0:
        logger.info(f"Removing {n_removed} constant features")
        X_train_scaled = X_train_scaled[:, valid_features]
        X_val_scaled = X_val_scaled[:, valid_features]
        X_test_scaled = X_test_scaled[:, valid_features]
        features = [f for f, v in zip(features, valid_features) if v]
    
    logger.info(f"Feature scaling: max={np.max(np.abs(X_train_scaled)):.2f}, "
                f"mean_abs={np.mean(np.abs(X_train_scaled)):.4f}")
    
    # Label statistics
    total_clear = n_clear_up + n_clear_down
    label_stats = {
        'n_clear_up': n_clear_up,
        'n_clear_down': n_clear_down,
        'n_unclear': n_unclear,
        'clear_rate': total_clear / max(total_clear + n_unclear, 1),
        'up_rate': n_clear_up / max(total_clear, 1),
        'threshold': threshold,
        'lookahead': lookahead,
    }
    
    logger.info(f"Direction labels: {n_clear_up} up, {n_clear_down} down, {n_unclear} unclear "
                f"({label_stats['clear_rate']:.1%} clear, {label_stats['up_rate']:.1%} up)")
    
    result = {
        'X_train': X_train_scaled.astype(np.float32),
        'y_train': y[train_idx],
        'w_train': weights[train_idx],
        'X_val': X_val_scaled.astype(np.float32),
        'y_val': y[val_idx],
        'w_val': weights[val_idx],
        'X_test': X_test_scaled.astype(np.float32),
        'y_test': y[test_idx],
        'w_test': weights[test_idx],
        'feature_names': features,
        'label_stats': label_stats,
        'scaler': scaler,  # Save scaler for inference
    }
    
    logger.info(f"Direction data: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, features={len(features)}")
    return result


def _add_directional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add computed directional features to dataframe.
    
    These features capture market structure and trend direction,
    which are more predictive of direction than volatility features.
    """
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


# Backward compatibility alias
def load_tcn_data(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    lookahead: int = 6,
    threshold: float = 0.005,
) -> Dict[str, np.ndarray]:
    """Backward compatible alias for load_direction_data."""
    return load_direction_data(df, split, lookahead, threshold)


# =============================================================================
# XGBOOST DATA LOADER - Momentum Analysis
# =============================================================================

def load_xgboost_data(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    momentum_window: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Load data for XGBoost model: Momentum analysis from normalized returns.
    
    Features: NORMALIZED returns and momentum indicators (instrument-agnostic)
    Targets:
        - momentum_score (0-1): How fast price is moving
        - acceleration (bool): Is momentum growing or shrinking?
    
    Args:
        df: DataFrame with OHLCV and features
        split: (train_frac, val_frac, test_frac)
        momentum_window: Window for momentum calculation
    
    Returns:
        Dict with X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    """
    logger.info("Loading XGBoost data (momentum analysis)...")
    
    # Compute normalized features if not already present
    if 'returns_1' not in df.columns:
        df = compute_normalized_features(df)
    
    # NORMALIZED MOMENTUM FEATURES - instrument-agnostic
    normalized_features = get_normalized_feature_names()['momentum']
    
    # Legacy features as fallback
    legacy_features = [
        'returns', 'roc_5', 'roc_10', 'roc_20',
        'high_low_ratio', 'momentum_10', 'macd', 'macd_hist',
        'stoch_k', 'stoch_d', 'mfi',
    ]
    
    # Prefer normalized features, fallback to legacy
    features = _ensure_features_exist(df, normalized_features)
    if len(features) < 10:
        logger.info(f"Only {len(features)} normalized features found, adding legacy features")
        legacy_available = _ensure_features_exist(df, legacy_features)
        for f in legacy_available:
            if f not in features:
                features.append(f)
    
    # Fallback patterns
    fallback_patterns = ['return', 'atr_pct', 'volatility', 'volume_ratio', 'macd_norm', 'rsi_norm']
    if len(features) < 5:
        pattern_features = _find_features_by_pattern(df, fallback_patterns)
        for f in pattern_features:
            if f not in features and f not in ['open', 'high', 'low', 'close', 'volume', 'time']:
                features.append(f)
    
    if len(features) < 5:
        raise ValueError(f"XGBoost needs at least 5 features, got {len(features)}")
    
    logger.info(f"XGBoost features: {features[:10]}{'...' if len(features) > 10 else ''} ({len(features)} total)")
    
    # Extract feature matrix
    X = df[features].values.astype(np.float32)
    
    # Create momentum targets
    close = df['close'].values
    n = len(close)
    
    # Calculate rolling momentum as absolute return rate
    returns = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / max(close[i-1], 1e-8)
    
    # Calculate raw rolling momentum first
    raw_momentum_all = np.zeros(n, dtype=np.float32)
    for i in range(momentum_window, n):
        window_returns = np.abs(returns[i-momentum_window:i])
        raw_momentum_all[i] = np.mean(window_returns)
    
    # STABLE NORMALIZATION: Use fixed percentile-based scaling
    # This ensures consistent behavior across different market conditions
    valid_raw = raw_momentum_all[momentum_window:]
    valid_raw = valid_raw[valid_raw > 0]
    
    if len(valid_raw) > 0:
        p50_momentum = np.percentile(valid_raw, 50)  # Median momentum
        p90_momentum = np.percentile(valid_raw, 90)  # High momentum
        # Scale so median maps to 0.3, P90 maps to 0.7
        # This ensures threshold of 0.15 catches ~30% of bars
        norm_factor = p50_momentum / 0.3 if p50_momentum > 0 else 0.001
    else:
        norm_factor = 0.001
    
    logger.info(f"XGBoost momentum: P50={p50_momentum:.6f}, P90={p90_momentum:.6f}, norm_factor={norm_factor:.6f}")
    
    # Normalize to 0-1 scale with adaptive factor
    momentum_score = np.zeros(n, dtype=np.float32)
    for i in range(momentum_window, n):
        momentum_score[i] = min(raw_momentum_all[i] / norm_factor, 1.0)
    
    # Acceleration: Is momentum growing? Compare current vs previous momentum
    acceleration = np.zeros(n, dtype=np.float32)
    for i in range(momentum_window + 5, n):
        current_mom = momentum_score[i]
        prev_mom = momentum_score[i - 5]
        acceleration[i] = 1.0 if current_mom > prev_mom else 0.0
    
    # Combine targets: [momentum_score, acceleration]
    y = np.column_stack([momentum_score, acceleration]).astype(np.float32)
    
    # Drop initial rows without valid momentum
    valid_start = momentum_window + 5
    X = X[valid_start:]
    y = y[valid_start:]
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Temporal split
    train_idx, val_idx, test_idx = temporal_split(len(X), *split)
    
    result = {
        'X_train': X[train_idx],
        'y_train': y[train_idx],
        'X_val': X[val_idx],
        'y_val': y[val_idx],
        'X_test': X[test_idx],
        'y_test': y[test_idx],
        'feature_names': features,
        'momentum_norm_factor': float(norm_factor),  # Save for inference
    }
    
    logger.info(f"XGBoost data: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, features={len(features)}")
    return result


# =============================================================================
# RANDOM FOREST DATA LOADER - Risk Assessment
# =============================================================================

def load_rf_data(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    drawdown_horizon: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Load data for Random Forest model: Risk assessment.
    
    Features: NORMALIZED volatility and risk indicators (instrument-agnostic)
    Targets:
        - expected_drawdown_pct: Max adverse excursion as % of price (not pips!)
        - streak_prob: Probability that losing streak exceeds threshold
    
    Args:
        df: DataFrame with OHLCV and features
        split: (train_frac, val_frac, test_frac)
        drawdown_horizon: Bars ahead to measure drawdown
    
    Returns:
        Dict with X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    """
    logger.info("Loading Random Forest data (risk assessment)...")
    
    # Compute normalized features if not already present
    if 'returns_1' not in df.columns:
        df = compute_normalized_features(df)
    
    # NORMALIZED RISK FEATURES - instrument-agnostic
    normalized_features = get_normalized_feature_names()['risk']
    
    # Legacy features as fallback
    legacy_features = [
        'atr', 'volatility_5', 'volatility_10', 'volatility_20',
        'high_low_ratio', 'bb_width_20',
        'returns', 'momentum_10',
    ]
    
    # Prefer normalized features, fallback to legacy
    features = _ensure_features_exist(df, normalized_features)
    if len(features) < 10:
        logger.info(f"Only {len(features)} normalized features found, adding legacy features")
        legacy_available = _ensure_features_exist(df, legacy_features)
        for f in legacy_available:
            if f not in features:
                features.append(f)
    
    # Fallback patterns
    fallback_patterns = ['atr_pct', 'volatility', 'tr_pct', 'hl_range', 'zscore', 'return']
    if len(features) < 5:
        pattern_features = _find_features_by_pattern(df, fallback_patterns)
        for f in pattern_features:
            if f not in features and f not in ['open', 'high', 'low', 'close', 'volume', 'time']:
                features.append(f)
    
    if len(features) < 5:
        raise ValueError(f"RF needs at least 5 features, got {len(features)}")
    
    logger.info(f"RF features: {features[:10]}{'...' if len(features) > 10 else ''} ({len(features)} total)")
    
    # Extract feature matrix
    X = df[features].values.astype(np.float32)
    
    # Calculate risk targets based on CURRENT VOLATILITY (learnable!)
    # Instead of future drawdown, use ATR-scaled risk which IS predictable
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    n = len(close)
    
    # Get ATR as base for expected drawdown (this IS learnable from features)
    atr_pct = None
    if 'atr_pct_14' in df.columns:
        atr_pct = df['atr_pct_14'].values.astype(np.float64)
        # Check if it's valid (not all zeros/NaN)
        valid_mask = ~np.isnan(atr_pct) & (atr_pct > 0)
        if np.sum(valid_mask) < 10:
            atr_pct = None
        else:
            # atr_pct_14 should be decimal (e.g., 0.001 = 0.1%)
            # Typical forex ATR% is 0.0003-0.005 (3-50 bps per bar)
            mean_val = np.nanmean(np.abs(atr_pct[valid_mask]))
            logger.info(f"RF: Raw atr_pct_14 mean: {mean_val:.6f}")
            
            # If mean > 0.1 (10%), values might be in percentage form (need /100)
            # If mean > 1.0, definitely wrong scale
            if mean_val > 0.1:
                atr_pct = atr_pct / 100.0  # Convert to decimal
                logger.info(f"RF: Converted atr_pct_14 from percentage to decimal")
            
            # Robust outlier handling: cap at 99th percentile (max ~5% for forex)
            valid_atr = atr_pct[valid_mask]
            p99 = min(np.nanpercentile(valid_atr, 99), 0.05)  # Cap at 5%
            p01 = np.nanpercentile(valid_atr, 1)
            atr_pct = np.clip(atr_pct, p01, p99)
            
            logger.info(f"RF: Using atr_pct_14, range: {np.nanmin(atr_pct):.6f} to {np.nanmax(atr_pct):.6f}, mean: {np.nanmean(atr_pct):.6f}")
    
    if atr_pct is None:
        # Calculate ATR percentage manually - more robust version
        logger.info("RF: Computing ATR manually (atr_pct_14 not available or invalid)")
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        ])
        
        # Rolling ATR using cumsum for efficiency
        atr = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i < 14:
                atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
            else:
                atr[i] = np.mean(tr[i-13:i+1])
        
        # Convert to percentage
        atr_pct = atr / np.maximum(close, 1e-8)
        logger.info(f"RF: ATR pct range: {np.nanmin(atr_pct):.6f} to {np.nanmax(atr_pct):.6f}")
    
    # Expected drawdown = 2x ATR (typical stop loss distance)
    # Scaled to 0-1 range: 2% drawdown -> 0.02
    expected_drawdown_pct = np.clip(atr_pct * 2, 0.0001, 0.10).astype(np.float32)  # Min 1bp, cap at 10%
    
    # Streak probability based on recent volatility trend
    # High volatility increasing = higher streak probability
    volatility_10 = df['volatility_10'].values.astype(np.float64) if 'volatility_10' in df.columns else None
    volatility_20 = df['volatility_20'].values.astype(np.float64) if 'volatility_20' in df.columns else None
    
    # Fallback if volatility columns not available
    if volatility_10 is None or np.nansum(np.abs(volatility_10)) < 1e-10:
        logger.info("RF: Computing volatility_10 manually")
        returns = np.diff(close, prepend=close[0]) / np.maximum(np.roll(close, 1), 1e-8)
        returns[0] = 0
        volatility_10 = np.array([np.std(returns[max(0,i-9):i+1]) if i >= 1 else 0.01 for i in range(n)])
    
    if volatility_20 is None or np.nansum(np.abs(volatility_20)) < 1e-10:
        logger.info("RF: Computing volatility_20 manually")
        returns = np.diff(close, prepend=close[0]) / np.maximum(np.roll(close, 1), 1e-8)
        returns[0] = 0
        volatility_20 = np.array([np.std(returns[max(0,i-19):i+1]) if i >= 1 else 0.01 for i in range(n)])
    
    streak_prob = np.zeros(n, dtype=np.float32)
    for i in range(20, n):
        # If short-term vol > long-term vol, higher streak probability
        if volatility_20[i] > 1e-10:
            vol_ratio = volatility_10[i] / volatility_20[i]
            streak_prob[i] = np.clip((vol_ratio - 0.8) / 0.4, 0, 1)  # 0.8->0, 1.2->1
        else:
            streak_prob[i] = 0.5
    
    # Fill early values with mean
    mean_streak = np.mean(streak_prob[20:]) if n > 20 else 0.5
    streak_prob[:20] = mean_streak
    
    logger.info(f"RF targets: drawdown_pct range [{np.min(expected_drawdown_pct):.4f}, {np.max(expected_drawdown_pct):.4f}], "
                f"streak range [{np.min(streak_prob):.4f}, {np.max(streak_prob):.4f}]")
    
    # Combine targets: [expected_drawdown_pct, streak_prob]
    y = np.column_stack([expected_drawdown_pct, streak_prob]).astype(np.float32)
    
    # No need to drop rows - we're not using future data anymore
    # Just drop first 20 rows for volatility calculation warmup
    valid_start = 20
    X = X[valid_start:]
    y = y[valid_start:]
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Temporal split
    train_idx, val_idx, test_idx = temporal_split(len(X), *split)
    
    result = {
        'X_train': X[train_idx],
        'y_train': y[train_idx],
        'X_val': X[val_idx],
        'y_val': y[val_idx],
        'X_test': X[test_idx],
        'y_test': y[test_idx],
        'feature_names': features,
    }
    
    logger.info(f"RF data: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, features={len(features)}")
    return result


# =============================================================================
# RIDGE DATA LOADER - Confidence Scoring
# =============================================================================

def load_ridge_data(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    confidence_window: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Load data for Ridge model: Confidence scoring from NORMALIZED variance and volume.
    
    Features: NORMALIZED volatility, volume ratios, stability metrics (instrument-agnostic)
    Target: Confidence score (0-100)
        - High confidence: Low variance + smooth volume increase
        - Low confidence: High variance + volume spikes/drops
    
    Args:
        df: DataFrame with OHLCV and features
        split: (train_frac, val_frac, test_frac)
        confidence_window: Window for stability calculation
    
    Returns:
        Dict with X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    """
    logger.info("Loading Ridge data (confidence scoring)...")
    
    # Compute normalized features if not already present
    if 'returns_1' not in df.columns:
        df = compute_normalized_features(df)
    
    # NORMALIZED CONFIDENCE FEATURES - instrument-agnostic
    normalized_features = get_normalized_feature_names()['confidence']
    
    # Legacy features as fallback
    legacy_features = [
        'volatility_5', 'volatility_10', 'volatility_20',
        'atr', 'bb_width_20', 'bb_position_20',
        'adx', 'returns',
    ]
    
    # Prefer normalized features, fallback to legacy
    features = _ensure_features_exist(df, normalized_features)
    if len(features) < 8:
        logger.info(f"Only {len(features)} normalized features found, adding legacy features")
        legacy_available = _ensure_features_exist(df, legacy_features)
        for f in legacy_available:
            if f not in features:
                features.append(f)
    
    # Fallback patterns
    fallback_patterns = ['atr_pct', 'volatility', 'volume_ratio', 'sma_ratio', 'return', 'zscore']
    if len(features) < 5:
        pattern_features = _find_features_by_pattern(df, fallback_patterns)
        for f in pattern_features:
            if f not in features and f not in ['open', 'high', 'low', 'close', 'volume', 'time']:
                features.append(f)
    
    if len(features) < 5:
        raise ValueError(f"Ridge needs at least 5 features, got {len(features)}")
    
    logger.info(f"Ridge features: {features[:10]}{'...' if len(features) > 10 else ''} ({len(features)} total)")
    
    # Extract feature matrix
    X = df[features].values.astype(np.float32)
    
    # Calculate confidence based on TREND CLARITY (learnable from ADX, RSI, volatility)
    # High confidence = strong trend (high ADX) + RSI not extreme + low volatility
    # This IS learnable because these are computed from the same features!
    n = len(df)
    
    # Get indicators
    adx = df['adx'].values if 'adx' in df.columns else np.ones(n) * 25
    rsi = df['rsi'].values if 'rsi' in df.columns else np.ones(n) * 50
    atr_pct = df['atr_pct_14'].values if 'atr_pct_14' in df.columns else np.ones(n) * 0.01
    
    # Additional indicators for richer confidence signal
    bb_pos = df['bb_position_20'].values if 'bb_position_20' in df.columns else np.ones(n) * 0.5
    volume_ratio = df['volume_ratio_20'].values if 'volume_ratio_20' in df.columns else np.ones(n) * 1.0
    
    # Compute ADX percentile thresholds from actual data for better scaling
    adx_valid = adx[~np.isnan(adx)]
    adx_p25 = np.percentile(adx_valid, 25) if len(adx_valid) > 0 else 15
    adx_p75 = np.percentile(adx_valid, 75) if len(adx_valid) > 0 else 30
    adx_range = max(adx_p75 - adx_p25, 5)  # Avoid division by zero
    
    confidence = np.zeros(n, dtype=np.float32)
    
    for i in range(confidence_window, n):
        # ===== ADX component: Strong trend = high confidence =====
        # Use percentile-based scaling for the instrument's actual ADX range
        # Maps ADX to [0, 1] where p25->0.25, p75->0.75, p90+->1.0
        adx_normalized = (adx[i] - adx_p25) / adx_range
        adx_score = np.clip(adx_normalized * 0.5 + 0.25, 0.0, 1.0)
        
        # ===== RSI component: Not extreme = high confidence =====
        # RSI 40-60: high confidence (centered), 30-70: medium, outside: low
        rsi_distance = abs(rsi[i] - 50)
        rsi_score = max(0, 1.0 - rsi_distance / 25)  # 50->1, 25/75->0
        
        # ===== Volatility component: Low vol = high confidence =====
        # Use instrument-relative scaling (ATR% typically 0.3%-1.5% for FX)
        vol_score = np.clip(1.0 - atr_pct[i] / 0.015, 0.0, 1.0)
        
        # ===== Bollinger Band position: Middle = high confidence =====
        # BB position 0.3-0.7: confident middle, extremes: overbought/oversold
        bb_distance = abs(bb_pos[i] - 0.5)
        bb_score = max(0, 1.0 - bb_distance * 2.5)  # 0.5->1, 0.1/0.9->0
        
        # ===== Volume confirmation: Above average = high confidence =====
        # Volume ratio > 1.0: good conviction, < 0.7: low conviction
        vol_conf_score = np.clip((volume_ratio[i] - 0.7) / 0.6, 0.0, 1.0)
        
        # ===== Combine with weights =====
        # ADX: 35% (trend strength)
        # RSI: 20% (not overbought/oversold)
        # Volatility: 20% (predictable conditions)
        # BB Position: 15% (price location)
        # Volume: 10% (conviction)
        raw_conf = (
            adx_score * 0.35 + 
            rsi_score * 0.20 + 
            vol_score * 0.20 + 
            bb_score * 0.15 + 
            vol_conf_score * 0.10
        )
        
        # Scale to 0-100 with slight boost for high-quality setups
        # Base range 20-80, can reach 10-95 with extreme values
        confidence[i] = 20 + raw_conf * 75  # Maps [0,1] -> [20,95]
    
    y = confidence.astype(np.float32)
    
    # Drop initial rows without valid confidence
    X = X[confidence_window:]
    y = y[confidence_window:]
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=50.0, posinf=100.0, neginf=0.0)
    
    # Temporal split
    train_idx, val_idx, test_idx = temporal_split(len(X), *split)
    
    result = {
        'X_train': X[train_idx],
        'y_train': y[train_idx],
        'X_val': X[val_idx],
        'y_val': y[val_idx],
        'X_test': X[test_idx],
        'y_test': y[test_idx],
        'feature_names': features,
    }
    
    logger.info(f"Ridge data: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, features={len(features)}")
    return result


# =============================================================================
# CONVENIENCE FUNCTION - Load All Data
# =============================================================================

def load_all_modular_data(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    direction_threshold: float = 0.005,
    direction_lookahead: int = 6,
    use_regime: bool = False,
    regime_lookback: int = 20,
    regime_lookahead: int = 12,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load data for all 4 models at once.
    
    IMPORTANT: First computes normalized features that are instrument-agnostic.
    This allows models trained on one pair to work on any other pair.
    
    Args:
        df: DataFrame with OHLCV and features
        split: (train_frac, val_frac, test_frac)
        direction_threshold: Min price change for clear direction labels
        direction_lookahead: Bars ahead for direction prediction
        use_regime: If True, load regime data instead of direction data
        regime_lookback: Bars to look back for regime detection
        regime_lookahead: Bars ahead to confirm regime
    
    Returns:
        Dict with 'direction'/'regime', 'xgboost', 'rf', 'ridge' keys, each containing
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
        (direction also includes w_train, w_val, w_test for sample weights)
    """
    # FIRST: Compute normalized features (instrument-agnostic)
    # This is the key to making models work across different currency pairs
    logger.info("Computing normalized features for instrument-agnostic training...")
    df_normalized = compute_normalized_features(df)
    logger.info(f"DataFrame now has {len(df_normalized.columns)} columns after normalization")
    
    result = {
        'xgboost': load_xgboost_data(df_normalized, split),
        'rf': load_rf_data(df_normalized, split),
        'ridge': load_ridge_data(df_normalized, split),
    }
    
    if use_regime:
        # REGIME MODE: Transformer classifies market regime (trend/chop/mean_revert)
        logger.info("Using REGIME classification mode (3 classes)")
        regime_data = load_regime_data(df_normalized, split, regime_lookback, regime_lookahead)
        result['regime'] = regime_data
    else:
        # DIRECTION MODE: Transformer predicts binary direction (legacy)
        logger.info("Using DIRECTION prediction mode (binary)")
        direction_data = load_direction_data(df_normalized, split, direction_lookahead, direction_threshold)
        result['direction'] = direction_data
        result['tcn'] = direction_data  # Alias for backward compat
    
    return result

