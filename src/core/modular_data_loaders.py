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
    
    # =================================================================
    # 11. TIME-BASED FEATURES (for session/day patterns)
    # =================================================================
    
    # Try to extract hour and day-of-week from index or time column
    time_col = None
    if isinstance(df.index, pd.DatetimeIndex):
        time_col = df.index
    elif 'time' in df.columns:
        try:
            time_col = pd.to_datetime(df['time'])
        except Exception:
            pass
    elif 'timestamp' in df.columns:
        try:
            time_col = pd.to_datetime(df['timestamp'])
        except Exception:
            pass
    
    if time_col is not None:
        try:
            hours = time_col.hour.values.astype(np.float64)
            dow = time_col.dayofweek.values.astype(np.float64)
            
            # Cyclical encoding (sin/cos) - captures circular nature of time
            df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
            df['dow_sin'] = np.sin(2 * np.pi * dow / 5.0)  # 5 trading days
            df['dow_cos'] = np.cos(2 * np.pi * dow / 5.0)
            
            # Session indicators (London/NY overlap = higher vol)
            # London: 7-16 UTC, NY: 13-22 UTC, Overlap: 13-16 UTC
            df['session_london'] = ((hours >= 7) & (hours < 16)).astype(np.float32)
            df['session_ny'] = ((hours >= 13) & (hours < 22)).astype(np.float32)
            df['session_overlap'] = ((hours >= 13) & (hours < 16)).astype(np.float32)
        except Exception as e:
            logger.debug(f"Could not extract time features: {e}")
    
    # =================================================================
    # 12. VOLUME-MA RATIO (volume leads volatility)
    # =================================================================
    
    if 'volume_ratio_20' not in df.columns and volume is not None:
        vol_ma_20 = np.zeros(n)
        for i in range(20, n):
            vol_ma_20[i] = np.mean(volume[i-20:i])
        vol_ma_ratio = np.zeros(n)
        for i in range(20, n):
            if vol_ma_20[i] > 1e-10:
                vol_ma_ratio[i] = volume[i] / vol_ma_20[i]
        df['volume_ma_ratio'] = np.clip(vol_ma_ratio, 0, 5)
    
    # =================================================================
    # CRITICAL: SANITIZE NaN/Inf VALUES
    # =================================================================
    # This MUST be done to prevent NaN propagation during training which
    # causes loss to become NaN and gradient explosion
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Replace infinities with NaN first
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Forward-fill NaN (use most recent valid value)
    df[numeric_cols] = df[numeric_cols].ffill()
    
    # Backward-fill any remaining NaN at the start
    df[numeric_cols] = df[numeric_cols].bfill()
    
    # Final safety: fill any remaining NaN with 0
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    
    # Verify no NaN/Inf remain
    nan_count = df[numeric_cols].isna().sum().sum()
    try:
        # Convert to float array for isinf check (handles mixed dtypes)
        numeric_values = df[numeric_cols].values.astype(np.float64)
        inf_count = np.isinf(numeric_values).sum()
    except (TypeError, ValueError):
        # If conversion fails, skip inf check
        inf_count = 0
    if nan_count > 0 or inf_count > 0:
        logger.warning(f"⚠️ Remaining NaN: {nan_count}, Inf: {inf_count} after sanitization")
    
    logger.debug(f"Computed {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'time']])} normalized features")
    
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
    threshold: float = 0.001,  # 0.1% minimum move (reduced from 0.5% to include more samples)
    locked_feature_names: Optional[List[str]] = None,
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
        locked_feature_names: If provided, use these exact features (for warm-start consistency).
            Features not found in df are zero-filled. Skips dynamic feature selection.
    
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
    # LOCKED FEATURE NAMES: Use exact features from previous model for warm-start
    # This guarantees identical feature ordering between training sessions
    # =========================================================================
    if locked_feature_names is not None and len(locked_feature_names) > 0:
        available = set(df.columns)
        n_found = sum(1 for f in locked_feature_names if f in available)
        n_missing = len(locked_feature_names) - n_found
        
        if n_found < len(locked_feature_names) // 2:
            logger.warning(f"⚠️ Locked features: only {n_found}/{len(locked_feature_names)} found in data. "
                          f"Falling back to dynamic selection.")
        else:
            if n_missing > 0:
                missing = [f for f in locked_feature_names if f not in available]
                logger.warning(f"⚠️ Locked features: {n_missing} missing features will be zero-filled: "
                              f"{missing[:5]}{'...' if n_missing > 5 else ''}")
                # Add missing columns as zeros
                for f in missing:
                    df[f] = 0.0
            
            features = list(locked_feature_names)
            logger.info(f"🔒 Using {len(features)} locked features from previous model "
                       f"({n_found} found, {n_missing} zero-filled)")
    
    # =========================================================================
    # SMART FEATURE SELECTION: Keep top ~60 uncorrelated features
    # =========================================================================
    # VARIANCE-BASED FEATURE SELECTION (no look-ahead bias)
    # Select features with high variance and remove redundant ones
    max_features = 80  # Use more features since we're not overfitting to target
    correlation_threshold = 0.80  # Remove features correlated > 80%
    
    # Skip dynamic selection if features are already locked from previous model
    features_already_locked = (locked_feature_names is not None and 
                                len(locked_feature_names) > 0 and
                                features == list(locked_feature_names))
    
    if len(features) > max_features and not features_already_locked:
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
    
    # =========================================================================
    # CLASS DISTRIBUTION VALIDATION - Prevent training with extreme imbalance
    # =========================================================================
    min_class_ratio = 0.1  # Each class must be at least 10% of clear samples
    min_clear_samples = 100  # Need at least 100 clear samples total
    
    if total_clear < min_clear_samples:
        raise ValueError(
            f"❌ INSUFFICIENT TRAINING DATA: Only {total_clear} clear samples "
            f"(need at least {min_clear_samples}). Try:\n"
            f"  1. Increase training data size (add more candles)\n"
            f"  2. Lower threshold (current: {threshold:.3%})\n"
            f"  3. Use threshold=0 to include all samples"
        )
    
    up_ratio = n_clear_up / total_clear
    down_ratio = n_clear_down / total_clear
    
    if up_ratio < min_class_ratio or down_ratio < min_class_ratio:
        minority_class = "UP" if up_ratio < down_ratio else "DOWN"
        minority_count = n_clear_up if up_ratio < down_ratio else n_clear_down
        majority_count = n_clear_down if up_ratio < down_ratio else n_clear_up
        
        raise ValueError(
            f"❌ EXTREME CLASS IMBALANCE: {minority_class} class has only {minority_count} samples "
            f"({min(up_ratio, down_ratio):.1%}) vs {majority_count} for other class.\n"
            f"This will cause the model to predict only one direction.\n"
            f"Try:\n"
            f"  1. Use more training data (different time periods may have different bias)\n"
            f"  2. Lower threshold from {threshold:.3%} to include more small moves\n"
            f"  3. Use threshold=0 for all samples (let the model learn from noise too)"
        )
    
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
# TCN VOLATILITY REGIME DATA LOADER - 4-Class Classification
# =============================================================================

def load_volatility_regime_data(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    seq_len: int = 60,
    vol_low_threshold: float = 0.25,
    vol_normal_threshold: float = 0.50,
    vol_high_threshold: float = 0.75,
) -> Dict[str, np.ndarray]:
    """
    Load data for TCN Volatility Regime model: 4-class classification.
    
    Classifies volatility regimes based on ATR percentiles:
        - 0 = LOW: ATR < 25th percentile (quiet market, tight ranges)
        - 1 = NORMAL: ATR 25th-50th percentile (typical conditions)
        - 2 = HIGH: ATR 50th-75th percentile (elevated volatility)
        - 3 = EXTREME: ATR > 75th percentile (major moves, news events)
    
    Features: NORMALIZED volatility, momentum, and risk indicators
    Target: Volatility regime class (0-3)
    
    Research-backed hyperparameters for TCN (Bai et al. / Unit8):
        - kernel_size: 5 (≥ dilation_base for no receptive field holes)
        - dilation_base: 2 (exponential dilation)
        - num_residual_blocks: 5 (for seq_len=60, receptive_field=63)
        - num_filters: 64 (increased capacity for 4-class)
        - dropout: 0.2 (lower than Transformer, TCN more stable)
        - weight_norm: True (prevents gradient explosion)
    
    Args:
        df: DataFrame with OHLCV and features
        split: (train_frac, val_frac, test_frac)
        seq_len: Sequence length for TCN input
        vol_low_threshold: ATR percentile threshold for LOW (default 0.25)
        vol_normal_threshold: ATR percentile threshold for NORMAL (default 0.50)
        vol_high_threshold: ATR percentile threshold for HIGH (default 0.75)
    
    Returns:
        Dict with X_train, y_train, X_val, y_val, X_test, y_test, feature_names, class_weights
    """
    logger.info("Loading TCN Volatility Regime data (4-class classification)...")
    
    # Compute normalized features if not already present
    if 'returns_1' not in df.columns:
        df = compute_normalized_features(df)
    
    # =========================================================================
    # FEATURE SELECTION - Volatility and Risk Indicators
    # =========================================================================
    
    # Primary normalized volatility features (instrument-agnostic)
    volatility_features = [
        # ATR-based (primary)
        'atr_pct_5', 'atr_pct_10', 'atr_pct_14', 'atr_pct_20',
        # Rolling volatility
        'volatility_5', 'volatility_10', 'volatility_20',
        # Range-based
        'tr_pct', 'hl_range_pct',
        # Candle structure
        'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio',
    ]
    
    # Momentum features (regime changes often correlate with momentum shifts)
    momentum_features = [
        'returns_1', 'returns_2', 'returns_3', 'returns_5', 'returns_10',
        'log_returns_1', 'log_returns_5',
        'roc_5', 'roc_10', 'roc_20',
    ]
    
    # Trend/Regime indicators
    regime_features = [
        'adx', 'rsi_norm', 'stoch_k_norm', 'stoch_d_norm',
        'macd_norm', 'macd_hist_momentum',
        'bb_position_20', 'bb_width_20',
        'zscore_20', 'zscore_50',
    ]
    
    # Volume features (volume often leads volatility)
    volume_features = [
        'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',
        'volume_momentum_5',
    ]
    
    # Combine all features
    all_features = volatility_features + momentum_features + regime_features + volume_features
    
    # Filter to available features
    features = _ensure_features_exist(df, all_features)
    
    # Fallback to pattern matching if not enough features
    if len(features) < 20:
        fallback_patterns = ['atr', 'volatility', 'returns', 'rsi', 'adx', 'volume']
        pattern_features = _find_features_by_pattern(df, fallback_patterns)
        for f in pattern_features:
            if f not in features and f not in ['open', 'high', 'low', 'close', 'volume', 'time']:
                features.append(f)
    
    if len(features) < 10:
        raise ValueError(f"Volatility regime needs at least 10 features, got {len(features)}")
    
    logger.info(f"Volatility regime features: {features[:10]}{'...' if len(features) > 10 else ''} ({len(features)} total)")
    
    # =========================================================================
    # TARGET CREATION - 4-Class Volatility Regime
    # =========================================================================
    
    # Calculate ATR percentage for regime classification
    atr_pct = None
    
    # Try to use pre-computed ATR percentage
    if 'atr_pct_14' in df.columns:
        atr_pct = df['atr_pct_14'].values.astype(np.float64)
        valid_mask = ~np.isnan(atr_pct) & (atr_pct > 0)
        if np.sum(valid_mask) < 100:
            atr_pct = None
    
    # Fallback: Calculate ATR percentage manually
    if atr_pct is None:
        logger.info("Computing ATR percentage manually for regime classification")
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        ])
        
        # Rolling ATR (14 periods)
        atr = np.zeros(len(close), dtype=np.float64)
        for i in range(len(close)):
            if i < 14:
                atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
            else:
                atr[i] = np.mean(tr[i-13:i+1])
        
        atr_pct = atr / np.maximum(close, 1e-8)
    
    # Calculate rolling percentile of ATR (100-bar lookback for stability)
    lookback = 100
    atr_percentile = np.zeros(len(atr_pct), dtype=np.float64)
    
    for i in range(len(atr_pct)):
        if i < lookback:
            window = atr_pct[:i+1]
        else:
            window = atr_pct[i-lookback+1:i+1]
        
        # Percentile rank of current ATR within window
        atr_percentile[i] = np.sum(window <= atr_pct[i]) / len(window)
    
    # Classify into 4 regimes based on ATR percentile
    # 0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME
    regime_labels = np.zeros(len(atr_percentile), dtype=np.int32)
    regime_labels[atr_percentile < vol_low_threshold] = 0      # LOW
    regime_labels[(atr_percentile >= vol_low_threshold) & (atr_percentile < vol_normal_threshold)] = 1  # NORMAL
    regime_labels[(atr_percentile >= vol_normal_threshold) & (atr_percentile < vol_high_threshold)] = 2  # HIGH
    regime_labels[atr_percentile >= vol_high_threshold] = 3    # EXTREME
    
    # Log class distribution
    unique, counts = np.unique(regime_labels, return_counts=True)
    class_dist = dict(zip(unique, counts))
    regime_names = ['LOW', 'NORMAL', 'HIGH', 'EXTREME']
    logger.info(f"Volatility regime distribution: {', '.join([f'{regime_names[k]}={v}' for k, v in class_dist.items()])}")
    
    # Calculate label stats for summary display
    total = len(regime_labels)
    label_stats = {
        'LOW': class_dist.get(0, 0) / total if total > 0 else 0,
        'NORMAL': class_dist.get(1, 0) / total if total > 0 else 0,
        'HIGH': class_dist.get(2, 0) / total if total > 0 else 0,
        'EXTREME': class_dist.get(3, 0) / total if total > 0 else 0,
    }
    
    # =========================================================================
    # SEQUENCE CREATION
    # =========================================================================
    
    # Extract feature matrix
    X = df[features].values.astype(np.float32)
    y = regime_labels
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create sequences for TCN
    def create_sequences(X, y, seq_len):
        n_samples = len(X) - seq_len
        X_seq = np.zeros((n_samples, seq_len, X.shape[1]), dtype=np.float32)
        y_seq = np.zeros(n_samples, dtype=np.int32)
        
        for i in range(n_samples):
            X_seq[i] = X[i:i+seq_len]
            y_seq[i] = y[i+seq_len-1]  # Label at end of sequence
        
        return X_seq, y_seq
    
    X_seq, y_seq = create_sequences(X, y, seq_len)
    
    logger.info(f"Created {len(X_seq)} sequences of length {seq_len}")
    
    # =========================================================================
    # TEMPORAL SPLIT
    # =========================================================================
    
    train_frac, val_frac, test_frac = split
    n = len(X_seq)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    X_train = X_seq[:train_end]
    y_train = y_seq[:train_end]
    X_val = X_seq[train_end:val_end]
    y_val = y_seq[train_end:val_end]
    X_test = X_seq[val_end:]
    y_test = y_seq[val_end:]
    
    # =========================================================================
    # CLASS WEIGHTS (for imbalanced classes)
    # =========================================================================
    
    # Calculate class weights inversely proportional to frequency
    class_counts = np.bincount(y_train, minlength=4)
    total_samples = len(y_train)
    
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    
    # Inverse frequency weighting (higher weight for minority classes)
    class_weights = total_samples / (4 * class_counts)
    class_weights = class_weights / class_weights.sum() * 4  # Normalize to sum to 4
    
    # Convert to dict format for Keras
    class_weight_dict = {i: float(class_weights[i]) for i in range(4)}
    
    logger.info(f"Class weights: {class_weight_dict}")
    
    # Calculate sample weights for training
    sample_weights = np.array([class_weights[y] for y in y_train], dtype=np.float32)
    
    result = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': features,
        'class_weights': class_weight_dict,
        'sample_weights': sample_weights,
        'seq_len': seq_len,
        'n_classes': 4,
        'class_names': regime_names,
        'label_stats': label_stats,
    }
    
    logger.info(f"Volatility regime data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}, "
                f"features={len(features)}, seq_len={seq_len}")
    
    return result


# =============================================================================
# FORWARD-LOOKING VOLATILITY DATA LOADER - Predict Future Regime
# =============================================================================

def load_forward_volatility_data(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    seq_len: int = 60,
    lookahead: int = 48,  # 48 bars = 2 days for H1 (weekly scan workflow)
    quiet_threshold: float = 0.25,    # QUIET_NEXT: <25th percentile
    active_threshold: float = 0.60,   # ACTIVE_NEXT: 60th-85th (emphasize opportunity)
    extreme_threshold: float = 0.85,  # EXTREME_NEXT: >85th percentile
    min_change_for_clear: float = 0.05,  # 5% vol change minimum for clear label
) -> Dict[str, np.ndarray]:
    """
    Load data for TCN Forward Volatility model: 4-class PREDICTION.
    
    FORWARD-LOOKING: Predicts what volatility regime will exist in `lookahead` bars.
    Unlike load_volatility_regime_data() which classifies CURRENT volatility,
    this function creates labels based on FUTURE ATR percentile.
    
    Classes:
        - 0 = QUIET_NEXT: Future ATR < 25th percentile (skip - low vol expected)
        - 1 = STABLE_NEXT: Future ATR 25th-60th percentile (normal trading)
        - 2 = ACTIVE_NEXT: Future ATR 60th-85th percentile (opportunity!)
        - 3 = EXTREME_NEXT: Future ATR > 85th percentile (caution - reduce size)
    
    Also returns REGRESSION TARGET (% change in volatility) for dual-head model.
    
    Key differences from load_volatility_regime_data():
    1. Uses FUTURE ATR percentile (lookahead bars ahead), not current
    2. Asymmetric thresholds (25/60/85) emphasize ACTIVE_NEXT class
    3. Confidence-based sample weights (large vol changes = high weight)
    4. Returns both classification labels AND regression targets
    5. Includes time features (hour, dow) that predict volatility patterns
    
    Args:
        df: DataFrame with OHLCV and features
        split: (train_frac, val_frac, test_frac)
        seq_len: Sequence length for TCN input
        lookahead: How many bars ahead to predict (48 = 2 days for H1)
        quiet_threshold: Percentile threshold for QUIET_NEXT (default 0.25)
        active_threshold: Percentile threshold for ACTIVE_NEXT (default 0.60)
        extreme_threshold: Percentile threshold for EXTREME_NEXT (default 0.85)
        min_change_for_clear: Minimum vol change to count as "clear" label
    
    Returns:
        Dict with X_train, y_train, w_train (weights), regression targets, etc.
    """
    logger.info(f"Loading FORWARD Volatility data (lookahead={lookahead}, thresholds={quiet_threshold}/{active_threshold}/{extreme_threshold})...")
    
    # Compute normalized features if not already present
    if 'returns_1' not in df.columns:
        df = compute_normalized_features(df)
    
    # =========================================================================
    # FEATURE SELECTION - Volatility predictors (NOT including atr_pct_14 directly)
    # =========================================================================
    
    # Features that PREDICT future volatility (not just measure current)
    volatility_predictors = [
        # Lagged volatility (volatility is autocorrelated)
        'atr_pct_5', 'atr_pct_10', 'atr_pct_20',  # Shorter/longer ATR ratios
        'volatility_5', 'volatility_10', 'volatility_20',
        # Range-based (leading indicators)
        'tr_pct', 'hl_range_pct',
        # Candle structure (body/wick ratios predict future moves)
        'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio',
    ]
    
    # Volume features (volume LEADS volatility - key predictor)
    volume_predictors = [
        'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',
        'volume_zscore', 'volume_ma_ratio',
    ]
    
    # Momentum features (momentum shifts often precede vol changes)
    momentum_features = [
        'returns_1', 'returns_2', 'returns_5', 'returns_10',
        'log_returns_1',
        'rsi_norm', 'stoch_k_norm',
        'macd_norm', 'macd_hist_norm',
    ]
    
    # Time features (session/day patterns predict volatility)
    time_features = [
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'session_london', 'session_ny', 'session_overlap',
    ]
    
    # Trend/Regime context
    context_features = [
        'zscore_20', 'zscore_50',
        'sma_ratio_20', 'ema_ratio_26',
        'pct_rank_20',
    ]
    
    # Combine all features
    all_features = volatility_predictors + volume_predictors + momentum_features + time_features + context_features
    
    # Filter to available features
    features = _ensure_features_exist(df, all_features)
    
    # Fallback to pattern matching if not enough features
    if len(features) < 15:
        fallback_patterns = ['atr', 'volatility', 'volume', 'returns', 'hour', 'dow', 'session']
        pattern_features = _find_features_by_pattern(df, fallback_patterns)
        for f in pattern_features:
            if f not in features and f not in ['open', 'high', 'low', 'close', 'volume', 'time', 'atr_pct_14']:
                features.append(f)
    
    # NOTE: We intentionally EXCLUDE atr_pct_14 from features if it's the primary
    # label source, to avoid circular feature-label dependency
    # (but include shorter/longer ATR periods as predictors)
    
    if len(features) < 10:
        raise ValueError(f"Forward volatility needs at least 10 features, got {len(features)}")
    
    logger.info(f"Forward volatility features: {features[:10]}{'...' if len(features) > 10 else ''} ({len(features)} total)")
    
    # =========================================================================
    # TARGET CREATION - FORWARD-LOOKING 4-Class + Regression
    # =========================================================================
    
    # Get ATR percentage for regime classification
    if 'atr_pct_14' in df.columns:
        atr_pct = df['atr_pct_14'].values.astype(np.float64)
    else:
        # Compute ATR percentage manually
        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        ])
        
        atr = np.zeros(len(close), dtype=np.float64)
        for i in range(len(close)):
            if i < 14:
                atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
            else:
                atr[i] = np.mean(tr[i-13:i+1])
        
        atr_pct = atr / np.maximum(close, 1e-8)
    
    n = len(atr_pct)
    
    # Calculate rolling percentile of ATR (100-bar lookback)
    lookback = 100
    atr_percentile = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if i < lookback:
            window = atr_pct[:i+1]
        else:
            window = atr_pct[i-lookback+1:i+1]
        atr_percentile[i] = np.sum(window <= atr_pct[i]) / len(window)
    
    # FORWARD-LOOKING: Get FUTURE ATR percentile (lookahead bars ahead)
    future_atr_percentile = np.roll(atr_percentile, -lookahead)
    future_atr_pct = np.roll(atr_pct, -lookahead)
    
    # Mark last lookahead rows as invalid (no future data)
    future_atr_percentile[-lookahead:] = np.nan
    future_atr_pct[-lookahead:] = np.nan
    
    # Calculate % change in volatility (regression target)
    vol_change_pct = np.zeros(n, dtype=np.float64)
    for i in range(n - lookahead):
        current_vol = atr_pct[i]
        future_vol = future_atr_pct[i]
        if current_vol > 1e-10 and not np.isnan(future_vol):
            vol_change_pct[i] = (future_vol - current_vol) / current_vol
        else:
            vol_change_pct[i] = 0.0
    vol_change_pct[-lookahead:] = np.nan
    
    # Clip extreme regression targets
    vol_change_pct = np.clip(vol_change_pct, -2.0, 2.0)
    
    # Create classification labels based on FUTURE percentile
    regime_labels = np.zeros(n, dtype=np.int32)
    sample_weights = np.ones(n, dtype=np.float32)
    
    class_names = ['QUIET_NEXT', 'STABLE_NEXT', 'ACTIVE_NEXT', 'EXTREME_NEXT']
    
    for i in range(n - lookahead):
        future_pct = future_atr_percentile[i]
        change = vol_change_pct[i]
        
        if np.isnan(future_pct):
            sample_weights[i] = 0.0
            continue
        
        # Classify by FUTURE percentile
        if future_pct < quiet_threshold:
            regime_labels[i] = 0  # QUIET_NEXT
        elif future_pct < active_threshold:
            regime_labels[i] = 1  # STABLE_NEXT
        elif future_pct < extreme_threshold:
            regime_labels[i] = 2  # ACTIVE_NEXT
        else:
            regime_labels[i] = 3  # EXTREME_NEXT
        
        # Confidence-based weighting: more weight on clear regime changes
        abs_change = np.abs(change)
        if abs_change < min_change_for_clear:
            # Small change = low confidence, but still include
            sample_weights[i] = 0.5
        elif abs_change > 0.30:
            # Large change = high confidence
            sample_weights[i] = 1.0
        else:
            # Scale with change size
            sample_weights[i] = 0.5 + abs_change
    
    # Zero weight for last lookahead rows (no future data)
    sample_weights[-lookahead:] = 0.0
    regime_labels[-lookahead:] = 1  # Default to STABLE (doesn't matter, weight=0)
    
    # Log class distribution (before sequencing)
    valid_mask = sample_weights > 0
    valid_labels = regime_labels[valid_mask]
    if len(valid_labels) > 0:
        unique, counts = np.unique(valid_labels, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Forward volatility distribution: {', '.join([f'{class_names[k]}={v}' for k, v in class_dist.items()])}")
    
    # =========================================================================
    # FEATURE SCALING (fit on training portion only)
    # =========================================================================
    
    from sklearn.preprocessing import RobustScaler
    
    # Extract feature matrix
    X = df[features].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Temporal split indices (before sequencing)
    train_frac, val_frac, test_frac = split
    n_valid = n - lookahead
    train_end = int(n_valid * train_frac)
    val_end = int(n_valid * (train_frac + val_frac))
    
    # Fit scaler on training data only
    scaler = RobustScaler()
    X_train_flat = X[:train_end]
    scaler.fit(X_train_flat)
    X_scaled = scaler.transform(X)
    X_scaled = np.clip(X_scaled, -10, 10)  # Clip extreme values
    
    # =========================================================================
    # SEQUENCE CREATION
    # =========================================================================
    
    def create_sequences_with_weights(X, y_class, y_reg, weights, seq_len):
        """Create sequences preserving per-sample weights and regression targets."""
        n_samples = len(X) - seq_len
        X_seq = np.zeros((n_samples, seq_len, X.shape[1]), dtype=np.float32)
        y_class_seq = np.zeros(n_samples, dtype=np.int32)
        y_reg_seq = np.zeros(n_samples, dtype=np.float32)
        w_seq = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(n_samples):
            X_seq[i] = X[i:i+seq_len]
            y_class_seq[i] = y_class[i+seq_len-1]  # Label at end of sequence
            y_reg_seq[i] = y_reg[i+seq_len-1]
            w_seq[i] = weights[i+seq_len-1]
        
        return X_seq, y_class_seq, y_reg_seq, w_seq
    
    X_seq, y_class_seq, y_reg_seq, w_seq = create_sequences_with_weights(
        X_scaled, regime_labels, vol_change_pct, sample_weights, seq_len
    )
    
    logger.info(f"Created {len(X_seq)} sequences of length {seq_len}")
    
    # =========================================================================
    # TEMPORAL SPLIT (on sequences)
    # =========================================================================
    
    n_seq = len(X_seq)
    train_end_seq = int(n_seq * train_frac)
    val_end_seq = int(n_seq * (train_frac + val_frac))
    
    X_train = X_seq[:train_end_seq]
    y_train_class = y_class_seq[:train_end_seq]
    y_train_reg = y_reg_seq[:train_end_seq]
    w_train = w_seq[:train_end_seq]
    
    X_val = X_seq[train_end_seq:val_end_seq]
    y_val_class = y_class_seq[train_end_seq:val_end_seq]
    y_val_reg = y_reg_seq[train_end_seq:val_end_seq]
    w_val = w_seq[train_end_seq:val_end_seq]
    
    X_test = X_seq[val_end_seq:]
    y_test_class = y_class_seq[val_end_seq:]
    y_test_reg = y_reg_seq[val_end_seq:]
    w_test = w_seq[val_end_seq:]
    
    # =========================================================================
    # CLASS WEIGHTS (inverse frequency, computed on training set only)
    # =========================================================================
    
    # Only count samples with weight > 0
    train_valid_mask = w_train > 0
    valid_train_labels = y_train_class[train_valid_mask]
    
    if len(valid_train_labels) > 0:
        class_counts = np.bincount(valid_train_labels, minlength=4)
        total_samples = len(valid_train_labels)
        
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        
        # Inverse frequency weighting
        class_weights = total_samples / (4 * class_counts)
        class_weights = class_weights / class_weights.sum() * 4  # Normalize
        
        class_weight_dict = {i: float(class_weights[i]) for i in range(4)}
    else:
        class_weight_dict = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    
    logger.info(f"Class weights: {class_weight_dict}")
    
    # Calculate label stats
    if len(valid_train_labels) > 0:
        total = len(valid_train_labels)
        label_stats = {
            'QUIET_NEXT': np.sum(valid_train_labels == 0) / total,
            'STABLE_NEXT': np.sum(valid_train_labels == 1) / total,
            'ACTIVE_NEXT': np.sum(valid_train_labels == 2) / total,
            'EXTREME_NEXT': np.sum(valid_train_labels == 3) / total,
            'lookahead': lookahead,
            'min_change_for_clear': min_change_for_clear,
        }
    else:
        label_stats = {}
    
    result = {
        # Classification data
        'X_train': X_train,
        'y_train': y_train_class,
        'w_train': w_train,
        'X_val': X_val,
        'y_val': y_val_class,
        'w_val': w_val,
        'X_test': X_test,
        'y_test': y_test_class,
        'w_test': w_test,
        
        # Regression data (dual-head)
        'y_train_reg': y_train_reg,
        'y_val_reg': y_val_reg,
        'y_test_reg': y_test_reg,
        
        # Metadata
        'feature_names': features,
        'class_weights': class_weight_dict,
        'seq_len': seq_len,
        'lookahead': lookahead,
        'n_classes': 4,
        'class_names': class_names,
        'label_stats': label_stats,
        'scaler': scaler,
        
        # Regression mapping thresholds (for inference fallback)
        'reg_thresholds': {
            'quiet': -0.15,      # <-15% → QUIET
            'stable_low': -0.15,
            'stable_high': 0.15, # -15% to +15% → STABLE
            'active_high': 0.40, # +15% to +40% → ACTIVE
            # >+40% → EXTREME
        },
    }
    
    logger.info(f"Forward volatility data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}, "
                f"features={len(features)}, seq_len={seq_len}, lookahead={lookahead}")
    
    return result


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
    
    # Drop initial rows without valid momentum BEFORE splitting
    # This ensures consistent indexing for train/val/test
    valid_start = momentum_window + 5
    X = X[valid_start:]
    raw_momentum_valid = raw_momentum_all[valid_start:]
    
    # Verify arrays have same length for consistent indexing
    assert len(X) == len(raw_momentum_valid), \
        f"Array length mismatch: X={len(X)}, raw_momentum_valid={len(raw_momentum_valid)}"
    
    # Handle NaN/Inf in features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Temporal split FIRST - before computing normalization factors
    # This prevents data leakage from val/test into training normalization
    train_idx, val_idx, test_idx = temporal_split(len(X), *split)
    
    # STABLE NORMALIZATION: Compute percentiles from TRAINING DATA ONLY
    # This prevents data leakage - val/test distributions not seen during training
    train_raw_momentum = raw_momentum_valid[train_idx]
    train_raw_valid = train_raw_momentum[train_raw_momentum > 0]
    
    if len(train_raw_valid) > 0:
        p50_momentum = np.percentile(train_raw_valid, 50)  # Median momentum (train only)
        p90_momentum = np.percentile(train_raw_valid, 90)  # High momentum (train only)
        # Scale so median maps to 0.3, P90 maps to 0.7
        # This ensures threshold of 0.15 catches ~30% of bars
        norm_factor = p50_momentum / 0.3 if p50_momentum > 0 else 0.001
    else:
        p50_momentum = 0.0
        p90_momentum = 0.0
        norm_factor = 0.001
    
    logger.info(f"XGBoost momentum (train-only): P50={p50_momentum:.6f}, P90={p90_momentum:.6f}, norm_factor={norm_factor:.6f}")
    
    # Normalize raw momentum to 0-1 scale using training-derived factor
    # Apply same normalization to all splits
    momentum_score = np.minimum(raw_momentum_valid / norm_factor, 1.0)
    
    # Acceleration: Is momentum growing? Compare current vs previous momentum
    # Note: We already dropped momentum_window rows, so we have momentum for all remaining
    acceleration = np.zeros(len(momentum_score), dtype=np.float32)
    for i in range(5, len(momentum_score)):
        current_mom = momentum_score[i]
        prev_mom = momentum_score[i - 5]
        acceleration[i] = 1.0 if current_mom > prev_mom else 0.0
    
    # Combine targets: [momentum_score, acceleration]
    y = np.column_stack([momentum_score, acceleration]).astype(np.float32)
    
    # Handle NaN/Inf in targets
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
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
    
    # Calculate risk targets based on ACTUAL FORWARD DRAWDOWN (no target leakage)
    # Previous approach used atr_pct * 2 which is trivially recoverable from input features.
    # Now we compute actual max adverse excursion over the next `drawdown_horizon` bars.
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    n = len(close)
    drawdown_horizon = 24  # Look ahead 24 bars (1 day for H1)
    
    # Compute actual forward max drawdown for each bar
    # For LONG: max drawdown = (entry - min_low_ahead) / entry  
    # For SHORT: max drawdown = (max_high_ahead - entry) / entry
    # Use the WORST case (max of both directions)
    expected_drawdown_pct = np.zeros(n, dtype=np.float32)
    for i in range(n - drawdown_horizon):
        entry = close[i]
        if entry <= 0:
            continue
        future_lows = low[i+1:i+1+drawdown_horizon]
        future_highs = high[i+1:i+1+drawdown_horizon]
        
        # Max adverse for long position
        long_drawdown = (entry - np.min(future_lows)) / entry
        # Max adverse for short position
        short_drawdown = (np.max(future_highs) - entry) / entry
        # Take the worst case
        expected_drawdown_pct[i] = max(long_drawdown, short_drawdown)
    
    # Fill last `drawdown_horizon` bars with rolling mean
    if n > drawdown_horizon + 100:
        fill_val = np.mean(expected_drawdown_pct[n-drawdown_horizon-100:n-drawdown_horizon])
    else:
        fill_val = np.mean(expected_drawdown_pct[:max(1, n-drawdown_horizon)])
    expected_drawdown_pct[n-drawdown_horizon:] = fill_val
    
    # Clip to realistic range
    expected_drawdown_pct = np.clip(expected_drawdown_pct, 0.0001, 0.10).astype(np.float32)
    
    logger.info(f"RF: Forward drawdown range [{np.min(expected_drawdown_pct):.4f}, "
                f"{np.max(expected_drawdown_pct):.4f}], mean={np.mean(expected_drawdown_pct):.4f}")
    
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
    
    # Drop initial rows without valid confidence BEFORE splitting
    valid_start = confidence_window
    X = X[valid_start:]
    adx_valid_range = adx[valid_start:]
    rsi_valid_range = rsi[valid_start:]
    atr_pct_valid_range = atr_pct[valid_start:]
    bb_pos_valid_range = bb_pos[valid_start:]
    volume_ratio_valid_range = volume_ratio[valid_start:]
    
    # Verify arrays have same length for consistent indexing
    assert len(X) == len(adx_valid_range) == len(rsi_valid_range), \
        f"Array length mismatch: X={len(X)}, adx={len(adx_valid_range)}, rsi={len(rsi_valid_range)}"
    
    # Handle NaN/Inf in features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Temporal split FIRST - before computing normalization factors
    # This prevents data leakage from val/test into training normalization
    train_idx, val_idx, test_idx = temporal_split(len(X), *split)
    
    # Compute ADX percentile thresholds from TRAINING DATA ONLY for better scaling
    # This prevents data leakage - val/test distributions not seen during training
    train_adx = adx_valid_range[train_idx]
    train_adx_valid = train_adx[~np.isnan(train_adx)]
    adx_p25 = np.percentile(train_adx_valid, 25) if len(train_adx_valid) > 0 else 15
    adx_p75 = np.percentile(train_adx_valid, 75) if len(train_adx_valid) > 0 else 30
    adx_range = max(adx_p75 - adx_p25, 5)  # Avoid division by zero
    
    logger.info(f"Ridge ADX (train-only): P25={adx_p25:.2f}, P75={adx_p75:.2f}, range={adx_range:.2f}")
    
    # Calculate confidence scores using training-derived percentiles
    # Apply same normalization to all data (train/val/test)
    confidence = np.zeros(len(X), dtype=np.float32)
    
    for i in range(len(X)):
        # ===== ADX component: Strong trend = high confidence =====
        # Use percentile-based scaling for the instrument's actual ADX range (train-derived)
        # Maps ADX to [0, 1] where p25->0.25, p75->0.75, p90+->1.0
        adx_normalized = (adx_valid_range[i] - adx_p25) / adx_range
        adx_score = np.clip(adx_normalized * 0.5 + 0.25, 0.0, 1.0)
        
        # ===== RSI component: Not extreme = high confidence =====
        # RSI 40-60: high confidence (centered), 30-70: medium, outside: low
        rsi_distance = abs(rsi_valid_range[i] - 50)
        rsi_score = max(0, 1.0 - rsi_distance / 25)  # 50->1, 25/75->0
        
        # ===== Volatility component: Low vol = high confidence =====
        # Use instrument-relative scaling (ATR% typically 0.3%-1.5% for FX)
        vol_score = np.clip(1.0 - atr_pct_valid_range[i] / 0.015, 0.0, 1.0)
        
        # ===== Bollinger Band position: Middle = high confidence =====
        # BB position 0.3-0.7: confident middle, extremes: overbought/oversold
        bb_distance = abs(bb_pos_valid_range[i] - 0.5)
        bb_score = max(0, 1.0 - bb_distance * 2.5)  # 0.5->1, 0.1/0.9->0
        
        # ===== Volume confirmation: Above average = high confidence =====
        # Volume ratio > 1.0: good conviction, < 0.7: low conviction
        vol_conf_score = np.clip((volume_ratio_valid_range[i] - 0.7) / 0.6, 0.0, 1.0)
        
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
    
    # Handle NaN/Inf in targets
    y = np.nan_to_num(y, nan=50.0, posinf=100.0, neginf=0.0)
    
    result = {
        'X_train': X[train_idx],
        'y_train': y[train_idx],
        'X_val': X[val_idx],
        'y_val': y[val_idx],
        'X_test': X[test_idx],
        'y_test': y[test_idx],
        'feature_names': features,
        'adx_p25': float(adx_p25),  # Save for inference
        'adx_p75': float(adx_p75),  # Save for inference
    }
    
    logger.info(f"Ridge data: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, features={len(features)}")
    return result


# =============================================================================
# DATA LEAKAGE VALIDATION
# =============================================================================

def validate_no_leakage(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    max_correlation: float = 0.99,
    target_std_tolerance: float = 0.01,
) -> Dict[str, bool]:
    """
    Validate that there is no data leakage across train/val/test splits.
    
    This function checks for common signs of data leakage:
    1. Feature means too similar across splits (suggests global normalization)
    2. Feature-target correlations identical across splits (suggests shared computation)
    3. Target distributions suspiciously similar
    
    Args:
        X_train, X_val, X_test: Feature matrices
        y_train, y_val, y_test: Target arrays
        max_correlation: Maximum allowed correlation between split means (default 0.99)
        target_std_tolerance: Tolerance for target std comparison (default 0.01 = 1%)
    
    Returns:
        Dict with validation results:
            - 'passed': Overall pass/fail
            - 'feature_means_ok': Feature means are sufficiently different
            - 'target_distributions_ok': Target distributions are different
            - 'warnings': List of any warnings
    """
    warnings = []
    
    # Check 1: Feature means should differ across splits
    train_mean = X_train.mean(axis=0)
    val_mean = X_val.mean(axis=0)
    test_mean = X_test.mean(axis=0)
    
    # Handle edge cases for correlation calculation
    # - Single feature: set correlation to 0 (can't compute meaningfully)
    # - Zero variance: corrcoef may produce NaN/inf
    train_val_corr = 0.0
    train_test_corr = 0.0
    
    if len(train_mean) > 1:
        # Check for zero variance which would cause NaN in corrcoef
        train_std = np.std(train_mean)
        val_std = np.std(val_mean)
        test_std = np.std(test_mean)
        
        if train_std > 1e-10 and val_std > 1e-10:
            try:
                train_val_corr = np.corrcoef(train_mean, val_mean)[0, 1]
            except (FloatingPointError, RuntimeWarning):
                train_val_corr = 0.0
        
        if train_std > 1e-10 and test_std > 1e-10:
            try:
                train_test_corr = np.corrcoef(train_mean, test_mean)[0, 1]
            except (FloatingPointError, RuntimeWarning):
                train_test_corr = 0.0
    
    # Handle NaN correlations (can still occur in edge cases)
    train_val_corr = 0.0 if np.isnan(train_val_corr) or np.isinf(train_val_corr) else train_val_corr
    train_test_corr = 0.0 if np.isnan(train_test_corr) or np.isinf(train_test_corr) else train_test_corr
    
    feature_means_ok = train_val_corr < max_correlation and train_test_corr < max_correlation
    
    if not feature_means_ok:
        warnings.append(f"Feature means too similar: train-val corr={train_val_corr:.4f}, train-test corr={train_test_corr:.4f}")
        logger.warning(f"⚠️ Potential data leakage: {warnings[-1]}")
    
    # Check 2: Target distributions should differ across temporal splits
    y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
    y_val_flat = y_val.flatten() if len(y_val.shape) > 1 else y_val
    y_test_flat = y_test.flatten() if len(y_test.shape) > 1 else y_test
    
    train_target_std = np.std(y_train_flat)
    val_target_std = np.std(y_val_flat)
    test_target_std = np.std(y_test_flat)
    
    # For temporal data, we expect some drift in target statistics
    # If all stds are identical (within tolerance), it might indicate shared computation
    std_ratio_val = val_target_std / max(train_target_std, 1e-8)
    std_ratio_test = test_target_std / max(train_target_std, 1e-8)
    
    lower_bound = 1.0 - target_std_tolerance
    upper_bound = 1.0 + target_std_tolerance
    target_distributions_ok = not (lower_bound < std_ratio_val < upper_bound and lower_bound < std_ratio_test < upper_bound)
    
    if not target_distributions_ok:
        warnings.append(f"Target std identical across splits: train={train_target_std:.6f}, val={val_target_std:.6f}, test={test_target_std:.6f}")
        logger.warning(f"⚠️ Possible target computation issue: {warnings[-1]}")
    
    passed = feature_means_ok and target_distributions_ok
    
    if passed:
        logger.info("✅ No data leakage detected")
    else:
        logger.warning("❌ Potential data leakage detected - review data preparation")
    
    return {
        'passed': passed,
        'feature_means_ok': feature_means_ok,
        'target_distributions_ok': target_distributions_ok,
        'train_val_feature_corr': float(train_val_corr),
        'train_test_feature_corr': float(train_test_corr),
        'warnings': warnings,
    }


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
    locked_feature_names: Optional[List[str]] = None,
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
        locked_feature_names: If provided, use these exact direction features (for warm-start consistency)
    
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
    
    # SECOND: Apply FeatureEngineering to add advanced features
    # This ensures training uses the same features as inference
    try:
        from src.data.feature_engineering import FeatureEngineering
        fe = FeatureEngineering({})
        # Don't apply candle smoothing (already done by compute_normalized_features if needed)
        df_fe = fe.create_features(df.copy(), include_all=True, apply_candle_smoothing=False)
        
        # Merge features from FeatureEngineering that aren't already present
        # Use index-based join to handle potential row count differences
        new_cols = [c for c in df_fe.columns if c not in df_normalized.columns]
        if new_cols:
            df_fe_aligned = df_fe[new_cols].reindex(df_normalized.index)
            df_normalized = pd.concat([df_normalized, df_fe_aligned], axis=1)
            logger.info(f"Added {len(new_cols)} features from FeatureEngineering, total={len(df_normalized.columns)}")
    except Exception as e:
        logger.warning(f"FeatureEngineering failed (features may be incomplete): {e}")
    
    # Clean any NaN/inf from merged features
    df_normalized = df_normalized.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    
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
        direction_data = load_direction_data(df_normalized, split, direction_lookahead, direction_threshold,
                                              locked_feature_names=locked_feature_names)
        result['direction'] = direction_data
        result['tcn'] = direction_data  # Alias for backward compat
    
    return result

