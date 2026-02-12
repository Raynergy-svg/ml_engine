"""
Utility functions for RL modules.

Provides shared functionality for feature normalization, regime detection,
and performance calculations used across RL environments.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_regime(
    features: np.ndarray,
    adx_idx: int = 0,
    volatility_idx: int = 1,
    momentum_idx: int = 2,
) -> Tuple[np.ndarray, str]:
    """
    Detect market regime from features.

    Regimes:
        - TREND (0): ADX > 25, clear directional momentum
        - CHOP (1): ADX < 20, low volatility, no clear direction
        - MEAN_REVERT (2): High volatility, momentum exhaustion signals

    Args:
        features: Feature array for current timestep
        adx_idx: Index of ADX feature in array
        volatility_idx: Index of volatility feature
        momentum_idx: Index of momentum feature

    Returns:
        Tuple of (one-hot encoding [3], regime name)
    """
    if len(features) < max(adx_idx, volatility_idx, momentum_idx) + 1:
        # Not enough features, default to TREND
        return np.array([1.0, 0.0, 0.0], dtype=np.float32), 'TREND'

    adx = features[adx_idx] if adx_idx < len(features) else 25.0
    volatility = features[volatility_idx] if volatility_idx < len(features) else 0.5
    momentum = abs(features[momentum_idx]) if momentum_idx < len(features) else 0.3

    # Classify regime
    if adx > 25 and momentum > 0.2:
        regime = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # TREND
        name = 'TREND'
    elif adx < 20 and volatility < 0.3:
        regime = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # CHOP
        name = 'CHOP'
    else:
        regime = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # MEAN_REVERT
        name = 'MEAN_REVERT'

    return regime, name


def calculate_sharpe(
    returns: List[float],
    risk_free_rate: float = 0.0,
    annualization_factor: float = np.sqrt(252 * 24),  # Hourly data
) -> float:
    """
    Calculate Sharpe ratio from a list of returns.

    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate (default 0)
        annualization_factor: Factor for annualizing (sqrt(periods per year))

    Returns:
        Sharpe ratio (0 if insufficient data or zero volatility)
    """
    if len(returns) < 2:
        return 0.0

    returns_arr = np.array(returns)
    mean_return = np.mean(returns_arr) - risk_free_rate
    std_return = np.std(returns_arr, ddof=1)

    if std_return < 1e-8:
        return 0.0

    return (mean_return / std_return) * annualization_factor


def calculate_sortino(
    returns: List[float],
    risk_free_rate: float = 0.0,
    annualization_factor: float = np.sqrt(252 * 24),
) -> float:
    """
    Calculate Sortino ratio (downside-risk adjusted returns).

    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate (default 0)
        annualization_factor: Factor for annualizing

    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    returns_arr = np.array(returns)
    excess_returns = returns_arr - risk_free_rate
    mean_excess = np.mean(excess_returns)

    # Downside deviation (only negative returns)
    negative_returns = returns_arr[returns_arr < 0]
    if len(negative_returns) < 2:
        return calculate_sharpe(returns, risk_free_rate, annualization_factor)

    downside_std = np.std(negative_returns, ddof=1)

    if downside_std < 1e-8:
        return 0.0

    return (mean_excess / downside_std) * annualization_factor


def calculate_calmar(
    returns: List[float],
    max_drawdown: float,
    annualization_factor: float = 252 * 24,  # Hourly periods per year
) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).

    Args:
        returns: List of period returns
        max_drawdown: Maximum drawdown (positive value, e.g., 0.10 for 10%)
        annualization_factor: Periods per year for annualization

    Returns:
        Calmar ratio
    """
    if len(returns) < 1 or max_drawdown < 1e-8:
        return 0.0

    total_return = np.sum(returns)
    annualized_return = total_return * (annualization_factor / len(returns))

    return annualized_return / max_drawdown


def normalize_features_for_rl(
    features: np.ndarray,
    scaler: Optional[object] = None,
) -> np.ndarray:
    """
    Normalize features for RL environment observation space.

    Uses StandardScaler if provided, otherwise clips to [-3, 3] range
    assuming features are already somewhat normalized.

    Args:
        features: Raw feature array
        scaler: Optional sklearn StandardScaler

    Returns:
        Normalized feature array
    """
    if scaler is not None:
        try:
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            normalized = scaler.transform(features).flatten()
        except Exception:
            normalized = np.clip(features.flatten(), -3, 3)
    else:
        # Simple clip normalization
        normalized = np.clip(features.flatten(), -3, 3)

    # Replace NaN/Inf with 0
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=3.0, neginf=-3.0)

    return normalized.astype(np.float32)


def extract_gate_features(
    df: pd.DataFrame,
    lookback: int = 20,
) -> Dict[str, float]:
    """
    Extract features relevant to gate threshold decisions.

    Args:
        df: DataFrame with market data
        lookback: Number of bars for rolling calculations

    Returns:
        Dictionary of gate-relevant features
    """
    features: Dict[str, float] = {}

    # ADX-based confidence proxy
    if 'adx' in df.columns:
        features['adx'] = float(df['adx'].iloc[-1])
        features['adx_trend'] = float(df['adx'].diff().iloc[-lookback:].mean())

    # Momentum features
    if 'close' in df.columns:
        returns = df['close'].pct_change()
        features['momentum'] = float(returns.iloc[-lookback:].sum())
        features['momentum_acceleration'] = float(returns.diff().iloc[-5:].sum())

    # Volatility features
    if 'atr' in df.columns:
        features['atr'] = float(df['atr'].iloc[-1])
        features['atr_percentile'] = float(
            (df['atr'].iloc[-1] - df['atr'].iloc[-100:].min()) /
            (df['atr'].iloc[-100:].max() - df['atr'].iloc[-100:].min() + 1e-8)
        )

    # Recent max drawdown from close prices over the lookback window
    if 'close' in df.columns and len(df) >= lookback:
        recent_close = df['close'].iloc[-lookback:]
        # Compute running maximum and drawdown from peak
        running_max = recent_close.cummax()
        drawdowns = (running_max - recent_close) / running_max.replace(0, np.nan)
        # Max drawdown as a positive fraction (e.g., 0.05 = 5% drawdown)
        drawdowns = drawdowns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        features['recent_drawdown'] = float(drawdowns.max())
    else:
        # Not enough data; default to 0.0 (no observed drawdown)
        features['recent_drawdown'] = 0.0

    return features


def compute_rl_features(
    df: pd.DataFrame,
    lookback: int = 20,
    *,
    as_array: bool = True,
) -> np.ndarray | Dict[str, float]:
    """Compute a lightweight RL feature vector from market data.

    This is a backward-compatibility helper referenced by CI import validation.

    Args:
        df: Market DataFrame.
        lookback: Rolling window used by :func:`extract_gate_features`.
        as_array: When True returns a deterministic 1D numpy array (sorted by key).

    Returns:
        1D float32 feature array or the raw feature dict.
    """

    features = extract_gate_features(df=df, lookback=lookback)
    if not as_array:
        return features
    keys = sorted(features.keys())
    return np.array([float(features[k]) for k in keys], dtype=np.float32)


def simulate_trade(
    entry_price: float,
    direction: int,  # 1=long, -1=short
    prices: np.ndarray,
    stop_loss_pct: float = 0.01,
    take_profit_pct: float = 0.02,
    max_bars: int = 24,
) -> Dict[str, float]:
    """
    Simulate a trade outcome.

    Args:
        entry_price: Entry price
        direction: 1 for long, -1 for short
        prices: Array of future prices
        stop_loss_pct: Stop loss as percentage
        take_profit_pct: Take profit as percentage
        max_bars: Maximum bars to hold

    Returns:
        Dictionary with trade result
    """
    stop_price = entry_price * (1 - stop_loss_pct * direction)
    tp_price = entry_price * (1 + take_profit_pct * direction)

    for i, price in enumerate(prices[:max_bars]):
        if direction == 1:  # Long
            if price <= stop_price:
                return {
                    'exit_price': stop_price,
                    'pnl_pct': -stop_loss_pct,
                    'bars_held': i + 1,
                    'exit_reason': 'stop_loss',
                }
            elif price >= tp_price:
                return {
                    'exit_price': tp_price,
                    'pnl_pct': take_profit_pct,
                    'bars_held': i + 1,
                    'exit_reason': 'take_profit',
                }
        else:  # Short
            if price >= stop_price:
                return {
                    'exit_price': stop_price,
                    'pnl_pct': -stop_loss_pct,
                    'bars_held': i + 1,
                    'exit_reason': 'stop_loss',
                }
            elif price <= tp_price:
                return {
                    'exit_price': tp_price,
                    'pnl_pct': take_profit_pct,
                    'bars_held': i + 1,
                    'exit_reason': 'take_profit',
                }

    # Max bars reached - exit at market
    exit_price = prices[min(max_bars - 1, len(prices) - 1)]
    pnl_pct = (exit_price - entry_price) / entry_price * direction

    return {
        'exit_price': exit_price,
        'pnl_pct': pnl_pct,
        'bars_held': max_bars,
        'exit_reason': 'max_bars',
    }
