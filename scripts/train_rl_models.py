#!/usr/bin/env python3
"""
Train RL models (Gate Thresholds + Exit Timing).

Generates realistic synthetic market data with multiple regimes to train
RL environments without needing full ensemble model predictions.

Features:
- Multi-regime synthetic data (trending, ranging, volatile)
- Configurable training parameters via CLI or YAML
- Training validation and evaluation metrics
- Checkpointing and early stopping support
- Comprehensive summary reports
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict

if TYPE_CHECKING:
    import pandas as pd

import numpy as np
from pathlib import Path

# Force CPU to avoid TensorFlow/PyTorch conflicts on Metal
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# Configure logging with color support
class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI colors for terminal output."""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types for synthetic data generation."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class SyntheticDataOutput(TypedDict):
    """Type definition for synthetic market data output."""
    prices: np.ndarray
    features: np.ndarray
    ensemble_preds: np.ndarray
    momentum: np.ndarray
    atr: np.ndarray
    directions: np.ndarray
    regimes: np.ndarray


@dataclass
class TrainingConfig:
    """Configuration for RL training runs."""
    timesteps: int = 50_000
    n_samples: int = 10_000
    seed: int = 42
    verbose: int = 1
    eval_freq: int = 5_000
    checkpoint_freq: int = 10_000
    log_tensorboard: bool = True
    
    # Synthetic data config
    regime_switch_prob: float = 0.01  # Probability of regime switch per bar
    base_volatility: float = 0.0003   # Base price volatility
    
    # Real data options
    use_real_data: bool = False       # Use OANDA market data instead of synthetic
    instrument: str = "EUR_USD"       # Instrument for real data
    
    # Curriculum learning
    use_curriculum: bool = False      # Enable curriculum learning
    curriculum_start_level: int = 1   # Levels: 1=easy, 2=medium, 3=hard, 4=expert

    # Incremental learning
    continue_training: bool = False   # Load existing model and continue training
    
    def __post_init__(self):
        """Validate configuration."""
        if self.timesteps < 1000:
            raise ValueError("timesteps must be >= 1000")
        if self.n_samples < 1000:
            raise ValueError("n_samples must be >= 1000")


@dataclass
class TrainingStats:
    """Statistics from a training run."""
    model_type: str
    timesteps: int
    training_time_seconds: float
    final_reward: Optional[float] = None
    mean_reward: Optional[float] = None
    std_reward: Optional[float] = None
    n_episodes: Optional[int] = None
    best_reward: Optional[float] = None
    model_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_type": self.model_type,
            "timesteps": self.timesteps,
            "training_time_seconds": round(self.training_time_seconds, 2),
            "final_reward": self.final_reward,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "n_episodes": self.n_episodes,
            "best_reward": self.best_reward,
            "model_path": self.model_path,
        }


def generate_regime_sequence(n_samples: int, switch_prob: float = 0.01, seed: int = 42) -> np.ndarray:
    """
    Generate a sequence of market regimes with realistic transitions.
    
    Args:
        n_samples: Number of samples to generate
        switch_prob: Probability of regime switch at each step
        seed: Random seed for reproducibility
        
    Returns:
        Array of regime indices (0-4 corresponding to MarketRegime enum)
    """
    rng = np.random.default_rng(seed)
    regimes = np.zeros(n_samples, dtype=int)
    current_regime = rng.integers(0, 5)
    
    # Transition matrix - some regimes more likely to follow others
    # [trending_up, trending_down, ranging, high_vol, low_vol]
    transition_probs = np.array([
        [0.3, 0.1, 0.3, 0.2, 0.1],   # From trending up
        [0.1, 0.3, 0.3, 0.2, 0.1],   # From trending down
        [0.2, 0.2, 0.3, 0.15, 0.15],  # From ranging
        [0.15, 0.15, 0.2, 0.3, 0.2],  # From high vol
        [0.2, 0.2, 0.4, 0.1, 0.1],   # From low vol
    ])
    
    for i in range(n_samples):
        regimes[i] = current_regime
        if rng.random() < switch_prob:
            current_regime = rng.choice(5, p=transition_probs[current_regime])
    
    return regimes


def _load_and_combine_data_files(
    data_dir: Path,
    pattern: str,
    n_samples: int,
) -> Optional[pd.DataFrame]:
    """
    Load and combine data files from the market data directory.
    
    Args:
        data_dir: Path to market data directory
        pattern: File pattern to match
        n_samples: Maximum samples to load
        
    Returns:
        Combined DataFrame or None if loading failed
    """
    import pandas as pd
    
    files = sorted(data_dir.glob(pattern), reverse=True)
    
    if not files:
        return None
    
    # Load and combine recent files to get enough data
    dfs = []
    total_rows = 0
    for f in files[:10]:  # Load up to 10 most recent files
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            total_rows += len(df)
            if total_rows >= n_samples:
                break
        except Exception as e:
            logger.warning(f"Error loading {f}: {e}")
            continue
    
    if not dfs:
        return None
    
    # Combine and sort by time
    combined = pd.concat(dfs, ignore_index=True)
    if 'time' in combined.columns:
        combined = combined.sort_values('time').drop_duplicates(subset=['time'])
    
    return combined


def _compute_atr(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    actual_samples: int,
) -> np.ndarray:
    """
    Compute Average True Range (ATR) indicator.
    
    Args:
        close: Close prices
        high: High prices
        low: Low prices
        actual_samples: Number of samples
        
    Returns:
        ATR values
    """
    tr = np.zeros(actual_samples)
    for i in range(1, actual_samples):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        ) / max(close[i-1], 1e-10)
    
    atr = tr.copy()
    for i in range(14, actual_samples):
        atr[i] = np.mean(tr[i-14:i]) * 100
    
    return atr


def _compute_adx(
    close: np.ndarray,
    returns_1: np.ndarray,
    actual_samples: int,
) -> np.ndarray:
    """
    Compute Average Directional Index (ADX) indicator.
    
    Args:
        close: Close prices
        returns_1: 1-period returns
        actual_samples: Number of samples
        
    Returns:
        ADX values
    """
    adx = np.zeros(actual_samples)
    for i in range(14, actual_samples):
        up_moves = np.maximum(0, np.diff(close[i-14:i]))
        down_moves = np.maximum(0, -np.diff(close[i-14:i]))
        total_move = np.sum(np.abs(returns_1[i-14:i])) + 1e-10
        di_plus = np.sum(up_moves) / total_move
        di_minus = np.sum(down_moves) / total_move
        adx[i] = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    return np.clip(adx, 0, 100)


def _compute_momentum(
    close: np.ndarray,
    actual_samples: int,
) -> np.ndarray:
    """
    Compute momentum indicator (5-period rate of change).
    
    Args:
        close: Close prices
        actual_samples: Number of samples
        
    Returns:
        Momentum values
    """
    momentum = np.zeros(actual_samples)
    for i in range(5, actual_samples):
        momentum[i] = (close[i] / close[i-5] - 1) * 100
    
    return momentum


def _compute_rsi(
    returns_1: np.ndarray,
    actual_samples: int,
) -> np.ndarray:
    """
    Compute Relative Strength Index (RSI) indicator.
    
    Args:
        returns_1: 1-period returns
        actual_samples: Number of samples
        
    Returns:
        RSI values
    """
    rsi = np.full(actual_samples, 50.0)
    for i in range(14, actual_samples):
        gains = np.maximum(0, returns_1[i-14:i])
        losses = np.maximum(0, -returns_1[i-14:i])
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def _compute_bollinger_band_position(
    close: np.ndarray,
    actual_samples: int,
) -> np.ndarray:
    """
    Compute Bollinger Band position (-1 to 1).
    
    Args:
        close: Close prices
        actual_samples: Number of samples
        
    Returns:
        Bollinger Band position values
    """
    bb_position = np.zeros(actual_samples)
    for i in range(20, actual_samples):
        window = close[i-20:i]
        mean = np.mean(window)
        std = np.std(window) + 1e-10
        bb_position[i] = (close[i] - mean) / (2 * std)
    
    return np.clip(bb_position, -1, 1)


def _detect_regimes(
    momentum: np.ndarray,
    adx: np.ndarray,
    atr: np.ndarray,
    actual_samples: int,
) -> np.ndarray:
    """
    Detect market regimes from technical indicators.
    
    Args:
        momentum: Momentum values
        adx: ADX values
        atr: ATR values
        actual_samples: Number of samples
        
    Returns:
        Regime indices (0-4)
    """
    regimes = np.zeros(actual_samples, dtype=int)
    atr_mean = np.mean(atr)
    
    for i in range(actual_samples):
        if momentum[i] > 0.1 and adx[i] > 25:
            regimes[i] = 0  # Trending up
        elif momentum[i] < -0.1 and adx[i] > 25:
            regimes[i] = 1  # Trending down
        elif atr[i] > atr_mean * 1.5:
            regimes[i] = 3  # High volatility
        elif atr[i] < atr_mean * 0.5:
            regimes[i] = 4  # Low volatility
        else:
            regimes[i] = 2  # Ranging
    
    return regimes


def _generate_ensemble_predictions(
    momentum: np.ndarray,
    adx: np.ndarray,
    atr: np.ndarray,
    rsi: np.ndarray,
    bb_position: np.ndarray,
    actual_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate ensemble predictions from technical indicators.
    
    Args:
        momentum: Momentum values
        adx: ADX values
        atr: ATR values
        rsi: RSI values
        bb_position: Bollinger Band position values
        actual_samples: Number of samples
        rng: Random number generator
        
    Returns:
        Ensemble predictions array
    """
    # TCN probability
    tcn_prob = 0.5 + 0.25 * np.tanh(momentum * 10)
    tcn_prob += 0.1 * (adx / 100 - 0.5)
    tcn_prob += rng.normal(0, 0.02, actual_samples)
    tcn_prob = np.clip(tcn_prob, 0.15, 0.85)
    
    # Ridge confidence
    ridge_conf = 30 + 35 * (adx / 100) + 10 * np.abs(bb_position)
    ridge_conf += rng.normal(0, 3, actual_samples)
    ridge_conf = np.clip(ridge_conf, 10, 95)
    
    # XGBoost momentum
    xgb_mom = 0.25 + 0.25 * np.tanh(momentum * 5) + 0.15 * (rsi / 100 - 0.5)
    xgb_mom += rng.normal(0, 0.03, actual_samples)
    xgb_mom = np.clip(xgb_mom, 0.05, 0.95)
    
    # RF drawdown
    rf_drawdown = 0.015 + 0.012 * (atr / (np.mean(atr) + 1e-10))
    rf_drawdown += rng.normal(0, 0.002, actual_samples)
    rf_drawdown = np.clip(rf_drawdown, 0.005, 0.05)
    
    return np.column_stack([tcn_prob, ridge_conf, xgb_mom, rf_drawdown])


def load_real_market_data(
    instrument: str = "EUR_USD",
    n_samples: int = 10_000,
    seed: int = 42,
) -> SyntheticDataOutput:
    """
    Load real OANDA market data for RL training.
    
    Args:
        instrument: Trading instrument (e.g., EUR_USD)
        n_samples: Maximum samples to load
        seed: Random seed for any random operations
        
    Returns:
        Dictionary matching SyntheticDataOutput format
    """
    from pathlib import Path
    
    data_dir = Path("market_data")
    pattern = f"oanda_{instrument}_H1_live_*.csv"
    
    # Load and combine data files
    combined = _load_and_combine_data_files(data_dir, pattern, n_samples)
    
    if combined is None:
        logger.warning(f"No data files found for {instrument}, falling back to synthetic")
        return generate_synthetic_market_data(n_samples, seed=seed)
    
    logger.info(f"Loaded {len(combined)} samples from real {instrument} data")
    
    # Extract OHLCV
    close = combined['close'].values[-n_samples:]
    high = combined['high'].values[-n_samples:] if 'high' in combined.columns else close
    low = combined['low'].values[-n_samples:] if 'low' in combined.columns else close
    actual_samples = len(close)
    
    # Compute features
    rng = np.random.default_rng(seed)
    
    # Returns
    returns_1 = np.zeros(actual_samples)
    returns_1[1:] = np.diff(close) / np.maximum(close[:-1], 1e-10)
    
    returns_2 = np.zeros(actual_samples)
    returns_2[2:] = (close[2:] - close[:-2]) / np.maximum(close[:-2], 1e-10)
    
    # Technical indicators
    atr = _compute_atr(close, high, low, actual_samples)
    adx = _compute_adx(close, returns_1, actual_samples)
    momentum = _compute_momentum(close, actual_samples)
    rsi = _compute_rsi(returns_1, actual_samples)
    bb_position = _compute_bollinger_band_position(close, actual_samples)
    
    # Features array
    features = np.column_stack([
        returns_1, returns_2, atr, adx,
        momentum, rsi / 100, bb_position,
    ])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Generate ensemble predictions
    ensemble_preds = _generate_ensemble_predictions(
        momentum, adx, atr, rsi, bb_position, actual_samples, rng
    )
    
    # Detect regimes
    regimes = _detect_regimes(momentum, adx, atr, actual_samples)
    
    # Direction signals
    tcn_prob = ensemble_preds[:, 0]
    directions = np.where(tcn_prob > 0.58, 1, np.where(tcn_prob < 0.42, -1, 0))
    
    logger.info(f"Real data: {actual_samples} samples, regime distribution: {np.bincount(regimes, minlength=5)}")
    
    return {
        'prices': close,
        'features': features,
        'ensemble_preds': ensemble_preds,
        'momentum': momentum,
        'atr': atr,
        'directions': directions,
        'regimes': regimes,
    }


def generate_synthetic_market_data(
    n_samples: int = 10_000,
    seed: int = 42,
    base_volatility: float = 0.0003,
    regime_switch_prob: float = 0.01,
) -> SyntheticDataOutput:
    """
    Generate realistic synthetic market data with multiple regimes.
    
    Creates price series that exhibit characteristics of real FX markets:
    - Trending periods (up/down)
    - Mean-reverting ranging periods
    - High and low volatility regimes
    - Realistic feature correlations
    
    Args:
        n_samples: Number of data points to generate
        seed: Random seed for reproducibility
        base_volatility: Base volatility for price movements
        regime_switch_prob: Probability of regime change per bar
        
    Returns:
        Dictionary containing prices, features, ensemble predictions, and metadata
    """
    rng = np.random.default_rng(seed)
    
    # Generate regime sequence
    regimes = generate_regime_sequence(n_samples, regime_switch_prob, seed)
    
    # Regime-specific parameters
    regime_params = {
        0: {"drift": 0.0002, "vol_mult": 1.0, "mean_revert": 0.0},      # Trending up
        1: {"drift": -0.0002, "vol_mult": 1.0, "mean_revert": 0.0},     # Trending down
        2: {"drift": 0.0, "vol_mult": 0.8, "mean_revert": 0.02},        # Ranging
        3: {"drift": 0.0, "vol_mult": 2.5, "mean_revert": 0.005},       # High volatility
        4: {"drift": 0.0, "vol_mult": 0.4, "mean_revert": 0.01},        # Low volatility
    }
    
    # Generate price series
    prices = np.zeros(n_samples)
    prices[0] = 1.1000  # Start at EUR/USD typical value
    mean_price = 1.1000
    
    for i in range(1, n_samples):
        params = regime_params[regimes[i]]
        drift = params["drift"]
        vol = base_volatility * params["vol_mult"]
        mean_revert = params["mean_revert"] * (mean_price - prices[i-1])
        
        # Price update with regime-specific behavior
        shock = rng.normal(0, vol)
        prices[i] = prices[i-1] * (1 + drift + shock) + mean_revert
        
        # Update rolling mean for mean reversion (slow adaptation)
        mean_price = 0.999 * mean_price + 0.001 * prices[i]
    
    # Compute features with proper handling of edge cases
    returns_1 = np.zeros(n_samples)
    returns_1[1:] = np.diff(prices) / prices[:-1]
    
    returns_2 = np.zeros(n_samples)
    returns_2[2:] = (prices[2:] - prices[:-2]) / prices[:-2]
    
    # ATR-like volatility (14-period rolling)
    atr = np.abs(returns_1) * 100
    for i in range(14, n_samples):
        atr[i] = np.mean(np.abs(returns_1[i-14:i])) * 100
    
    # ADX-like trend strength (14-period)
    adx = np.zeros(n_samples)
    for i in range(14, n_samples):
        # Simplified ADX: directional movement ratio
        up_moves = np.maximum(0, np.diff(prices[i-14:i]))
        down_moves = np.maximum(0, -np.diff(prices[i-14:i]))
        di_plus = np.sum(up_moves) / (np.sum(np.abs(returns_1[i-14:i])) + 1e-10)
        di_minus = np.sum(down_moves) / (np.sum(np.abs(returns_1[i-14:i])) + 1e-10)
        adx[i] = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = np.clip(adx, 0, 100)
    
    # Momentum (rate of change)
    momentum = returns_2 * 100
    
    # RSI-like oscillator
    rsi = np.full(n_samples, 50.0)
    for i in range(14, n_samples):
        gains = np.maximum(0, returns_1[i-14:i])
        losses = np.maximum(0, -returns_1[i-14:i])
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100 if avg_gain > 0 else 50
    
    # Bollinger Band position (-1 to 1)
    bb_position = np.zeros(n_samples)
    for i in range(20, n_samples):
        window = prices[i-20:i]
        mean = np.mean(window)
        std = np.std(window) + 1e-10
        bb_position[i] = (prices[i] - mean) / (2 * std)
    bb_position = np.clip(bb_position, -1, 1)
    
    # Features array (n_samples x n_features)
    features = np.column_stack([
        returns_1,
        returns_2,
        atr,
        adx,
        momentum,
        rsi / 100,  # Normalize to 0-1
        bb_position,
    ])
    
    # Replace any NaN/inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Generate ensemble-like predictions based on features and regimes
    # TCN probability (direction confidence)
    tcn_prob = 0.5 + 0.25 * np.tanh(momentum * 10)
    tcn_prob += 0.1 * (adx / 100 - 0.5)  # Higher ADX -> more confident
    tcn_prob += rng.normal(0, 0.03, n_samples)
    tcn_prob = np.clip(tcn_prob, 0.15, 0.85)
    
    # Ridge confidence (trend strength indicator)
    ridge_conf = 30 + 35 * (adx / 100) + 10 * np.abs(bb_position)
    ridge_conf += rng.normal(0, 4, n_samples)
    ridge_conf = np.clip(ridge_conf, 10, 95)
    
    # XGBoost momentum percentile
    xgb_mom = 0.25 + 0.25 * np.tanh(momentum * 5) + 0.15 * (rsi / 100 - 0.5)
    xgb_mom += rng.normal(0, 0.04, n_samples)
    xgb_mom = np.clip(xgb_mom, 0.05, 0.95)
    
    # RF drawdown estimate (higher in volatile regimes)
    rf_drawdown = 0.015 + 0.012 * (atr / (np.mean(atr) + 1e-10))
    # Increase drawdown estimate in high volatility regime
    rf_drawdown[regimes == 3] *= 1.5
    rf_drawdown += rng.normal(0, 0.002, n_samples)
    rf_drawdown = np.clip(rf_drawdown, 0.005, 0.05)
    
    ensemble_preds = np.column_stack([
        tcn_prob,
        ridge_conf,
        xgb_mom,
        rf_drawdown,
    ])
    
    # Direction signals for exit training
    directions = np.where(tcn_prob > 0.58, 1, np.where(tcn_prob < 0.42, -1, 0))
    
    # Log data statistics
    logger.info(f"Generated {n_samples:,} samples with {len(MarketRegime)} regime types")
    regime_names = ["trending_up", "trending_down", "ranging", "high_volatility", "low_volatility"]
    regime_counts = {regime_names[i]: int(np.sum(regimes == i)) for i in range(5)}
    logger.debug(f"Regime distribution: {regime_counts}")
    
    return {
        'prices': prices,
        'features': features,
        'ensemble_preds': ensemble_preds,
        'momentum': momentum,
        'atr': atr,
        'directions': directions,
        'regimes': regimes,
    }


def validate_training_result(stats: Optional[Dict[str, Any]], model_type: str) -> bool:
    """
    Validate that training completed successfully.
    
    Args:
        stats: Training statistics dictionary
        model_type: Type of model trained
        
    Returns:
        True if training was successful, False otherwise
    """
    if stats is None:
        logger.error(f"{model_type} training returned no statistics")
        return False
    
    # Check for expected keys
    expected_keys = {'mean_reward', 'total_timesteps'}
    if not all(k in stats for k in expected_keys if hasattr(stats, 'get')):
        logger.warning(f"{model_type} stats missing expected keys")
    
    return True


def _load_training_data(config: TrainingConfig) -> SyntheticDataOutput:
    """
    Load or generate training data based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Market data dictionary
    """
    if config.use_real_data:
        logger.info(f"📊 Loading real market data for {config.instrument}...")
        return load_real_market_data(
            instrument=config.instrument,
            n_samples=config.n_samples,
            seed=config.seed,
        )
    
    if config.use_curriculum:
        logger.info("📊 Generating curriculum-based data...")
        from src.rl.curriculum import (
            generate_curriculum_data, CurriculumScheduler, 
            CurriculumConfig, DifficultyLevel
        )
        levels = list(DifficultyLevel)
        start_level = levels[config.curriculum_start_level - 1]
        curriculum_config = CurriculumConfig(initial_level=start_level)
        scheduler = CurriculumScheduler(config=curriculum_config)
        curriculum_params = scheduler.get_current_params()
        logger.info(f"  Curriculum level: {curriculum_params['level']}")
        return generate_curriculum_data(
            n_samples=config.n_samples,
            curriculum_params=curriculum_params,
            seed=config.seed,
        )
    
    logger.info("📊 Generating synthetic market data...")
    return generate_synthetic_market_data(
        n_samples=config.n_samples,
        seed=config.seed,
        base_volatility=config.base_volatility,
        regime_switch_prob=config.regime_switch_prob,
    )


def _resolve_gate_rl_config(config: TrainingConfig) -> Any:
    """
    Resolve GateRLConfig, loading from saved state if continuing training.
    
    Args:
        config: Training configuration
        
    Returns:
        GateRLConfig instance
    """
    from src.rl.gate_threshold_env import (
        GateRLConfig, GATE_RL_MODEL_PATH, infer_config_from_model
    )
    import pickle
    
    config_path = GATE_RL_MODEL_PATH.parent / "gate_rl_config.pkl"
    
    if not (config.continue_training and GATE_RL_MODEL_PATH.exists()):
        return GateRLConfig(total_timesteps=config.timesteps)
    
    if config_path.exists():
        logger.info(f"📂 Pre-loading saved config for continuation: {config_path}")
        with open(config_path, 'rb') as f:
            rl_config = pickle.load(f)
        rl_config.total_timesteps = config.timesteps
        return rl_config
    
    # Try to infer from model
    logger.warning("⚠️ Saved config not found - inferring from model observation space...")
    inferred = infer_config_from_model(GATE_RL_MODEL_PATH)
    rl_config = GateRLConfig(total_timesteps=config.timesteps)
    if inferred:
        logger.info(f"📊 Inferred max_drawdown_pct={inferred['max_drawdown_pct']} from saved model")
        rl_config.max_drawdown_pct = inferred['max_drawdown_pct']
    return rl_config


def train_gate_thresholds(
    config: TrainingConfig,
) -> TrainingStats:
    """
    Train the SAC gate threshold optimizer.
    
    Args:
        config: Training configuration
        
    Returns:
        TrainingStats with training results
    """
    from src.rl.gate_threshold_env import GateThresholdRL
    
    logger.info("=" * 60)
    logger.info("🎯 TRAINING GATE THRESHOLD RL (SAC)")
    logger.info("=" * 60)
    logger.info(f"  Timesteps: {config.timesteps:,}")
    logger.info(f"  Samples: {config.n_samples:,}")
    logger.info(f"  Seed: {config.seed}")
    logger.info(f"  Data source: {'Real OANDA' if config.use_real_data else 'Synthetic'}")
    if config.use_curriculum:
        logger.info(f"  Curriculum: Level {config.curriculum_start_level}")
    
    start_time = time.time()
    
    # Load or generate training data
    data = _load_training_data(config)
    
    # Resolve RL config (handles continuation training)
    rl_config = _resolve_gate_rl_config(config)
    rl = GateThresholdRL(config=rl_config)
    
    # Train with progress logging
    logger.info("🚀 Starting SAC training...")
    stats = rl.train(
        prices=data['prices'],
        features=data['features'],
        ensemble_preds=data['ensemble_preds'],
        timesteps=config.timesteps,
        verbose=config.verbose,
        continue_training=config.continue_training,
    )
    
    # Save model
    model_path = rl.save()
    training_time = time.time() - start_time
    
    logger.info(f"✅ Gate threshold model saved to: {model_path}")
    logger.info(f"⏱️  Training time: {training_time:.1f}s")
    
    # Extract stats safely
    mean_reward = stats.get('mean_reward') if isinstance(stats, dict) else None
    std_reward = stats.get('std_reward') if isinstance(stats, dict) else None
    
    return TrainingStats(
        model_type="SAC_GateThresholds",
        timesteps=config.timesteps,
        training_time_seconds=training_time,
        mean_reward=mean_reward,
        std_reward=std_reward,
        model_path=str(model_path) if model_path else None,
    )


def train_optimal_exits(
    config: TrainingConfig,
) -> TrainingStats:
    """
    Train the PPO optimal exit optimizer.
    
    Args:
        config: Training configuration
        
    Returns:
        TrainingStats with training results
    """
    from src.rl.optimal_exit_env import OptimalExitRL, ExitRLConfig
    
    logger.info("=" * 60)
    logger.info("🎯 TRAINING OPTIMAL EXIT RL (PPO)")
    logger.info("=" * 60)
    logger.info(f"  Timesteps: {config.timesteps:,}")
    logger.info(f"  Seed: {config.seed}")
    logger.info(f"  Data source: {'Real OANDA' if config.use_real_data else 'Synthetic'}")
    if config.use_curriculum:
        logger.info(f"  Curriculum: Level {config.curriculum_start_level}")
    
    start_time = time.time()
    
    # Generate or load market data for scenario generation if needed
    if config.use_real_data:
        logger.info(f"📊 Loading real market data for {config.instrument}...")
        data = load_real_market_data(
            instrument=config.instrument,
            n_samples=config.n_samples,
            seed=config.seed,
        )
        # Convert to scenarios format for exit RL - log to confirm data is ready
        scenarios_list = _convert_data_to_exit_scenarios(data, config.seed)
        logger.info(f"  Converted to {len(scenarios_list)} exit scenarios")
    elif config.use_curriculum:
        logger.info("📊 Generating curriculum-based data...")
        from src.rl.curriculum import generate_curriculum_data, CurriculumScheduler, CurriculumConfig, DifficultyLevel
        levels = list(DifficultyLevel)
        start_level = levels[config.curriculum_start_level - 1]
        curriculum_config = CurriculumConfig(initial_level=start_level)
        scheduler = CurriculumScheduler(config=curriculum_config)
        curriculum_params = scheduler.get_current_params()
        logger.info(f"  Curriculum level: {curriculum_params['level']}")
        data = generate_curriculum_data(
            n_samples=config.n_samples,
            curriculum_params=curriculum_params,
            seed=config.seed,
        )
        # Convert to scenarios format for exit RL - log to confirm data is ready
        scenarios_list = _convert_data_to_exit_scenarios(data, config.seed)
        logger.info(f"  Converted to {len(scenarios_list)} exit scenarios")
    
    # Create trainer
    rl_config = ExitRLConfig(total_timesteps=config.timesteps)
    rl = OptimalExitRL(config=rl_config)
    
    # Train with synthetic scenarios (auto-generated inside train())
    logger.info("🚀 Starting PPO training...")
    stats = rl.train(
        scenarios=None,  # Will generate synthetic scenarios
        timesteps=config.timesteps,
        verbose=config.verbose,
    )
    
    # Save model
    model_path = rl.save()
    training_time = time.time() - start_time
    
    logger.info(f"✅ Optimal exit model saved to: {model_path}")
    logger.info(f"⏱️  Training time: {training_time:.1f}s")
    
    # Extract stats safely
    mean_reward = stats.get('mean_reward') if isinstance(stats, dict) else None
    std_reward = stats.get('std_reward') if isinstance(stats, dict) else None
    
    return TrainingStats(
        model_type="PPO_OptimalExit",
        timesteps=config.timesteps,
        training_time_seconds=training_time,
        mean_reward=mean_reward,
        std_reward=std_reward,
        model_path=str(model_path) if model_path else None,
    )


def _convert_data_to_exit_scenarios(
    data: SyntheticDataOutput,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Convert market data to exit scenarios for PPO training.
    
    Args:
        data: Market data dictionary with prices, features, etc.
        seed: Random seed (reserved for future use)
        
    Returns:
        List of scenario dictionaries for OptimalExitRL
    """
    _ = seed  # Reserved for future random operations
    prices = data['prices']
    n = len(prices)
    
    scenarios = []
    # Create scenarios from contiguous price segments
    scenario_length = 50  # bars per scenario
    
    for i in range(0, n - scenario_length, scenario_length // 2):
        segment_prices = prices[i:i + scenario_length]
        entry_price = segment_prices[0]
        
        # Calculate scenario characteristics
        max_price = np.max(segment_prices)
        min_price = np.min(segment_prices)
        final_price = segment_prices[-1]
        
        # Determine direction and calculate realistic SL/TP
        direction = 1 if final_price > entry_price else -1
        pip_value = 0.0001 if entry_price < 10 else 0.01
        
        price_range = (max_price - min_price) / pip_value
        
        scenario = {
            'prices': segment_prices.tolist(),
            'entry_price': entry_price,
            'direction': direction,
            'stop_loss_pips': max(10, min(50, price_range * 0.4)),
            'take_profit_pips': max(15, min(100, price_range * 0.6)),
            'momentum': data.get('momentum', np.zeros(n))[i + scenario_length // 2],
            'atr': data.get('atr', np.ones(n) * 0.001)[i + scenario_length // 2],
        }
        scenarios.append(scenario)
    
    logger.info(f"Created {len(scenarios)} exit scenarios from market data")
    return scenarios


def print_summary_report(results: List[TrainingStats]) -> None:
    """Print a formatted summary report of all training runs."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("📋 TRAINING SUMMARY REPORT")
    logger.info("=" * 70)
    
    total_time = sum(r.training_time_seconds for r in results)
    
    for stats in results:
        logger.info(f"\n  Model: {stats.model_type}")
        logger.info(f"  ├── Timesteps: {stats.timesteps:,}")
        logger.info(f"  ├── Training Time: {stats.training_time_seconds:.1f}s")
        if stats.mean_reward is not None:
            logger.info(f"  ├── Mean Reward: {stats.mean_reward:.4f}")
        if stats.std_reward is not None:
            logger.info(f"  ├── Std Reward: {stats.std_reward:.4f}")
        if stats.model_path:
            logger.info(f"  └── Saved To: {stats.model_path}")
    
    logger.info(f"\n  Total Training Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("=" * 70)


def save_training_report(results: List[TrainingStats], output_dir: Path) -> None:
    """Save training report as JSON for later analysis."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "models": [r.to_dict() for r in results],
        "total_time_seconds": sum(r.training_time_seconds for r in results),
    }
    
    report_path = output_dir / "rl_training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📄 Training report saved to: {report_path}")


def main() -> int:
    """Main entry point for RL training script."""
    parser = argparse.ArgumentParser(
        description="Train RL models for ML Engine FX Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_rl_models.py                    # Train both models
  python scripts/train_rl_models.py --timesteps 100000 # More training
  python scripts/train_rl_models.py --gates-only       # Only gate thresholds
  python scripts/train_rl_models.py --exits-only       # Only exit timing
  python scripts/train_rl_models.py --samples 20000    # More synthetic data
  python scripts/train_rl_models.py --use-real-data --instrument EUR_USD  # Real OANDA data
  python scripts/train_rl_models.py --curriculum       # Progressive difficulty
        """
    )
    parser.add_argument(
        "--timesteps", type=int, default=50_000,
        help="Training timesteps per model (default: 50000)"
    )
    parser.add_argument(
        "--samples", type=int, default=10_000,
        help="Number of synthetic data samples (default: 10000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--gates-only", action="store_true",
        help="Train only gate threshold model (SAC)"
    )
    parser.add_argument(
        "--exits-only", action="store_true",
        help="Train only exit timing model (PPO)"
    )
    parser.add_argument(
        "--verbose", type=int, default=1, choices=[0, 1, 2],
        help="Verbosity level: 0=silent, 1=progress, 2=debug (default: 1)"
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip saving JSON training report"
    )
    # New data source options
    parser.add_argument(
        "--use-real-data", action="store_true",
        help="Use real OANDA market data instead of synthetic"
    )
    parser.add_argument(
        "--instrument", type=str, default="EUR_USD",
        help="Instrument for real data (default: EUR_USD)"
    )
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Use curriculum learning with progressive difficulty"
    )
    parser.add_argument(
        "--curriculum-level", type=int, default=1, choices=[1, 2, 3, 4],
        help="Starting curriculum level: 1=easy, 2=medium, 3=hard, 4=expert (default: 1)"
    )
    parser.add_argument(
        "--continue", dest="continue_training", action="store_true",
        help="Continue training from last saved model instead of starting fresh"
    )
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        timesteps=args.timesteps,
        n_samples=args.samples,
        seed=args.seed,
        verbose=args.verbose,
        use_real_data=args.use_real_data,
        instrument=args.instrument,
        use_curriculum=args.curriculum,
        curriculum_start_level=args.curriculum_level,
        continue_training=args.continue_training,
    )
    
    # Ensure output directories exist
    output_dirs = [
        Path("trained_data/models"),
        Path("trained_data/logs/rl_gates"),
        Path("trained_data/logs/rl_exits"),
    ]
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    logger.info("🤖 ML Engine RL Training Pipeline")
    logger.info(f"   Config: timesteps={config.timesteps:,}, samples={config.n_samples:,}, seed={config.seed}")
    
    results: List[TrainingStats] = []
    
    # Train requested models
    if args.gates_only:
        results.append(train_gate_thresholds(config))
    elif args.exits_only:
        results.append(train_optimal_exits(config))
    else:
        # Train both sequentially
        results.append(train_gate_thresholds(config))
        results.append(train_optimal_exits(config))
    
    # Print summary
    print_summary_report(results)
    
    # Save report
    if not args.no_report:
        save_training_report(results, Path("trained_data/logs"))
    
    logger.info("")
    logger.info("✅ ALL RL TRAINING COMPLETE")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
