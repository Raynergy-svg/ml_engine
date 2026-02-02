"""
Curriculum Learning for RL Training.

Implements progressive difficulty increase for RL training:
1. Start with easy scenarios (strong trends, clear signals)
2. Gradually add harder scenarios (choppy markets, noise)
3. Final stage: full market complexity

This helps the RL agent learn fundamental skills before
facing the full complexity of real market data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DifficultyLevel(IntEnum):
    """Curriculum difficulty levels."""
    EASY = 1      # Strong trends, low noise, clear signals
    MEDIUM = 2    # Mixed trends, moderate noise
    HARD = 3      # Choppy markets, high noise, ambiguous signals
    EXPERT = 4    # Full market complexity with edge cases


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    # Difficulty progression
    initial_level: DifficultyLevel = DifficultyLevel.EASY
    max_level: DifficultyLevel = DifficultyLevel.EXPERT
    
    # Advancement criteria
    min_reward_threshold: float = 0.6  # Min normalized reward to advance
    min_episodes_per_level: int = 50   # Min episodes before advancement
    advancement_window: int = 20        # Rolling window for advancement check
    
    # Level-specific parameters
    easy_trend_strength: float = 0.0005   # Strong drift in easy mode
    easy_noise_mult: float = 0.5          # 50% of normal noise
    medium_trend_strength: float = 0.0002
    medium_noise_mult: float = 0.8
    hard_trend_strength: float = 0.00005
    hard_noise_mult: float = 1.2
    expert_trend_strength: float = 0.0    # No guaranteed trend
    expert_noise_mult: float = 1.5        # Extra noise
    
    # Regime mix per level (trending_up, trending_down, ranging, high_vol, low_vol)
    easy_regime_probs: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.1, 0.05, 0.05])
    medium_regime_probs: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.2, 0.1, 0.1])
    hard_regime_probs: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.3, 0.15, 0.15])
    expert_regime_probs: List[float] = field(default_factory=lambda: [0.15, 0.15, 0.3, 0.25, 0.15])


class CurriculumScheduler:
    """
    Manages curriculum learning progression.
    
    Tracks agent performance and advances difficulty when ready.
    """
    
    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()
        self.current_level = self.config.initial_level
        self.episode_count = 0
        self.level_episode_count = 0
        self.reward_history: List[float] = []
        self.level_history: List[DifficultyLevel] = []
        
    def get_current_params(self) -> Dict[str, Any]:
        """Get parameters for current difficulty level."""
        level = self.current_level
        config = self.config
        
        if level == DifficultyLevel.EASY:
            return {
                'trend_strength': config.easy_trend_strength,
                'noise_mult': config.easy_noise_mult,
                'regime_probs': config.easy_regime_probs,
                'level': 'easy',
                'level_int': int(level),
            }
        elif level == DifficultyLevel.MEDIUM:
            return {
                'trend_strength': config.medium_trend_strength,
                'noise_mult': config.medium_noise_mult,
                'regime_probs': config.medium_regime_probs,
                'level': 'medium',
                'level_int': int(level),
            }
        elif level == DifficultyLevel.HARD:
            return {
                'trend_strength': config.hard_trend_strength,
                'noise_mult': config.hard_noise_mult,
                'regime_probs': config.hard_regime_probs,
                'level': 'hard',
                'level_int': int(level),
            }
        else:  # EXPERT
            return {
                'trend_strength': config.expert_trend_strength,
                'noise_mult': config.expert_noise_mult,
                'regime_probs': config.expert_regime_probs,
                'level': 'expert',
                'level_int': int(level),
            }
    
    def record_episode(self, reward: float, max_possible_reward: float = 1.0) -> bool:
        """
        Record episode result and check for level advancement.
        
        Args:
            reward: Episode reward
            max_possible_reward: Maximum possible reward for normalization
            
        Returns:
            True if level advanced, False otherwise
        """
        normalized_reward = reward / max(max_possible_reward, 1e-10)
        self.reward_history.append(normalized_reward)
        self.episode_count += 1
        self.level_episode_count += 1
        self.level_history.append(self.current_level)
        
        # Check for advancement
        advanced = False
        if self._should_advance():
            self._advance_level()
            advanced = True
            
        return advanced
    
    def _should_advance(self) -> bool:
        """Check if agent should advance to next level."""
        # Can't advance beyond max level
        if self.current_level >= self.config.max_level:
            return False
        
        # Need minimum episodes at current level
        if self.level_episode_count < self.config.min_episodes_per_level:
            return False
        
        # Check recent performance
        window = min(self.config.advancement_window, len(self.reward_history))
        recent_rewards = self.reward_history[-window:]
        avg_reward = np.mean(recent_rewards)
        
        return avg_reward >= self.config.min_reward_threshold
    
    def _advance_level(self):
        """Advance to next difficulty level."""
        old_level = self.current_level
        self.current_level = DifficultyLevel(min(
            int(self.current_level) + 1,
            int(self.config.max_level)
        ))
        self.level_episode_count = 0
        
        logger.info(f"🎓 Curriculum advanced: {old_level.name} → {self.current_level.name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        return {
            'current_level': self.current_level.name,
            'current_level_int': int(self.current_level),
            'total_episodes': self.episode_count,
            'level_episodes': self.level_episode_count,
            'mean_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'recent_reward': np.mean(self.reward_history[-20:]) if self.reward_history else 0.0,
        }
    
    def reset(self):
        """Reset curriculum to initial state."""
        self.current_level = self.config.initial_level
        self.episode_count = 0
        self.level_episode_count = 0
        self.reward_history = []
        self.level_history = []


def generate_curriculum_data(
    n_samples: int,
    curriculum_params: Dict[str, Any],
    seed: int = 42,
    base_volatility: float = 0.0003,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic market data with curriculum-adjusted difficulty.
    
    Args:
        n_samples: Number of samples to generate
        curriculum_params: Parameters from CurriculumScheduler
        seed: Random seed
        base_volatility: Base price volatility
        
    Returns:
        Dictionary with prices, features, ensemble predictions
    """
    rng = np.random.default_rng(seed)
    
    trend_strength = curriculum_params['trend_strength']
    noise_mult = curriculum_params['noise_mult']
    regime_probs = curriculum_params['regime_probs']
    
    # Generate regime sequence with curriculum-adjusted probabilities
    regimes = np.zeros(n_samples, dtype=int)
    current_regime = rng.choice(5, p=regime_probs)
    switch_prob = 0.01 * noise_mult  # More switches in harder levels
    
    for i in range(n_samples):
        regimes[i] = current_regime
        if rng.random() < switch_prob:
            current_regime = rng.choice(5, p=regime_probs)
    
    # Regime-specific parameters adjusted by curriculum
    regime_params = {
        0: {"drift": trend_strength, "vol_mult": 1.0 * noise_mult},      # Trending up
        1: {"drift": -trend_strength, "vol_mult": 1.0 * noise_mult},     # Trending down
        2: {"drift": 0.0, "vol_mult": 0.8 * noise_mult},                 # Ranging
        3: {"drift": 0.0, "vol_mult": 2.5 * noise_mult},                 # High volatility
        4: {"drift": 0.0, "vol_mult": 0.4 * noise_mult},                 # Low volatility
    }
    
    # Generate price series
    prices = np.zeros(n_samples)
    prices[0] = 1.1000
    
    for i in range(1, n_samples):
        params = regime_params[regimes[i]]
        drift = params["drift"]
        vol = base_volatility * params["vol_mult"]
        
        shock = rng.normal(0, vol)
        prices[i] = prices[i-1] * (1 + drift + shock)
    
    # Compute features (simplified for curriculum training)
    returns = np.zeros(n_samples)
    returns[1:] = np.diff(prices) / prices[:-1]
    
    # ATR
    atr = np.abs(returns) * 100
    for i in range(14, n_samples):
        atr[i] = np.mean(np.abs(returns[i-14:i])) * 100
    
    # Momentum
    momentum = np.zeros(n_samples)
    for i in range(5, n_samples):
        momentum[i] = (prices[i] / prices[i-5] - 1) * 100
    
    # RSI
    rsi = np.full(n_samples, 50.0)
    for i in range(14, n_samples):
        gains = np.maximum(0, returns[i-14:i])
        losses = np.maximum(0, -returns[i-14:i])
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    # ADX (simplified)
    adx = np.zeros(n_samples)
    for i in range(14, n_samples):
        adx[i] = 100 * abs(np.sum(np.sign(returns[i-14:i]))) / 14
    adx = np.clip(adx, 0, 100)
    
    # Features array
    features = np.column_stack([
        returns,
        np.zeros_like(returns),  # returns_2 placeholder
        atr,
        adx,
        momentum,
        rsi / 100,
        np.zeros_like(returns),  # bb_position placeholder
    ])
    
    # Ensemble predictions (clearer signals in easier levels)
    signal_clarity = 1.0 / max(noise_mult, 0.5)  # Clearer in easy mode
    
    # TCN probability
    tcn_prob = 0.5 + signal_clarity * 0.3 * np.tanh(momentum * 10)
    tcn_prob += rng.normal(0, 0.03 * noise_mult, n_samples)
    tcn_prob = np.clip(tcn_prob, 0.15, 0.85)
    
    # Ridge confidence
    ridge_conf = 30 + signal_clarity * 40 * (adx / 100)
    ridge_conf += rng.normal(0, 4 * noise_mult, n_samples)
    ridge_conf = np.clip(ridge_conf, 10, 95)
    
    # XGBoost momentum
    xgb_mom = 0.25 + signal_clarity * 0.3 * np.tanh(momentum * 5)
    xgb_mom += rng.normal(0, 0.04 * noise_mult, n_samples)
    xgb_mom = np.clip(xgb_mom, 0.05, 0.95)
    
    # RF drawdown
    rf_drawdown = 0.015 + 0.01 * (atr / (np.mean(atr) + 1e-10))
    rf_drawdown[regimes == 3] *= 1.5
    rf_drawdown += rng.normal(0, 0.002 * noise_mult, n_samples)
    rf_drawdown = np.clip(rf_drawdown, 0.005, 0.05)
    
    ensemble_preds = np.column_stack([
        tcn_prob,
        ridge_conf,
        xgb_mom,
        rf_drawdown,
    ])
    
    # Direction signals
    directions = np.where(tcn_prob > 0.58, 1, np.where(tcn_prob < 0.42, -1, 0))
    
    return {
        'prices': prices,
        'features': features,
        'ensemble_preds': ensemble_preds,
        'momentum': momentum,
        'atr': atr,
        'directions': directions,
        'regimes': regimes,
        'curriculum_level': curriculum_params['level'],
    }
