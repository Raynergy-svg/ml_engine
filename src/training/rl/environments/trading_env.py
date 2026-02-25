"""Gymnasium trading environment for PPO-based position sizing.

REWARD SYSTEM OVERVIEW:
- Base reward: Profit/Loss scaled by equity
- Transaction costs: Spread + commission per trade
- Position change penalty: Discourages excessive trading/churning
- Asymmetric scaling: Losses penalized more heavily than gains rewarded
- Holding incentives: Bonus for holding winners, penalty for holding losers
- Risk adjustments: Drawdown penalty, Sharpe/Sortino components
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.training.rl import runtime

logger = logging.getLogger(__name__)

# Discrete position size levels (percentage of equity).
POSITION_LEVELS = [0.0, 0.01, 0.03, 0.05, 0.07, 0.10]


@dataclass
class TradingRewardConfig:
    """Configuration for reward shaping in trading environment.
    
    This config addresses the key issues that caused 43K trades:
    - Transaction costs make frequent trading expensive
    - Position change penalties discourage churning
    - Asymmetric rewards make losses hurt more than gains feel good
    - Holding incentives encourage letting winners run and cutting losers
    """
    # Transaction costs (CRITICAL for preventing excessive trading)
    spread_cost_pct: float = 0.0002  # 0.02% spread (2 pips for FX)
    commission_pct: float = 0.0001   # 0.01% commission
    slippage_pct: float = 0.0001     # 0.01% slippage
    
    # Position change penalty (discourages churning)
    position_change_penalty: float = 0.001  # Penalty per position change
    
    # Asymmetric reward scaling (losses hurt more)
    profit_multiplier: float = 1.5    # Bonus multiplier for profits
    loss_multiplier: float = 2.0      # Penalty multiplier for losses
    
    # Holding incentives
    winner_holding_bonus: float = 0.0005  # Bonus per step for holding winners
    loser_holding_penalty: float = 0.002  # Penalty per step for holding losers
    
    # Trade frequency penalty
    trade_frequency_penalty: float = 0.0002  # Penalty per trade (discourages high frequency)
    max_trades_per_episode: int = 500  # Soft cap on trades
    
    # Risk adjustments (from original config)
    sharpe_weight: float = 0.1
    drawdown_penalty: float = 0.5
    drawdown_threshold: float = 0.05  # Start penalizing at 5% DD
    win_rate_bonus: float = 0.2
    win_rate_threshold: float = 0.5
    
    # Reward clipping
    max_reward_per_step: float = 5.0
    min_reward_per_step: float = -5.0


def _get_gym_env_base():
    """Get gym.Env base class lazily."""
    runtime._ensure_gym_imported()
    if runtime.GYM_AVAILABLE and runtime.gym is not None:
        return runtime.gym.Env
    return object


class TradingEnv(_get_gym_env_base()):
    """Environment for RL-based position sizing with proper reward shaping.
    
    KEY FIXES from the 43K trades issue:
    1. Transaction costs (spread + commission + slippage) make frequent trading expensive
    2. Position change penalty discourages churning
    3. Asymmetric rewards (losses hurt 2x more than gains feel good)
    4. Holding incentives encourage letting winners run and cutting losers
    5. Trade frequency penalty scales with number of trades
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        ensemble_predictions: np.ndarray,
        prices: np.ndarray,
        config: Optional[Any] = None,
        reward_config: Optional[TradingRewardConfig] = None,
    ):
        runtime._ensure_gym_imported()
        if not runtime.GYM_AVAILABLE:
            raise ImportError("gymnasium is required. Install with: pip install gymnasium")

        super().__init__()

        if config is None:
            # Avoid import cycles by importing lazily.
            from src.training.rl.position_sizer import RLConfig

            config = RLConfig()

        self.config = config
        self.reward_config = reward_config or TradingRewardConfig()
        
        self.features = features
        self.ensemble_predictions = ensemble_predictions
        self.prices = prices
        self.n_samples = len(prices)

        # Validate data alignment.
        assert len(features) == len(prices), "Features and prices must have same length"
        assert len(ensemble_predictions) == len(prices), "Predictions and prices must have same length"

        # State tracking.
        self.current_step = self.config.sequence_length
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0.0
        self.position_price = 0.0
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_pnl = 0.0
        self.max_equity = self.initial_balance
        
        # NEW: Track position changes and holding time for reward shaping
        self.prev_position_pct = 0.0
        self.position_entry_step = 0
        self.position_unrealized_pnl = 0.0
        self.total_transaction_costs = 0.0

        # Observation space: features + predictions + account state.
        n_features = features.shape[1] if len(features.shape) > 1 else 1
        obs_dim = n_features + 2 + 4

        self.observation_space = runtime.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action space: discretized position sizing.
        self.action_space = runtime.spaces.Discrete(len(POSITION_LEVELS))
        self.position_levels = POSITION_LEVELS

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        if len(self.features.shape) > 1:
            market_features = self.features[self.current_step]
        else:
            market_features = np.array([self.features[self.current_step]])

        pred = self.ensemble_predictions[self.current_step]
        direction_prob = pred[0] if len(pred) > 0 else 0.5
        confidence = pred[1] if len(pred) > 1 else 0.5

        equity_pct = self.equity / self.initial_balance - 1.0
        drawdown = (
            (self.max_equity - self.equity) / self.max_equity
            if self.max_equity > 0
            else 0.0
        )
        daily_pnl_pct = self.daily_pnl / self.initial_balance

        recent_trades = self.trade_history[-20:] if self.trade_history else []
        win_rate = (
            sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
            / max(len(recent_trades), 1)
        )

        obs = np.concatenate(
            [
                market_features.flatten(),
                [direction_prob, confidence],
                [equity_pct, drawdown, daily_pnl_pct, win_rate],
            ]
        ).astype(np.float32)

        return obs

    def _calculate_transaction_costs(self, position_value: float, is_new_position: bool) -> float:
        """Calculate transaction costs for a trade.
        
        CRITICAL: This is what was missing - without transaction costs,
        the agent can trade every step with no penalty, leading to 43K trades.
        
        Args:
            position_value: Value of the position being opened/closed
            is_new_position: True if opening, False if closing
            
        Returns:
            Total transaction cost in dollars
        """
        cfg = self.reward_config
        
        # Spread cost (applies to both open and close)
        spread_cost = position_value * cfg.spread_cost_pct
        
        # Commission (applies to both open and close)
        commission = position_value * cfg.commission_pct
        
        # Slippage (applies to both open and close)
        slippage = position_value * cfg.slippage_pct
        
        total_cost = spread_cost + commission + slippage
        
        return total_cost

    def _calculate_reward(
        self, 
        pnl: float, 
        position_pct: float,
        position_changed: bool,
        transaction_costs: float,
    ) -> float:
        """Calculate shaped reward with transaction costs and asymmetric scaling.
        
        KEY PRINCIPLES:
        1. Transaction costs penalize excessive trading
        2. Losses hurt MORE than gains feel good (loss_multiplier > profit_multiplier)
        3. Position changes are penalized to discourage churning
        4. Holding winners is rewarded, holding losers is penalized
        5. Trade frequency penalty scales with total trades
        """
        cfg = self.reward_config
        reward = 0.0
        
        # ========================================
        # 1. BASE P/L REWARD WITH ASYMMETRIC SCALING
        # ========================================
        # Normalize P/L to percentage of initial balance
        pnl_pct = pnl / self.initial_balance
        
        if pnl > 0:
            # Profitable trade: apply bonus multiplier
            # This makes the agent seek profitable trades more aggressively
            base_reward = pnl_pct * 100 * cfg.profit_multiplier
        else:
            # Losing trade: apply penalty multiplier (losses hurt MORE)
            # This makes the agent more risk-averse
            base_reward = pnl_pct * 100 * cfg.loss_multiplier
        
        reward += base_reward
        
        # ========================================
        # 2. TRANSACTION COSTS (CRITICAL FIX)
        # ========================================
        # Deduct transaction costs from reward
        # This is the KEY fix for 43K trades - without this, trading is "free"
        cost_penalty = transaction_costs / self.initial_balance * 100
        reward -= cost_penalty * 2.0  # Double penalty to really discourage excessive trading
        
        # ========================================
        # 3. POSITION CHANGE PENALTY
        # ========================================
        # Penalize changing position size (discourages churning)
        if position_changed:
            reward -= cfg.position_change_penalty * 10  # Scale to be meaningful
        
        # ========================================
        # 4. HOLDING INCENTIVES
        # ========================================
        # Reward holding winning positions, penalize holding losers
        if self.position > 0 and self.position_price > 0:
            # Calculate unrealized P/L for position holding incentive
            current_price = self.prices[self.current_step - 1]
            unrealized_pnl_pct = (current_price - self.position_price) / self.position_price
            
            if unrealized_pnl_pct > 0:
                # Holding a winner - small bonus to encourage letting winners run
                reward += cfg.winner_holding_bonus
            else:
                # Holding a loser - penalty to encourage cutting losses
                reward -= cfg.loser_holding_penalty
        
        # ========================================
        # 5. TRADE FREQUENCY PENALTY
        # ========================================
        # Penalize high trade frequency (discourages 43K trades)
        n_trades = len(self.trade_history)
        if n_trades > 0:
            # Progressive penalty that increases with trade count
            freq_penalty = cfg.trade_frequency_penalty * np.sqrt(n_trades)
            reward -= freq_penalty
        
        # ========================================
        # 6. RISK ADJUSTMENTS (from original)
        # ========================================
        # Sharpe ratio component (reward consistency)
        if len(self.trade_history) > 5:
            returns = [t.get("pnl", 0) for t in self.trade_history[-20:]]
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns)
                reward += cfg.sharpe_weight * sharpe * 0.5  # Reduced weight

        # Drawdown penalty (risk management)
        drawdown = (
            (self.max_equity - self.equity) / self.max_equity
            if self.max_equity > 0
            else 0.0
        )
        if drawdown > cfg.drawdown_threshold:
            # Progressive penalty that increases with drawdown
            dd_penalty = cfg.drawdown_penalty * (drawdown ** 1.5) * 50
            reward -= dd_penalty

        # Win rate bonus (reward consistency)
        recent_trades = self.trade_history[-10:] if self.trade_history else []
        if recent_trades:
            win_rate = (
                sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
                / len(recent_trades)
            )
            if win_rate > cfg.win_rate_threshold:
                reward += cfg.win_rate_bonus * (win_rate - cfg.win_rate_threshold) * 5

        # ========================================
        # 7. CLIP REWARD TO PREVENT EXPLOSION
        # ========================================
        reward = np.clip(reward, cfg.min_reward_per_step, cfg.max_reward_per_step)
        
        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step with transaction cost tracking."""
        position_pct = self.position_levels[action]

        current_price = self.prices[self.current_step]
        
        # Track position changes for penalty calculation
        position_changed = (position_pct != self.prev_position_pct)
        
        # Calculate transaction costs if position changed
        transaction_costs = 0.0
        if position_changed and position_pct > 0:
            # Opening or changing position - incur transaction costs
            position_value = self.equity * position_pct
            transaction_costs = self._calculate_transaction_costs(position_value, is_new_position=True)
            self.total_transaction_costs += transaction_costs
            
            # Track position entry for holding incentives
            if self.prev_position_pct == 0:
                # New position opened
                self.position_entry_step = self.current_step
                self.position_price = current_price
        
        self.current_step += 1

        if self.current_step >= self.n_samples - 1:
            obs = self._get_observation()
            return obs, 0.0, True, False, {"reason": "end_of_data"}

        next_price = self.prices[self.current_step]

        pred = self.ensemble_predictions[self.current_step - 1]
        direction_prob = pred[0] if len(pred) > 0 else 0.5
        direction = 1 if direction_prob > 0.5 else -1

        position_value = self.equity * position_pct
        price_change_pct = (next_price - current_price) / current_price
        pnl = position_value * price_change_pct * direction
        
        # Deduct transaction costs from equity (CRITICAL FIX)
        self.equity += pnl - transaction_costs
        self.daily_pnl += pnl - transaction_costs
        self.max_equity = max(self.max_equity, self.equity)
        
        # Update position tracking
        self.prev_position_pct = position_pct
        self.position = position_pct

        if position_pct > 0:
            self.trade_history.append(
                {
                    "step": self.current_step,
                    "direction": direction,
                    "position_pct": position_pct,
                    "entry_price": current_price,
                    "exit_price": next_price,
                    "pnl": pnl,
                    "transaction_costs": transaction_costs,
                }
            )

        # Use the new reward function with all components
        reward = self._calculate_reward(
            pnl=pnl,
            position_pct=position_pct,
            position_changed=position_changed,
            transaction_costs=transaction_costs,
        )

        terminated = False
        truncated = False
        info: Dict[str, Any] = {
            "pnl": pnl,
            "equity": self.equity,
            "transaction_costs": transaction_costs,
            "total_transaction_costs": self.total_transaction_costs,
            "position_changed": position_changed,
        }

        drawdown = (
            (self.max_equity - self.equity) / self.max_equity
            if self.max_equity > 0
            else 0.0
        )
        if drawdown > self.config.max_drawdown_pct:
            terminated = True
            reward -= 10.0
            info["reason"] = "max_drawdown"

        if self.daily_pnl < -self.initial_balance * self.config.daily_loss_limit_pct:
            truncated = True
            info["reason"] = "daily_loss_limit"
        
        # Hard cap on trades per episode (prevents runaway trading)
        if len(self.trade_history) >= self.reward_config.max_trades_per_episode:
            truncated = True
            info["reason"] = "max_trades_reached"

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment state."""
        super().reset(seed=seed)

        self.current_step = self.config.sequence_length
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0.0
        self.position_price = 0.0
        self.trade_history = []
        self.daily_pnl = 0.0
        self.max_equity = self.initial_balance
        
        # Reset new tracking variables
        self.prev_position_pct = 0.0
        self.position_entry_step = 0
        self.position_unrealized_pnl = 0.0
        self.total_transaction_costs = 0.0

        obs = self._get_observation()
        return obs, {}

    def render(self, mode: str = "human") -> None:
        """Render state summary for debugging."""
        print(
            f"Step: {self.current_step}, Equity: ${self.equity:.2f}, "
            f"Drawdown: {(self.max_equity - self.equity) / self.max_equity * 100:.1f}%, "
            f"Trades: {len(self.trade_history)}"
        )

