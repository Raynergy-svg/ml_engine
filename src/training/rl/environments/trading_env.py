"""Gymnasium trading environment for PPO-based position sizing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.training.rl import runtime

# Discrete position size levels (percentage of equity).
POSITION_LEVELS = [0.0, 0.01, 0.03, 0.05, 0.07, 0.10]


def _get_gym_env_base():
    """Get gym.Env base class lazily."""
    runtime._ensure_gym_imported()
    if runtime.GYM_AVAILABLE and runtime.gym is not None:
        return runtime.gym.Env
    return object


class TradingEnv(_get_gym_env_base()):
    """Environment for RL-based position sizing."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        ensemble_predictions: np.ndarray,
        prices: np.ndarray,
        config: Optional[Any] = None,
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

    def _calculate_reward(self, pnl: float) -> float:
        """Calculate shaped reward from PnL, consistency, and risk."""
        reward = pnl / self.initial_balance * 100

        if len(self.trade_history) > 5:
            returns = [t.get("pnl", 0) for t in self.trade_history[-20:]]
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns)
                reward += self.config.sharpe_weight * sharpe

        drawdown = (
            (self.max_equity - self.equity) / self.max_equity
            if self.max_equity > 0
            else 0.0
        )
        if drawdown > 0.05:
            reward -= self.config.drawdown_penalty * drawdown * 100

        recent_trades = self.trade_history[-10:] if self.trade_history else []
        if recent_trades:
            win_rate = (
                sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
                / len(recent_trades)
            )
            if win_rate > 0.5:
                reward += self.config.win_rate_bonus * (win_rate - 0.5) * 10

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        position_pct = self.position_levels[action]

        current_price = self.prices[self.current_step]
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

        self.equity += pnl
        self.daily_pnl += pnl
        self.max_equity = max(self.max_equity, self.equity)

        if position_pct > 0:
            self.trade_history.append(
                {
                    "step": self.current_step,
                    "direction": direction,
                    "position_pct": position_pct,
                    "entry_price": current_price,
                    "exit_price": next_price,
                    "pnl": pnl,
                }
            )

        reward = self._calculate_reward(pnl)

        terminated = False
        truncated = False
        info: Dict[str, Any] = {"pnl": pnl, "equity": self.equity}

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

        obs = self._get_observation()
        return obs, {}

    def render(self, mode: str = "human") -> None:
        """Render state summary for debugging."""
        print(
            f"Step: {self.current_step}, Equity: ${self.equity:.2f}, "
            f"Drawdown: {(self.max_equity - self.equity) / self.max_equity * 100:.1f}%, "
            f"Trades: {len(self.trade_history)}"
        )

