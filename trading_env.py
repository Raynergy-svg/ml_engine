"""Trading environment for reinforcement learning"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None
    import logging

    logging.warning(
        "stable_baselines3 not installed. RL training will not be available."
    )


# ========= Reinforcement Learning Environment =========
class TradingEnv(gym.Env):
    """
    A trading environment for reinforcement learning.

    Observation:
      - A window of close prices (length = engine.sequence_length).
    Action space:
      - Discrete(3): 0 = hold, 1 = buy, 2 = sell.
    Reward:
      - Custom reward based on trade sequences.
    """

    def __init__(self, engine, data: pd.DataFrame):
        super(TradingEnv, self).__init__()
        self.engine = engine
        self.data = data
        self.current_step = engine.sequence_length
        self.max_steps = len(data) - engine.sequence_length - 1
        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(engine.sequence_length,),
            dtype=np.float32,
        )
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.holdings = 0
        self.net_worth = self.initial_balance
        self.reward_window = []
        self.trade_history = []
        self.last_trade_price = None
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.engine.sequence_length
        self.balance = self.initial_balance
        self.holdings = 0
        self.net_worth = self.initial_balance
        self.reward_window = []
        self.trade_history = []
        self.last_trade_price = None
        self.done = False
        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        start = max(self.current_step - self.engine.sequence_length, 0)
        end = self.current_step
        observation = self.engine.preprocess_data(self.data.iloc[start:end])[
            0
        ]  # Access the features part
        if observation is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.array(observation).flatten().astype(np.float32)

    def get_reward(self):
        if not self.trade_history:
            return 0  # No trades, no reward
        cumulative_return = self.net_worth / self.initial_balance
        sharpe_ratio = (
            np.mean(self.reward_window) / np.std(self.reward_window)
            if len(self.reward_window) > 1
            else 0
        )
        reward = cumulative_return + 0.1 * sharpe_ratio
        return reward

    def step(self, action):
        # Execute one time step within the environment
        self.execute_trade(action)
        self.current_step += 1
        observation = self._get_observation()
        reward = self.get_reward()
        self.reward_window.append(reward)
        truncated = False  # Add truncated variable
        terminated = self.done or self.current_step >= self.max_steps
        info = {"net_worth": self.net_worth, "balance": self.balance}
        return observation, reward, terminated, truncated, info

    def execute_trade(self, action):
        close_price = self.data["close"].iloc[self.current_step]
        trade_size = 1  # Number of shares to buy or sell
        cost = trade_size * close_price

        if action == 1:  # Buy
            if self.balance >= cost:
                self.balance -= cost
                self.holdings += trade_size
                self.last_trade_price = close_price
                self.trade_history.append(
                    {
                        "step": self.current_step,
                        "action": "buy",
                        "price": close_price,
                        "shares": trade_size,
                    }
                )
            else:
                self.done = True  # Insufficient funds
        elif action == 2:  # Sell
            if self.holdings > 0:
                self.balance += cost
                self.holdings -= trade_size
                self.last_trade_price = close_price
                self.trade_history.append(
                    {
                        "step": self.current_step,
                        "action": "sell",
                        "price": close_price,
                        "shares": trade_size,
                    }
                )
            else:
                self.done = True  # No holdings to sell

        self.net_worth = self.balance + self.holdings * close_price
        if self.net_worth <= 0:
            self.done = True  # Bankrupt


def train_rl_agent(engine, data: pd.DataFrame):
    """
    Train a reinforcement learning agent using PPO on the
    trading environment.
    """
    if PPO is None:
        raise ImportError(
            "stable_baselines3 is required for RL training. "
            "Install with: pip install stable-baselines3"
        )

    env = TradingEnv(engine, data)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    return model
# — Raynergy-svg —
