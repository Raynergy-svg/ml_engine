"""
RL-Based Position Sizing for ML Engine

Reinforcement Learning environment for dynamic position sizing using gymnasium.
Uses PPO from stable-baselines3 (optional dependency) to learn optimal position sizes
based on market conditions and ensemble model predictions.

Training Pipeline Order:
1. Optuna HPO (find best hyperparams for ensemble)
2. Full Ensemble Training (Transformer + XGBoost + RF + Ridge + HistGB)
3. RL Position Sizer Training (uses trained ensemble predictions as observations)

Usage:
    from rl_position_sizing import RLPositionSizer, TradingEnv
    
    # Train RL agent
    sizer = RLPositionSizer()
    sizer.train(df_features, ensemble_predictions, prices)
    
    # Use for inference
    position_size = sizer.get_position_size(features, ensemble_pred, account_equity=10000)

Dependencies:
    pip install gymnasium stable-baselines3  # Optional
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# Optional dependencies - can take 5-10s to import due to PyTorch
# ============================================================================
_import_start = time.perf_counter()

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    gym = None
    spaces = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    PPO = None
    BaseCallback = None
    SB3_AVAILABLE = False

_import_time = time.perf_counter() - _import_start
if _import_time > 5.0:
    logger.info(f"rl_position_sizing dependencies loaded in {_import_time:.1f}s (PyTorch is slow to import)")

# Model save path
RL_MODEL_PATH = Path("trained_data/models/rl_position_sizer.zip")
RL_SCALER_PATH = Path("trained_data/models/rl_scaler.pkl")


@dataclass
class RLConfig:
    """Configuration for RL position sizing."""
    # Environment
    sequence_length: int = 60  # Observation window
    max_position_pct: float = 0.05  # Max 5% of account per trade
    min_position_pct: float = 0.005  # Min 0.5% of account per trade
    
    # Reward shaping
    sharpe_weight: float = 0.1  # Weight for Sharpe ratio in reward
    drawdown_penalty: float = 0.5  # Penalty for drawdowns
    win_rate_bonus: float = 0.2  # Bonus for winning trades
    
    # Training
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95
    
    # Risk limits
    max_drawdown_pct: float = 0.10  # 10% max drawdown triggers reset
    daily_loss_limit_pct: float = 0.03  # 3% daily loss limit


class TradingEnv(gym.Env if GYM_AVAILABLE else object):
    """
    Gymnasium environment for RL-based position sizing.
    
    Observation Space:
        - Market features (from feature engineering)
        - Ensemble model predictions (direction probability, confidence)
        - Account state (equity, drawdown, recent P/L)
    
    Action Space:
        - Continuous: Position size as fraction of account [0, max_position_pct]
        - Or discrete: [0%, 0.5%, 1%, 2%, 3%, 5%] of account
    
    Reward:
        - Risk-adjusted returns (Sharpe-like)
        - Drawdown penalty
        - Win rate bonus
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        features: np.ndarray,
        ensemble_predictions: np.ndarray,
        prices: np.ndarray,
        config: Optional[RLConfig] = None,
    ):
        """
        Initialize trading environment.
        
        Args:
            features: Market features array (n_samples, n_features)
            ensemble_predictions: Ensemble predictions (n_samples, 2) - [direction_prob, confidence]
            prices: Close prices (n_samples,)
            config: RL configuration
        """
        if not GYM_AVAILABLE:
            raise ImportError("gymnasium is required. Install with: pip install gymnasium")
        
        super().__init__()
        
        self.config = config or RLConfig()
        self.features = features
        self.ensemble_predictions = ensemble_predictions
        self.prices = prices
        self.n_samples = len(prices)
        
        # Validate data
        assert len(features) == len(prices), "Features and prices must have same length"
        assert len(ensemble_predictions) == len(prices), "Predictions and prices must have same length"
        
        # State tracking
        self.current_step = self.config.sequence_length
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = 0.0  # Current position size (in lots)
        self.position_price = 0.0  # Entry price
        self.trade_history: List[Dict] = []
        self.daily_pnl = 0.0
        self.max_equity = self.initial_balance
        
        # Observation space: features + predictions + account state
        n_features = features.shape[1] if len(features.shape) > 1 else 1
        obs_dim = n_features + 2 + 4  # features + (direction, confidence) + (equity_pct, drawdown, daily_pnl, win_rate)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Action space: position size as percentage [0, max_position_pct]
        # Discretized into 6 levels for stability
        self.action_space = spaces.Discrete(6)  # [0%, 0.5%, 1%, 2%, 3%, 5%]
        self.position_levels = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        # Market features (use latest)
        if len(self.features.shape) > 1:
            market_features = self.features[self.current_step]
        else:
            market_features = np.array([self.features[self.current_step]])
        
        # Ensemble predictions
        pred = self.ensemble_predictions[self.current_step]
        direction_prob = pred[0] if len(pred) > 0 else 0.5
        confidence = pred[1] if len(pred) > 1 else 0.5
        
        # Account state
        equity_pct = self.equity / self.initial_balance - 1.0  # Relative to start
        drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0.0
        daily_pnl_pct = self.daily_pnl / self.initial_balance
        
        # Win rate from recent trades
        recent_trades = self.trade_history[-20:] if self.trade_history else []
        win_rate = sum(1 for t in recent_trades if t.get("pnl", 0) > 0) / max(len(recent_trades), 1)
        
        obs = np.concatenate([
            market_features.flatten(),
            [direction_prob, confidence],
            [equity_pct, drawdown, daily_pnl_pct, win_rate],
        ]).astype(np.float32)
        
        return obs
    
    def _calculate_reward(self, pnl: float) -> float:
        """
        Calculate shaped reward.
        
        Reward components:
        1. Raw P/L (normalized)
        2. Sharpe-like adjustment
        3. Drawdown penalty
        4. Win rate bonus
        """
        # Base reward: normalized P/L
        reward = pnl / self.initial_balance * 100  # Scale to percentage
        
        # Sharpe component: reward consistency
        if len(self.trade_history) > 5:
            returns = [t.get("pnl", 0) for t in self.trade_history[-20:]]
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns)
                reward += self.config.sharpe_weight * sharpe
        
        # Drawdown penalty
        drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0.0
        if drawdown > 0.05:  # Penalize drawdowns > 5%
            reward -= self.config.drawdown_penalty * drawdown * 100
        
        # Win rate bonus
        recent_trades = self.trade_history[-10:] if self.trade_history else []
        if recent_trades:
            win_rate = sum(1 for t in recent_trades if t.get("pnl", 0) > 0) / len(recent_trades)
            if win_rate > 0.5:
                reward += self.config.win_rate_bonus * (win_rate - 0.5) * 10
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step.
        
        Args:
            action: Position size level (0-5)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get position size from action
        position_pct = self.position_levels[action]
        
        # Get current and next price
        current_price = self.prices[self.current_step]
        self.current_step += 1
        
        if self.current_step >= self.n_samples - 1:
            # End of data
            obs = self._get_observation()
            return obs, 0.0, True, False, {"reason": "end_of_data"}
        
        next_price = self.prices[self.current_step]
        
        # Get ensemble prediction for direction
        pred = self.ensemble_predictions[self.current_step - 1]
        direction_prob = pred[0] if len(pred) > 0 else 0.5
        direction = 1 if direction_prob > 0.5 else -1  # 1 = long, -1 = short
        
        # Calculate position value
        position_value = self.equity * position_pct
        
        # Calculate P/L
        price_change_pct = (next_price - current_price) / current_price
        pnl = position_value * price_change_pct * direction
        
        # Update account
        self.equity += pnl
        self.daily_pnl += pnl
        self.max_equity = max(self.max_equity, self.equity)
        
        # Record trade
        if position_pct > 0:
            self.trade_history.append({
                "step": self.current_step,
                "direction": direction,
                "position_pct": position_pct,
                "entry_price": current_price,
                "exit_price": next_price,
                "pnl": pnl,
            })
        
        # Calculate reward
        reward = self._calculate_reward(pnl)
        
        # Check termination conditions
        terminated = False
        truncated = False
        info = {"pnl": pnl, "equity": self.equity}
        
        # Max drawdown check
        drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0.0
        if drawdown > self.config.max_drawdown_pct:
            terminated = True
            reward -= 10.0  # Big penalty for blowing up
            info["reason"] = "max_drawdown"
        
        # Daily loss limit
        if self.daily_pnl < -self.initial_balance * self.config.daily_loss_limit_pct:
            truncated = True
            info["reason"] = "daily_loss_limit"
        
        obs = self._get_observation()
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
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
        """Render environment state."""
        print(f"Step: {self.current_step}, Equity: ${self.equity:.2f}, "
              f"Drawdown: {(self.max_equity - self.equity) / self.max_equity * 100:.1f}%, "
              f"Trades: {len(self.trade_history)}")


class RLProgressCallback:
    """Callback that prints periodic RL training progress to console."""

    def __new__(cls, *args, **kwargs):
        # Only create if SB3 is available
        if BaseCallback is None:
            return None
        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        total_timesteps: int,
        print_freq: int = 50_000,
        console_print=None,
    ):
        if BaseCallback is None:
            return
        # Dynamically subclass BaseCallback
        self._total = total_timesteps
        self._print_freq = print_freq
        self._console_print = console_print or print
        self._last_printed = 0
        self._start_time = None
        self._inner = self._make_inner()

    def _make_inner(self):
        """Create the actual SB3 BaseCallback subclass instance."""
        outer = self

        class _Inner(BaseCallback):
            def __init__(self_inner):
                super().__init__(verbose=0)

            def _on_training_start(self_inner):
                import time as _t
                outer._start_time = _t.perf_counter()
                outer._console_print(
                    f"  ▶ PPO training started ({outer._total:,} timesteps)..."
                )

            def _on_step(self_inner) -> bool:
                import time as _t
                step = self_inner.num_timesteps
                if step - outer._last_printed >= outer._print_freq:
                    outer._last_printed = step
                    elapsed = _t.perf_counter() - outer._start_time if outer._start_time else 0
                    pct = 100 * step / outer._total
                    rate = step / elapsed if elapsed > 0 else 0
                    eta = (outer._total - step) / rate if rate > 0 else 0
                    outer._console_print(
                        f"  ⏳ {step:>8,} / {outer._total:,} ({pct:5.1f}%) "
                        f"| {rate:,.0f} steps/s | ETA {eta:.0f}s"
                    )
                return True

            def _on_training_end(self_inner):
                import time as _t
                elapsed = _t.perf_counter() - outer._start_time if outer._start_time else 0
                outer._console_print(
                    f"  ✔ PPO training finished in {elapsed:.1f}s"
                )

        return _Inner()

    def unwrap(self):
        """Return the actual SB3 BaseCallback for use in callbacks list."""
        return self._inner


class RLPositionSizer:
    """
    RL-based position sizer using PPO.
    
    Uses trained ensemble predictions + market features to determine
    optimal position size for each trade.
    """
    
    def __init__(self, config: Optional[RLConfig] = None):
        self.config = config or RLConfig()
        self.model = None  # PPO model (loaded lazily)
        self.scaler = None
        self._is_trained = False
    
    def train(
        self,
        features: np.ndarray,
        ensemble_predictions: np.ndarray,
        prices: np.ndarray,
        *,
        eval_freq: int = 10000,
        verbose: int = 1,
        callback: Any = None,
    ) -> Dict[str, Any]:
        """
        Train RL position sizer.
        
        Args:
            features: Market features (n_samples, n_features)
            ensemble_predictions: Predictions (n_samples, 2) - [direction_prob, confidence]
            prices: Close prices (n_samples,)
            eval_freq: Evaluation frequency
            verbose: Verbosity level
            callback: Optional callback for progress tracking
            
        Returns:
            Training statistics
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for RL training. "
                "Install with: pip install stable-baselines3"
            )
        
        logger.info("🤖 Training RL Position Sizer...")
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Create environment
        env = TradingEnv(
            features=features_scaled,
            ensemble_predictions=ensemble_predictions,
            prices=prices,
            config=self.config,
        )
        
        # Wrap in vectorized env
        vec_env = DummyVecEnv([lambda: env])
        
        # Create PPO model
        # CRITICAL: Use device='cpu' to avoid segfaults on macOS with TensorFlow + PyTorch
        logger.info("Creating PPO model (MlpPolicy, cpu device)...")
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            verbose=0,  # Suppress SB3 default output; we use our own callback
            device="cpu",  # Force CPU to avoid GPU conflicts with TensorFlow Metal
        )
        logger.info(f"PPO model created: lr={self.config.learning_rate}, batch={self.config.batch_size}, "
                     f"n_steps={self.config.n_steps}, n_epochs={self.config.n_epochs}")
        
        # Setup evaluation callback
        eval_env = DummyVecEnv([lambda: TradingEnv(
            features=features_scaled,
            ensemble_predictions=ensemble_predictions,
            prices=prices,
            config=self.config,
        )])
        
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5,
            min_evals=10,
            verbose=1,
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(RL_MODEL_PATH.parent),
            log_path=str(Path("trained_data/logs/rl")),
            eval_freq=eval_freq,
            callback_after_eval=stop_callback,
            deterministic=True,
            render=False,
        )
        
        # Build callback list
        callbacks = [eval_callback]
        if callback is not None:
            callbacks.append(callback)
        
        # Add progress callback for console visibility
        progress_cb = RLProgressCallback(
            total_timesteps=self.config.total_timesteps,
            print_freq=max(10_000, self.config.total_timesteps // 20),
        )
        if progress_cb is not None:
            callbacks.append(progress_cb.unwrap())
        
        # Train
        logger.info(f"Starting PPO learn() for {self.config.total_timesteps:,} timesteps...")
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks,
            progress_bar=False,  # Use our custom callback instead (tqdm may not flush)
        )
        
        self._is_trained = True
        
        # Save model and scaler
        self.save()
        
        # Get training stats
        stats = {
            "total_timesteps": self.config.total_timesteps,
            "final_equity": env.equity,
            "max_drawdown": (env.max_equity - env.equity) / env.max_equity if env.max_equity > 0 else 0,
            "total_trades": len(env.trade_history),
            "win_rate": sum(1 for t in env.trade_history if t.get("pnl", 0) > 0) / max(len(env.trade_history), 1),
        }
        
        logger.info(f"✅ RL training complete: {stats}")
        return stats
    
    def get_position_size(
        self,
        features: np.ndarray,
        ensemble_prediction: np.ndarray,
        account_equity: float = 10000.0,
    ) -> float:
        """
        Get optimal position size for current market state.
        
        Args:
            features: Current market features (n_features,)
            ensemble_prediction: Current prediction [direction_prob, confidence]
            account_equity: Current account equity
            
        Returns:
            Position size in dollars (not percentage)
        """
        if not self._is_trained or self.model is None:
            # Fallback to simple 1% rule
            logger.warning("RL model not trained, using default 1% position size")
            return account_equity * 0.01
        
        # Scale features
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features.reshape(1, -1)).flatten()
        else:
            features_scaled = features
        
        # Construct observation
        obs = np.concatenate([
            features_scaled,
            ensemble_prediction,
            [0.0, 0.0, 0.0, 0.5],  # Placeholder account state (neutral)
        ]).astype(np.float32)
        
        # Get action from model
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Convert action to position size
        position_levels = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
        position_pct = position_levels[int(action)]
        
        return account_equity * position_pct
    
    def save(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        """Save model and scaler."""
        model_path = model_path or RL_MODEL_PATH
        scaler_path = scaler_path or RL_SCALER_PATH
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            self.model.save(str(model_path))
            logger.info(f"💾 RL model saved to {model_path}")
        
        if self.scaler is not None:
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            logger.info(f"💾 RL scaler saved to {scaler_path}")
    
    def load(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None) -> bool:
        """Load model and scaler.
        
        When TensorFlow is already loaded in the process, PPO.load() can deadlock
        (known TF/PyTorch conflict on Intel Macs). In that case, we load the model
        in a subprocess and transfer state back via pickle.
        """
        model_path = model_path or RL_MODEL_PATH
        scaler_path = scaler_path or RL_SCALER_PATH
        
        if not SB3_AVAILABLE:
            logger.warning("stable-baselines3 not available, cannot load RL model")
            return False
        
        try:
            if model_path.exists():
                # Check if TF is loaded — if so, use subprocess to avoid deadlock
                import sys
                if 'tensorflow' in sys.modules:
                    self._load_via_subprocess(model_path)
                else:
                    # Force CPU to avoid GPU conflicts with TensorFlow Metal
                    self.model = PPO.load(str(model_path), device="cpu")
                    self._is_trained = True
                    logger.info(f"📂 RL model loaded from {model_path}")
            
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info(f"📂 RL scaler loaded from {scaler_path}")
            
            return self._is_trained
            
        except Exception as e:
            logger.warning(f"Failed to load RL model: {e}")
            return False
    
    def _load_via_subprocess(self, model_path: Path) -> None:
        """Load PPO model in a child process to avoid TF/PyTorch deadlock.
        
        The child process doesn't have TF imported, so PPO.load() works fine.
        We transfer the loaded model back via cloudpickle (handles lambdas).
        """
        import subprocess, tempfile, sys
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Use cloudpickle (SB3 dependency) since PPO contains unpicklable lambdas
            script = (
                "import sys\n"
                "try:\n"
                "    import cloudpickle\n"
                "    from stable_baselines3 import PPO\n"
                f'    model = PPO.load("{model_path}", device="cpu")\n'
                f'    with open("{tmp_path}", "wb") as f:\n'
                "        cloudpickle.dump(model, f)\n"
                "    sys.exit(0)\n"
                "except Exception as e:\n"
                '    print(f"RL subprocess load failed: {e}", file=sys.stderr)\n'
                "    sys.exit(1)\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=30,
                cwd=str(Path.cwd()),
            )
            
            if result.returncode == 0 and Path(tmp_path).exists():
                import cloudpickle
                with open(tmp_path, 'rb') as f:
                    self.model = cloudpickle.load(f)
                self._is_trained = True
                logger.info(f"📂 RL model loaded from {model_path} (via subprocess)")
            else:
                err = result.stderr.strip() if result.stderr else "unknown error"
                logger.warning(f"RL subprocess load failed: {err}")
        except subprocess.TimeoutExpired:
            logger.warning("RL subprocess load timed out (30s)")
        except Exception as e:
            logger.warning(f"RL subprocess load failed: {e}")
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
    
    @property
    def is_available(self) -> bool:
        """Check if RL position sizing is available."""
        return SB3_AVAILABLE and GYM_AVAILABLE


def train_rl_position_sizer(
    features: np.ndarray,
    ensemble_predictions: np.ndarray,
    prices: np.ndarray,
    config: Optional[RLConfig] = None,
) -> RLPositionSizer:
    """
    Convenience function to train RL position sizer.
    
    Args:
        features: Market features
        ensemble_predictions: Ensemble predictions [direction_prob, confidence]
        prices: Close prices
        config: RL configuration
        
    Returns:
        Trained RLPositionSizer
    """
    sizer = RLPositionSizer(config)
    sizer.train(features, ensemble_predictions, prices)
    return sizer


def get_position_sizer() -> RLPositionSizer:
    """
    Get RL position sizer, loading from disk if available.
    
    Returns:
        RLPositionSizer (trained if model exists, untrained otherwise)
    """
    sizer = RLPositionSizer()
    sizer.load()
    return sizer


if __name__ == "__main__":
    # Quick test
    if not GYM_AVAILABLE:
        print("❌ gymnasium not installed. Install with: pip install gymnasium")
    elif not SB3_AVAILABLE:
        print("⚠️ stable-baselines3 not installed. Install with: pip install stable-baselines3")
    else:
        print("✅ RL dependencies available")
        
        # Test with dummy data
        n_samples = 1000
        n_features = 20
        
        features = np.random.randn(n_samples, n_features)
        predictions = np.column_stack([
            np.random.uniform(0.4, 0.6, n_samples),  # direction_prob
            np.random.uniform(0.3, 0.7, n_samples),  # confidence
        ])
        prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
        
        # Create environment
        env = TradingEnv(features, predictions, prices)
        obs, info = env.reset()
        print(f"Observation shape: {obs.shape}")
        
        # Take a few steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                break
        
        print("✅ Environment test passed")
