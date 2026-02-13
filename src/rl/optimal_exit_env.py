"""
Optimal Exit Timing RL Environment.

PPO-based environment for learning when to exit trades instead of
using fixed take-profit/stop-loss ratios.

Uses discrete action space:
- HOLD: Continue holding position
- EXIT_PROFIT: Close position for profit target
- EXIT_LOSS: Close position for stop loss
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Lazy imports for gymnasium and stable-baselines3 to avoid 8+ second startup penalty
GYM_AVAILABLE = None  # Will be set on first access
SB3_AVAILABLE = None  # Will be set on first access
gym = None
spaces = None
PPO = None
EvalCallback = None
DummyVecEnv = None
Monitor = None


def _ensure_gym_imported():
    """Lazy import gymnasium only when needed."""
    global gym, spaces, GYM_AVAILABLE
    if GYM_AVAILABLE is None:
        try:
            import gymnasium as _gym
            from gymnasium import spaces as _spaces
            gym = _gym
            spaces = _spaces
            GYM_AVAILABLE = True
        except ImportError:
            GYM_AVAILABLE = False
    return GYM_AVAILABLE


def _ensure_sb3_imported():
    """Lazy import stable-baselines3 only when needed."""
    global PPO, EvalCallback, DummyVecEnv, Monitor, SB3_AVAILABLE
    if SB3_AVAILABLE is None:
        try:
            from stable_baselines3 import PPO as _PPO
            from stable_baselines3.common.callbacks import EvalCallback as _EvalCallback
            from stable_baselines3.common.vec_env import DummyVecEnv as _DummyVecEnv
            from stable_baselines3.common.monitor import Monitor as _Monitor
            PPO = _PPO
            EvalCallback = _EvalCallback
            DummyVecEnv = _DummyVecEnv
            Monitor = _Monitor
            SB3_AVAILABLE = True
        except ImportError:
            SB3_AVAILABLE = False
    return SB3_AVAILABLE


logger = logging.getLogger(__name__)

# Model paths
EXIT_RL_MODEL_PATH = Path("trained_data/models/ppo_optimal_exit.zip")
EXIT_RL_SCALER_PATH = Path("trained_data/models/exit_rl_scaler.pkl")


# Action constants
class ExitAction:
    HOLD = 0
    EXIT_PROFIT = 1
    EXIT_LOSS = 2


@dataclass
class ExitRLConfig:
    """Configuration for Optimal Exit RL."""
    # Default trade parameters
    default_stop_loss_pct: float = 0.01  # 1% stop loss
    default_take_profit_pct: float = 0.02  # 2% take profit
    max_bars_in_trade: int = 48  # Max 48 bars (2 days on H1)

    # Reward shaping
    holding_cost: float = 0.001  # Small penalty per bar held
    early_exit_bonus: float = 0.1  # Bonus for profitable early exit
    late_exit_penalty: float = 0.05  # Penalty for exiting too late

    # Training
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Episode settings
    max_trades_per_episode: int = 100

    # Network
    net_arch: List[int] = field(default_factory=lambda: [64, 64])


@dataclass
class TradeScenario:
    """A single trade scenario for the environment."""
    entry_price: float
    direction: int  # 1=long, -1=short
    prices: np.ndarray  # Future prices after entry
    momentum: np.ndarray  # Momentum indicator values
    atr: np.ndarray  # ATR values for volatility context
    optimal_exit_bar: int = -1  # Ground truth optimal exit (if known)


def _get_gym_env_base():
    """Get gym.Env base class lazily."""
    _ensure_gym_imported()
    if GYM_AVAILABLE and gym is not None:
        return gym.Env
    return object


class OptimalExitEnv(_get_gym_env_base()):
    """
    Gymnasium environment for learning optimal trade exit timing.

    State Space (4-dim Box):
        - unrealized_pnl: Current P/L in pips
        - time_in_trade: Bars since entry (normalized)
        - momentum: Current momentum indicator
        - atr_normalized: Current ATR (volatility context)

    Action Space (Discrete):
        - 0: HOLD (continue position)
        - 1: EXIT_PROFIT (close with profit target)
        - 2: EXIT_LOSS (close with stop loss)

    Reward:
        - Realized P/L at exit
        - Small negative reward per bar held (opportunity cost)
        - Bonus/penalty for timing quality
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenarios: List[TradeScenario],
        config: Optional[ExitRLConfig] = None,
    ):
        """
        Initialize optimal exit environment.

        Args:
            scenarios: List of trade scenarios to learn from
            config: RL configuration
        """
        _ensure_gym_imported()
        if not GYM_AVAILABLE:
            raise ImportError("gymnasium required. Install: pip install gymnasium")

        super().__init__()

        self.config = config or ExitRLConfig()
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        assert self.n_scenarios > 0, "Need at least one trade scenario"

        # Observation space: [pnl, time, momentum, atr]
        self.observation_space = spaces.Box(
            low=np.array([-100, 0, -1, 0], dtype=np.float32),
            high=np.array([100, 1, 1, 5], dtype=np.float32),
            dtype=np.float32
        )

        # Action space: discrete exit decisions
        self.action_space = spaces.Discrete(3)

        # Episode tracking
        self.scenario_idx = 0
        self.current_scenario: Optional[TradeScenario] = None
        self.step_in_trade = 0
        self.position_open = True
        self.episode_trades = 0
        self.episode_pnl = 0.0
        self.trade_results: List[Dict] = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment with new scenario."""
        super().reset(seed=seed)

        # Select random scenario
        self.scenario_idx = self.np_random.integers(0, self.n_scenarios)
        self.current_scenario = self.scenarios[self.scenario_idx]

        self.step_in_trade = 0
        self.position_open = True
        self.episode_trades = 0
        self.episode_pnl = 0.0
        self.trade_results = []

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        if self.current_scenario is None or not self.position_open:
            return np.zeros(4, dtype=np.float32)

        scenario = self.current_scenario
        idx = min(self.step_in_trade, len(scenario.prices) - 1)

        current_price = scenario.prices[idx]
        pnl_pips = (current_price - scenario.entry_price) / scenario.entry_price
        pnl_pips *= scenario.direction * 10000  # Convert to pips

        # Normalize time in trade to [0, 1]
        time_normalized = self.step_in_trade / self.config.max_bars_in_trade

        # Momentum and ATR
        momentum = scenario.momentum[idx] if idx < len(scenario.momentum) else 0.0
        atr = scenario.atr[idx] if idx < len(scenario.atr) else 0.01

        obs = np.array([
            np.clip(pnl_pips, -100, 100),
            np.clip(time_normalized, 0, 1),
            np.clip(momentum, -1, 1),
            np.clip(atr, 0, 5),
        ], dtype=np.float32)

        return obs

    def _calculate_pnl(self) -> float:
        """Calculate current unrealized P/L."""
        if self.current_scenario is None:
            return 0.0

        scenario = self.current_scenario
        idx = min(self.step_in_trade, len(scenario.prices) - 1)
        current_price = scenario.prices[idx]

        pnl_pct = (current_price - scenario.entry_price) / scenario.entry_price
        pnl_pct *= scenario.direction

        return pnl_pct

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step.

        Args:
            action: 0=HOLD, 1=EXIT_PROFIT, 2=EXIT_LOSS

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.current_scenario is None or not self.position_open:
            # Position already closed, start new trade
            return self._start_new_trade()

        scenario = self.current_scenario
        reward = 0.0
        terminated = False
        truncated = False
        info = {'action': action}

        # REWARD SCALING: Use tanh for soft clipping and better gradient flow
        # PnL is typically 0.1-2%, base scale by 1000, then tanh compress
        BASE_SCALE = 1000.0

        if action == ExitAction.HOLD:
            # Continue holding - adaptive holding cost based on position state
            pnl_so_far = self._calculate_pnl()
            pnl_pips = pnl_so_far * 10000 if scenario.entry_price < 10 else pnl_so_far * 100

            # Dynamic holding cost:
            # - Very low cost if trade is profitable (let winners run)
            # - Higher cost if trade is negative (encourage cutting losses)
            if pnl_pips > 0:
                reward = -0.001 * (1 + self.step_in_trade * 0.01)  # Slight time decay
            else:
                reward = -0.003 * (1 + abs(pnl_pips) * 0.1)  # Growing cost for losers

            self.step_in_trade += 1

            # Check if max bars reached
            if self.step_in_trade >= min(self.config.max_bars_in_trade, len(scenario.prices) - 1):
                # Forced exit at end
                pnl = self._calculate_pnl()
                # Use tanh for soft clipping, scale to roughly [-10, 10]
                reward = 10.0 * np.tanh(pnl * BASE_SCALE / 10.0)
                reward -= self.config.late_exit_penalty * 5  # Penalty for forced exit
                self._close_position(pnl, 'max_bars')
                info['exit_reason'] = 'max_bars'
                info['pnl'] = pnl

        else:
            # Exit position
            pnl = self._calculate_pnl()
            # Use tanh for soft clipping, scale to roughly [-10, 10]
            reward = 10.0 * np.tanh(pnl * BASE_SCALE / 10.0)

            exit_reason = 'profit_target' if action == ExitAction.EXIT_PROFIT else 'stop_loss'

            # === Timing bonuses (in comparable scale to base reward) ===
            # Bonus for profitable early exit (lock in gains)
            if pnl > 0 and self.step_in_trade < self.config.max_bars_in_trade * 0.3:
                reward += 0.5 * self.config.early_exit_bonus  # ~0.5-1.0 bonus

            # Bonus for cutting losses quickly
            if pnl < 0:
                # Bigger bonus for faster cuts
                speed_factor = max(0, (10 - self.step_in_trade) / 10)
                reward += 0.3 * speed_factor  # Up to 0.3 bonus for fast cut

            # Penalty for holding a loser too long
            if pnl < 0 and self.step_in_trade > 15:
                reward -= 0.2 * (self.step_in_trade - 15) / 10  # Growing penalty

            self._close_position(pnl, exit_reason)
            info['exit_reason'] = exit_reason
            info['pnl'] = pnl
            info['bars_held'] = self.step_in_trade

        # Check episode termination
        if self.episode_trades >= self.config.max_trades_per_episode:
            terminated = True
            info['episode_pnl'] = self.episode_pnl
            info['episode_trades'] = self.episode_trades

        # If position closed but episode continues, start new trade
        # BUT preserve the exit reward from the completed trade
        if not self.position_open and not terminated:
            # Select new scenario for next trade
            self.scenario_idx = self.np_random.integers(0, self.n_scenarios)
            self.current_scenario = self.scenarios[self.scenario_idx]
            self.step_in_trade = 0
            self.position_open = True
            info['new_trade_started'] = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _close_position(self, pnl: float, reason: str):
        """Close current position and record result."""
        self.position_open = False
        self.episode_trades += 1
        self.episode_pnl += pnl

        self.trade_results.append({
            'scenario_idx': self.scenario_idx,
            'pnl': pnl,
            'bars_held': self.step_in_trade,
            'exit_reason': reason,
        })

    def _start_new_trade(self) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Start a new trade within the episode."""
        # Select new scenario
        self.scenario_idx = self.np_random.integers(0, self.n_scenarios)
        self.current_scenario = self.scenarios[self.scenario_idx]
        self.step_in_trade = 0
        self.position_open = True

        obs = self._get_obs()
        info = {'new_trade': True, 'scenario_idx': self.scenario_idx}

        return obs, 0.0, False, False, info

    def render(self, mode: str = "human") -> None:
        """Render current state."""
        if self.current_scenario is None:
            print("No active trade")
            return

        pnl = self._calculate_pnl()
        print(f"Trade #{self.episode_trades}, Step: {self.step_in_trade}, "
              f"P/L: {pnl*100:.2f}%, Direction: {self.current_scenario.direction}")


class OptimalExitRL:
    """
    RL-based optimal exit timing using PPO.

    Learns when to exit trades to maximize returns.
    """

    def __init__(self, config: Optional[ExitRLConfig] = None):
        self.config = config or ExitRLConfig()
        self.model = None
        self.scaler = None
        self._is_trained = False

    @staticmethod
    def create_scenarios_from_data(
        prices: np.ndarray,
        directions: np.ndarray,
        momentum: np.ndarray,
        atr: np.ndarray,
        lookahead: int = 48,
    ) -> List[TradeScenario]:
        """
        Create trade scenarios from historical data.

        Args:
            prices: Price array
            directions: Trade direction signals (1=long, -1=short)
            momentum: Momentum indicator array
            atr: ATR array
            lookahead: Number of bars to include after entry

        Returns:
            List of TradeScenario objects
        """
        scenarios = []
        n_samples = len(prices)

        for i in range(n_samples - lookahead):
            direction = int(directions[i])
            if direction == 0:
                continue

            scenario = TradeScenario(
                entry_price=float(prices[i]),
                direction=direction,
                prices=prices[i:i+lookahead].copy(),
                momentum=momentum[i:i+lookahead].copy(),
                atr=atr[i:i+lookahead].copy(),
            )
            scenarios.append(scenario)

        logger.info(f"Created {len(scenarios)} trade scenarios from {n_samples} samples")
        return scenarios

    def train(
        self,
        scenarios: Optional[List[TradeScenario]] = None,
        timesteps: Optional[int] = None,
        eval_freq: int = 5000,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """
        Train PPO agent for optimal exit timing.

        Args:
            scenarios: List of trade scenarios (or pass via scenarios kwarg as numpy array)
            timesteps: Total training timesteps (overrides config)
            eval_freq: Evaluation frequency
            verbose: Verbosity level

        Returns:
            Training statistics
        """
        _ensure_sb3_imported()
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required: pip install stable-baselines3")

        # Import here to avoid slow startup
        from stable_baselines3.common.vec_env import DummyVecEnv

        total_timesteps = timesteps or self.config.total_timesteps
        logger.info(f"⏱️ Training Optimal Exit RL (PPO) for {total_timesteps:,} timesteps...")

        # If no scenarios provided, generate synthetic ones
        if scenarios is None:
            logger.info("Generating synthetic scenarios...")
            scenarios = self._generate_synthetic_scenarios(2000)  # More scenarios for diversity

        # Create environment
        env = OptimalExitEnv(scenarios=scenarios, config=self.config)
        vec_env = DummyVecEnv([lambda: env])

        # Create PPO model
        # CRITICAL: CPU only to avoid Metal/TensorFlow conflicts
        policy_kwargs = dict(net_arch=self.config.net_arch)

        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            ent_coef=0.05,  # Higher entropy for more exploration
            clip_range=0.3,  # Larger clip range for more policy updates
            policy_kwargs=policy_kwargs,
            device="cpu",  # CRITICAL
            verbose=verbose,
        )

        # Evaluation environment with Monitor wrapper for proper episode tracking
        log_dir = Path("trained_data/logs/rl_exits")
        log_dir.mkdir(parents=True, exist_ok=True)
        eval_env = DummyVecEnv([lambda: Monitor(
            OptimalExitEnv(scenarios=scenarios, config=self.config),
            str(log_dir / "eval_monitor")
        )])

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(EXIT_RL_MODEL_PATH.parent),
            log_path=str(Path("trained_data/logs/rl_exits")),
            eval_freq=eval_freq,
            n_eval_episodes=10,
            deterministic=True,
        )

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )

        # Load the BEST model saved by EvalCallback (not the final one)
        best_model_path = EXIT_RL_MODEL_PATH.parent / "best_model.zip"
        if best_model_path.exists():
            logger.info(f"📥 Loading best Exit RL model from {best_model_path}")
            self.model = PPO.load(best_model_path, env=vec_env, device="cpu")

        self._is_trained = True
        self.save()

        # Calculate stats
        total_trades = len(env.trade_results)
        win_rate = sum(1 for t in env.trade_results if t['pnl'] > 0) / max(total_trades, 1)
        avg_pnl = np.mean([t['pnl'] for t in env.trade_results]) if env.trade_results else 0.0
        avg_bars = np.mean([t['bars_held'] for t in env.trade_results]) if env.trade_results else 0.0

        stats = {
            'total_timesteps': total_timesteps,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl_pct': avg_pnl * 100,
            'avg_bars_held': avg_bars,
        }

        logger.info(f"✅ Optimal Exit RL training complete: {stats}")
        return stats

    def _generate_synthetic_scenarios(self, n_scenarios: int = 1000) -> List[TradeScenario]:
        """Generate synthetic trade scenarios for training with realistic FX volatility."""
        scenarios = []
        for _ in range(n_scenarios):
            # More realistic FX volatility: ~0.5% daily std for H1 candles
            # H1 volatility ~ daily_vol / sqrt(24) ~ 0.1% per bar
            n_steps = 50

            # Generate trending or mean-reverting scenarios
            scenario_type = np.random.choice(['trend_up', 'trend_down', 'mean_revert', 'choppy'])

            if scenario_type == 'trend_up':
                drift = 0.0003  # Upward trend
                vol = 0.002
            elif scenario_type == 'trend_down':
                drift = -0.0003  # Downward trend
                vol = 0.002
            elif scenario_type == 'mean_revert':
                drift = 0.0
                vol = 0.003  # Higher vol, mean reverts
            else:  # choppy
                drift = 0.0
                vol = 0.004  # High vol, no trend

            returns = np.random.normal(drift, vol, n_steps)

            # Add momentum for trends
            if scenario_type in ['trend_up', 'trend_down']:
                momentum_factor = np.linspace(0.5, 1.5, n_steps) if scenario_type == 'trend_up' else np.linspace(1.5, 0.5, n_steps)
                returns *= momentum_factor / momentum_factor.mean()

            prices = 1.0 * np.cumprod(1 + returns)  # Start at 1.0 for easier scaling

            # Momentum: rolling return
            momentum = np.zeros(n_steps)
            for i in range(5, n_steps):
                momentum[i] = (prices[i] / prices[i-5] - 1) * 100  # 5-bar momentum as percentage

            # ATR proxy: rolling std of returns
            atr = np.zeros(n_steps)
            for i in range(5, n_steps):
                atr[i] = np.std(returns[i-5:i]) * np.sqrt(24)  # Annualized-ish
            atr = np.clip(atr, 0.001, 0.05)

            # Direction: match the scenario type for learning signal
            if scenario_type == 'trend_up':
                direction = 1  # Long in uptrend
            elif scenario_type == 'trend_down':
                direction = -1  # Short in downtrend
            else:
                direction = 1 if np.random.random() > 0.5 else -1

            scenario = TradeScenario(
                entry_price=float(prices[0]),
                prices=prices,
                direction=direction,
                momentum=momentum,
                atr=atr,
            )
            scenarios.append(scenario)

        logger.info(f"Generated {n_scenarios} synthetic scenarios with realistic FX volatility")
        return scenarios

    def get_exit_decision(
        self,
        unrealized_pnl_pips: float,
        bars_in_trade: int,
        momentum: float,
        atr: float,
    ) -> Tuple[int, float]:
        """
        Get exit decision for current trade state.

        Args:
            unrealized_pnl_pips: Current P/L in pips
            bars_in_trade: Number of bars since entry
            momentum: Current momentum
            atr: Current ATR

        Returns:
            Tuple of (action, confidence)
        """
        if not self._is_trained or self.model is None:
            # Default: hold if profitable, exit if loss
            if unrealized_pnl_pips > 20:
                return ExitAction.EXIT_PROFIT, 0.6
            elif unrealized_pnl_pips < -15:
                return ExitAction.EXIT_LOSS, 0.6
            return ExitAction.HOLD, 0.5

        # Build observation
        time_normalized = bars_in_trade / self.config.max_bars_in_trade
        obs = np.array([
            np.clip(unrealized_pnl_pips, -100, 100),
            np.clip(time_normalized, 0, 1),
            np.clip(momentum, -1, 1),
            np.clip(atr, 0, 5),
        ], dtype=np.float32)

        # Get action
        action, _ = self.model.predict(obs, deterministic=True)

        # Get action probabilities for confidence
        # Note: PPO doesn't directly expose probs, so we estimate
        confidence = 0.7 if action != ExitAction.HOLD else 0.5

        return int(action), confidence

    def save(self):
        """Save model."""
        EXIT_RL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save(str(EXIT_RL_MODEL_PATH))
            logger.info(f"💾 Exit RL model saved to {EXIT_RL_MODEL_PATH}")

    def load(self, force_direct: bool = False) -> bool:
        """Load model.

        When TensorFlow is already loaded, uses subprocess to avoid
        TF/PyTorch deadlock on macOS (Intel and Metal).

        Args:
            force_direct: If True, skip the subprocess fallback and load
                directly with PPO.load(). Use when already in a child process
                (e.g. rl_worker) where the TF import is just an SB3 side-effect.
        """
        _ensure_sb3_imported()
        if not SB3_AVAILABLE:
            logger.debug("stable-baselines3 not available")
            return False

        try:
            if EXIT_RL_MODEL_PATH.exists():
                import sys
                if not force_direct and 'tensorflow' in sys.modules:
                    self._load_via_subprocess(EXIT_RL_MODEL_PATH)
                else:
                    self.model = PPO.load(str(EXIT_RL_MODEL_PATH), device="cpu")
                    self._is_trained = True
                    logger.info(f"📂 Exit RL model loaded from {EXIT_RL_MODEL_PATH}")
            return self._is_trained
        except Exception as e:
            logger.warning(f"Failed to load Exit RL model: {e}")
            return False

    def _load_via_subprocess(self, model_path: Path) -> None:
        """Load PPO model in a child process to avoid TF/PyTorch deadlock.

        The child process doesn't have TF imported, so PPO.load() works fine.
        We transfer the loaded model back via cloudpickle (handles lambdas).
        """
        import subprocess
        import tempfile
        import sys

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
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
                '    print(f"Exit RL subprocess load failed: {e}", file=sys.stderr)\n'
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
                logger.info(f"📂 Exit RL model loaded from {model_path} (via subprocess)")
            else:
                err = result.stderr.strip() if result.stderr else "unknown error"
                logger.warning(f"Exit RL subprocess load failed: {err}")
        except subprocess.TimeoutExpired:
            logger.warning("Exit RL subprocess load timed out (30s)")
        except Exception as e:
            logger.warning(f"Exit RL subprocess load failed: {e}")
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
