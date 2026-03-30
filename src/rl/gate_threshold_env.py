"""
Gate Threshold RL Environment.

SAC-based environment for learning optimal gate thresholds dynamically
based on market regime, recent performance, and account state.

Uses continuous action space to adjust:
- Confidence threshold (Ridge ADX gate)
- Momentum threshold (XGBoost gate)
- Risk threshold (RandomForest drawdown gate)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- numpy 1.x ↔ 2.x pickle compatibility shim ---
# Models saved with numpy 2.x reference `numpy._core.numeric` internally.
# numpy 1.x uses `numpy.core.numeric` (no underscore). This alias lets
# pickle find the module during deserialization regardless of numpy version.
import sys as _sys
if not hasattr(np, "_core"):
    import numpy.core as _np_core
    _sys.modules.setdefault("numpy._core", _np_core)
    for _sub in ("numeric", "multiarray", "_multiarray_umath", "fromnumeric"):
        _mod = getattr(_np_core, _sub, None)
        if _mod is not None:
            _sys.modules.setdefault(f"numpy._core.{_sub}", _mod)

# Lazy imports for gymnasium and stable-baselines3 to avoid 8+ second startup penalty
EvalCallback = None  # Will be imported lazily with SB3
GYM_AVAILABLE = None  # Will be set on first access
SB3_AVAILABLE = None  # Will be set on first access
gym = None
spaces = None
SAC = None


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
        except ImportError as e:
            GYM_AVAILABLE = False
            import logging
            logging.getLogger(__name__).warning(
                f"gymnasium not available: {e}. "
                "Install with: pip install gymnasium"
            )
    return GYM_AVAILABLE


def _ensure_sb3_imported():
    """Lazy import stable-baselines3 only when needed."""
    global SAC, EvalCallback, SB3_AVAILABLE
    if SB3_AVAILABLE is None:
        try:
            from stable_baselines3 import SAC as _SAC
            from stable_baselines3.common.callbacks import EvalCallback as _EvalCallback
            SAC = _SAC
            EvalCallback = _EvalCallback
            SB3_AVAILABLE = True
        except ImportError as e:
            SB3_AVAILABLE = False
            import logging
            logging.getLogger(__name__).warning(
                f"stable-baselines3 not available: {e}. "
                "Install with: pip install stable-baselines3"
            )
    return SB3_AVAILABLE


from .utils import detect_regime  # noqa: E402

logger = logging.getLogger(__name__)

# Model paths
GATE_RL_MODEL_PATH = Path("trained_data/models/sac_gate_thresholds.zip")
GATE_RL_SCALER_PATH = Path("trained_data/models/gate_rl_scaler.pkl")


def extract_observation_space_from_model(model_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract observation space bounds from saved SB3 model without full load.

    Args:
        model_path: Path to the saved .zip model file

    Returns:
        Tuple of (low, high) arrays or None if extraction fails
    """
    import zipfile
    import json

    try:
        with zipfile.ZipFile(model_path, 'r') as zf:
            if 'data' not in zf.namelist():
                return None
            data = json.loads(zf.read('data').decode('utf-8'))
            if 'observation_space' not in data:
                return None
            obs = data['observation_space']
            # Parse string arrays like '[1.  1.  1.  1.  0.5]'
            low = np.fromstring(obs['low'].strip('[]'), sep=' ', dtype=np.float32)
            high = np.fromstring(obs['high'].strip('[]'), sep=' ', dtype=np.float32)
            return low, high
    except Exception as e:
        logger.warning(f"Failed to extract observation space from {model_path}: {e}")
        return None


def infer_config_from_model(model_path: Path) -> Optional[Dict[str, float]]:
    """
    Infer GateRLConfig parameters from saved model's observation space.

    The observation space high values are: [1, 1, 1, 1, max_drawdown_pct]

    Args:
        model_path: Path to the saved .zip model file

    Returns:
        Dictionary of inferred config parameters or None if inference fails
    """
    bounds = extract_observation_space_from_model(model_path)
    if bounds is None:
        return None
    low, high = bounds
    # high = [1, 1, 1, 1, max_drawdown_pct]
    if len(high) >= 5:
        return {'max_drawdown_pct': float(high[4])}
    return None


@dataclass
class GateRLConfig:
    """Configuration for Gate Threshold RL."""
    # Base thresholds (from config_improved_H1.yaml)
    base_confidence: float = 50.0
    base_momentum: float = 0.20
    base_risk: float = 0.025

    # Threshold adjustment ranges
    confidence_delta_range: float = 0.15  # ±15 points (35-65)
    momentum_delta_range: float = 0.10   # ±0.10 (0.10-0.30)
    risk_delta_range: float = 0.015      # ±1.5% (1%-4%)

    # Reward shaping
    sharpe_weight: float = 0.15
    drawdown_penalty: float = 0.8
    win_rate_bonus: float = 0.25
    overtrade_penalty: float = 0.05  # Penalty per trade above threshold
    
    # Transaction costs (CRITICAL for preventing excessive trading)
    spread_cost_pct: float = 0.0002  # 0.02% spread (2 pips for FX)
    commission_pct: float = 0.0001   # 0.01% commission
    slippage_pct: float = 0.0001     # 0.01% slippage
    
    # Asymmetric reward scaling
    profit_multiplier: float = 1.5   # Bonus for profitable trades
    loss_multiplier: float = 2.0     # Extra penalty for losing trades

    # Training
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    buffer_size: int = 100_000  # Larger replay buffer for more diverse experiences
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    ent_coef: float = 0.1  # Fixed entropy coefficient (no auto-tuning)
    learning_starts: int = 10_000  # More random exploration before learning

    # Episode limits
    max_drawdown_pct: float = 0.10
    max_trades_per_episode: int = 200

    # Network architecture
    net_arch_pi: List[int] = field(default_factory=lambda: [64, 64])
    net_arch_qf: List[int] = field(default_factory=lambda: [256, 256])

    # Parallelization
    n_envs: int = 4  # Number of parallel environments (1 = no parallelization)


def _get_gym_env_base():
    """Get gym.Env base class lazily."""
    _ensure_gym_imported()
    if GYM_AVAILABLE and gym is not None:
        return gym.Env
    return object


class GateThresholdEnv(_get_gym_env_base()):
    """
    Gymnasium environment for learning optimal gate thresholds.

    State Space (5-dim Box):
        - regime_trend: 1 if TREND regime, else 0
        - regime_chop: 1 if CHOP regime, else 0
        - regime_mean_revert: 1 if MEAN_REVERT regime, else 0
        - recent_win_rate: Rolling 20-trade win rate
        - current_drawdown: % from peak equity

    Action Space (3-dim Box, continuous):
        - Δ confidence_threshold: scaled to [-delta, +delta]
        - Δ momentum_threshold: scaled to [-delta, +delta]
        - Δ risk_threshold: scaled to [-delta, +delta]

    Reward:
        - Trade P/L (normalized)
        - Sharpe component
        - Drawdown penalty
        - Win rate bonus
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        ensemble_preds: np.ndarray,
        config: Optional[GateRLConfig] = None,
        directions: Optional[np.ndarray] = None,
    ):
        """
        Initialize gate threshold environment.

        Args:
            prices: Close prices (n_samples,)
            features: Market features (n_samples, n_features)
            ensemble_preds: Ensemble predictions (n_samples, 4)
                           [tcn_prob, ridge_conf, xgb_mom, rf_drawdown]
            config: RL configuration
            directions: Direction signals (-1, 0, 1) for reward shaping
        """
        _ensure_gym_imported()
        if not GYM_AVAILABLE:
            raise ImportError("gymnasium required. Install: pip install gymnasium")

        super().__init__()

        self.config = config or GateRLConfig()
        self.prices = prices
        self.features = features
        self.ensemble_preds = ensemble_preds
        self.n_samples = len(prices)

        # Direction signals for reward shaping (derive from TCN if not provided)
        if directions is not None:
            self.directions = directions
        else:
            # Infer from TCN probability
            tcn_prob = ensemble_preds[:, 0] if ensemble_preds.shape[1] > 0 else np.full(len(prices), 0.5)
            self.directions = np.where(tcn_prob > 0.58, 1, np.where(tcn_prob < 0.42, -1, 0))

        # Validate data
        assert len(features) == len(prices), "Features and prices must match"
        assert len(ensemble_preds) == len(prices), "Predictions and prices must match"

        # Observation space: [regime(3), win_rate(1), drawdown(1)] = 5 dims
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, self.config.max_drawdown_pct], dtype=np.float32),
            dtype=np.float32
        )

        # Action space: threshold deltas (continuous, normalized to [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # State tracking
        self.current_step = 60  # Start after warmup
        self.initial_equity = 100000.0
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.trade_history: List[Dict] = []
        self.daily_pnl = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Randomize starting position for episode diversity
        min_start = 60  # Lookback buffer
        max_start = max(min_start + 1, self.n_samples - 500)  # Leave room for episode
        self.current_step = self.np_random.integers(min_start, max_start)
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.trade_history = []
        self.daily_pnl = 0.0

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        # Detect regime from features
        regime_onehot, _ = detect_regime(self.features[self.current_step])

        # Win rate from recent trades
        recent = self.trade_history[-20:] if self.trade_history else []
        win_rate = sum(1 for t in recent if t.get('pnl', 0) > 0) / max(len(recent), 1)

        # Current drawdown
        drawdown = (self.peak_equity - self.equity) / self.peak_equity

        obs = np.array([
            regime_onehot[0],  # TREND
            regime_onehot[1],  # CHOP
            regime_onehot[2],  # MEAN_REVERT
            win_rate,
            drawdown,
        ], dtype=np.float32)

        return obs

    def _scale_action(self, action: np.ndarray) -> Tuple[float, float, float]:
        """Scale normalized action to actual threshold deltas."""
        conf_delta = action[0] * self.config.confidence_delta_range * 100  # Scale to 0-100
        mom_delta = action[1] * self.config.momentum_delta_range
        risk_delta = action[2] * self.config.risk_delta_range

        return conf_delta, mom_delta, risk_delta

    def _check_gates(
        self,
        conf_thresh: float,
        mom_thresh: float,
        risk_thresh: float,
    ) -> Tuple[bool, Dict]:
        """Check if current market state passes adjusted gates."""
        pred = self.ensemble_preds[self.current_step]

        # Handle both scalar and array predictions
        if np.isscalar(pred) or (isinstance(pred, np.ndarray) and pred.ndim == 0):
            # Single probability value - derive gate values from features
            tcn_prob = float(pred)
            feat = self.features[self.current_step]
            ridge_conf = float(feat[2]) if len(feat) > 2 else 50.0  # ADX-like feature
            xgb_mom = float(abs(feat[1]) / 100) if len(feat) > 1 else 0.3  # Momentum feature
            rf_drawdown = float(feat[0] / 100) if len(feat) > 0 else 0.02  # Volatility as proxy
        elif hasattr(pred, '__len__') and len(pred) >= 4:
            # Extract predictions (format: [tcn_prob, ridge_conf, xgb_mom, rf_dd])
            tcn_prob = float(pred[0])
            ridge_conf = float(pred[1])
            xgb_mom = float(pred[2])
            rf_drawdown = float(pred[3])
        else:
            # Fallback: use first value as prob, derive rest
            tcn_prob = float(pred[0]) if hasattr(pred, '__getitem__') else 0.5
            feat = self.features[self.current_step]
            ridge_conf = float(feat[2]) if len(feat) > 2 else 50.0
            xgb_mom = float(abs(feat[1]) / 100) if len(feat) > 1 else 0.3
            rf_drawdown = float(feat[0] / 100) if len(feat) > 0 else 0.02

        # Apply gates with adjusted thresholds
        conf_passed = ridge_conf >= conf_thresh
        mom_passed = xgb_mom >= mom_thresh
        risk_passed = rf_drawdown <= risk_thresh

        all_passed = conf_passed and mom_passed and risk_passed

        gate_info = {
            'tcn_prob': tcn_prob,
            'ridge_conf': ridge_conf,
            'xgb_mom': xgb_mom,
            'rf_drawdown': rf_drawdown,
            'conf_passed': conf_passed,
            'mom_passed': mom_passed,
            'risk_passed': risk_passed,
            'thresholds': {
                'confidence': conf_thresh,
                'momentum': mom_thresh,
                'risk': risk_thresh,
            }
        }

        return all_passed, gate_info

    def _simulate_trade(
        self,
        gates_passed: bool,
        gate_info: Dict,
    ) -> Dict:
        """Simulate trade outcome based on gate decision with transaction costs."""
        if not gates_passed:
            return {'pnl': 0.0, 'traded': False, 'transaction_costs': 0.0}

        # Get prices
        entry_price = self.prices[self.current_step]

        # Look ahead for exit (simplified: next bar)
        exit_idx = min(self.current_step + 1, self.n_samples - 1)
        exit_price = self.prices[exit_idx]

        # Direction from TCN probability
        tcn_prob = gate_info['tcn_prob']
        direction = 1 if tcn_prob > 0.5 else -1

        # Calculate P/L (2% position size)
        position_value = self.equity * 0.02
        price_change = (exit_price - entry_price) / entry_price
        raw_pnl = position_value * price_change * direction
        
        # CRITICAL: Apply transaction costs
        # This prevents the agent from trading every step
        spread_cost = position_value * self.config.spread_cost_pct
        commission_cost = position_value * self.config.commission_pct
        slippage_cost = position_value * self.config.slippage_pct
        transaction_costs = spread_cost + commission_cost + slippage_cost
        
        # Net P/L after costs
        pnl = raw_pnl - transaction_costs

        return {
            'pnl': pnl,
            'raw_pnl': raw_pnl,
            'traded': True,
            'direction': direction,
            'entry': entry_price,
            'exit': exit_price,
            'transaction_costs': transaction_costs,
        }

    def _calculate_reward(self, trade_result: Dict) -> float:
        """Calculate shaped reward with transaction costs and asymmetric scaling."""
        pnl = trade_result.get('pnl', 0)
        traded = trade_result.get('traded', False)
        transaction_costs = trade_result.get('transaction_costs', 0)

        # === No-trade reward (important for learning when NOT to trade) ===
        if not traded:
            # Check if we SHOULD have traded (opportunity cost)
            if self.current_step > 0 and self.current_step < self.n_samples - 1:
                # Look at next period returns
                future_return = (
                    self.prices[min(self.current_step + 5, self.n_samples - 1)]
                    - self.prices[self.current_step]
                ) / self.prices[self.current_step]

                # Small reward for avoiding bad trades
                if abs(future_return) < 0.001:  # Flat market - good to skip
                    return 0.05  # Small positive for correct skip
                elif self.directions[self.current_step] == 0:  # No clear direction
                    return 0.02  # Small positive for avoiding uncertain trade
                else:
                    # Missed opportunity penalty (scaled by how good the move was)
                    return -0.02 * min(abs(future_return) * 100, 1.0)
            return 0.0

        # === Traded: base reward from P/L with asymmetric scaling ===
        # CRITICAL FIX: Apply asymmetric scaling - losses hurt MORE than gains feel good
        pnl_normalized = pnl / self.initial_equity * 100
        
        if pnl > 0:
            # Profitable trade: apply bonus multiplier
            base_reward = pnl_normalized * self.config.profit_multiplier
        else:
            # Losing trade: apply penalty multiplier (losses hurt MORE)
            base_reward = pnl_normalized * self.config.loss_multiplier
        
        # Use soft tanh clipping
        reward = 5.0 * np.tanh(base_reward / 5.0)  # Soft clip to ~[-5, 5]
        
        # === Transaction cost penalty (additional to P/L deduction) ===
        # This further discourages excessive trading
        cost_penalty = transaction_costs / self.initial_equity * 100
        reward -= cost_penalty * 1.5  # Extra penalty for trading costs

        # === Sortino ratio component (penalize downside more than upside) ===
        if len(self.trade_history) > 5:
            returns = np.array([t.get('pnl', 0) / self.initial_equity
                               for t in self.trade_history[-20:]])
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 1:
                downside_std = np.std(negative_returns)
                if downside_std > 1e-8:
                    sortino = np.mean(returns) / downside_std
                    reward += self.config.sharpe_weight * np.tanh(sortino / 2)

        # === Drawdown penalty (progressive) ===
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        if drawdown > 0.01:  # Start penalizing earlier
            # Progressive penalty: gentle at first, harsh at high DD
            dd_penalty = self.config.drawdown_penalty * (drawdown ** 1.5) * 50
            reward -= np.tanh(dd_penalty)  # Soft cap the penalty too

        # === Win rate bonus (only if enough trades) ===
        recent = self.trade_history[-10:] if len(self.trade_history) >= 5 else []
        if recent:
            win_rate = sum(1 for t in recent if t.get('pnl', 0) > 0) / len(recent)
            if win_rate > 0.45:  # Lower threshold to encourage learning
                reward += self.config.win_rate_bonus * (win_rate - 0.45) * 5

        # === Risk-adjusted bonus for profitable trades ===
        if pnl > 0:
            # Reward higher profit factor
            avg_loss = abs(np.mean([t.get('pnl', 0) for t in self.trade_history if t.get('pnl', 0) < 0] or [-1]))
            if avg_loss > 0:
                profit_factor = pnl / avg_loss
                reward += 0.1 * np.tanh(profit_factor - 1)  # Bonus for PF > 1

        return float(reward)  # No hard clip - tanh already bounds it

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step with adjusted thresholds.

        Args:
            action: Threshold adjustments (3-dim, normalized to [-1, 1])

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Scale action to actual threshold deltas
        conf_delta, mom_delta, risk_delta = self._scale_action(action)

        # Calculate adjusted thresholds
        conf_thresh = self.config.base_confidence + conf_delta
        mom_thresh = self.config.base_momentum + mom_delta
        risk_thresh = self.config.base_risk + risk_delta

        # Clip to valid ranges
        conf_thresh = np.clip(conf_thresh, 30, 70)
        mom_thresh = np.clip(mom_thresh, 0.05, 0.40)
        risk_thresh = np.clip(risk_thresh, 0.01, 0.05)

        # Check gates with adjusted thresholds
        gates_passed, gate_info = self._check_gates(
            conf_thresh, mom_thresh, risk_thresh
        )

        # Simulate trade
        trade_result = self._simulate_trade(gates_passed, gate_info)

        # Update state
        pnl = trade_result.get('pnl', 0)
        self.equity += pnl
        self.daily_pnl += pnl
        self.peak_equity = max(self.peak_equity, self.equity)

        if trade_result.get('traded', False):
            self.trade_history.append(trade_result)

        # Calculate reward
        reward = self._calculate_reward(trade_result)

        # Move to next step
        self.current_step += 1

        # Check termination
        terminated = False
        truncated = False
        info = {
            'pnl': pnl,
            'equity': self.equity,
            'gates_passed': gates_passed,
            'adjusted_thresholds': {
                'confidence': conf_thresh,
                'momentum': mom_thresh,
                'risk': risk_thresh,
            }
        }

        # End of data
        if self.current_step >= self.n_samples - 1:
            terminated = True
            info['reason'] = 'end_of_data'

        # Max drawdown
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        if drawdown > self.config.max_drawdown_pct:
            terminated = True
            reward -= 2.0  # Penalty (VecNormalize handles scaling)
            info['reason'] = 'max_drawdown'

        # Max trades
        if len(self.trade_history) >= self.config.max_trades_per_episode:
            truncated = True
            info['reason'] = 'max_trades'

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        """Render current state."""
        drawdown = (self.peak_equity - self.equity) / self.peak_equity * 100
        win_rate = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0) / max(len(self.trade_history), 1)
        print(f"Step: {self.current_step}, Equity: ${self.equity:.2f}, "
              f"DD: {drawdown:.1f}%, Trades: {len(self.trade_history)}, "
              f"WR: {win_rate:.1%}")


class GateThresholdRL:
    """
    RL-based gate threshold optimizer using SAC.

    Learns to dynamically adjust gate thresholds based on market conditions.
    """

    def __init__(self, config: Optional[GateRLConfig] = None):
        self.config = config or GateRLConfig()
        self.model = None
        self.scaler = None
        self._is_trained = False

    def train(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        ensemble_preds: np.ndarray,
        timesteps: Optional[int] = None,
        eval_freq: int = 5000,
        verbose: int = 1,
        continue_training: bool = False,
    ) -> Dict[str, Any]:
        """
        Train SAC agent for gate threshold optimization.

        Args:
            prices: Close prices
            features: Market features
            ensemble_preds: Ensemble model predictions
            timesteps: Total training timesteps (overrides config)
            eval_freq: Evaluation frequency
            verbose: Verbosity level
            continue_training: If True, load existing model and continue training

        Returns:
            Training statistics
        """
        _ensure_sb3_imported()
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required: pip install stable-baselines3")

        # Import here to avoid slow startup
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
        from stable_baselines3.common.monitor import Monitor

        total_timesteps = timesteps or self.config.total_timesteps
        n_envs = self.config.n_envs
        logger.info(f"🎯 Training Gate Threshold RL (SAC) for {total_timesteps:,} timesteps with {n_envs} envs...")

        # Check for incremental learning - load existing config first (affects observation space)
        from sklearn.preprocessing import StandardScaler
        config_path = GATE_RL_MODEL_PATH.parent / "gate_rl_config.pkl"
        scaler_path = GATE_RL_MODEL_PATH.parent / "gate_rl_scaler.pkl"

        if continue_training and GATE_RL_MODEL_PATH.exists():
            if config_path.exists():
                # Config file exists - load it
                logger.info(f"📂 Loading existing config for incremental training: {config_path}")
                with open(config_path, 'rb') as f:
                    self.config = pickle.load(f)
            else:
                # Config file missing - infer from model's observation space
                logger.warning("⚠️ Config file missing - inferring from saved model observation space")
                inferred = infer_config_from_model(GATE_RL_MODEL_PATH)
                if inferred:
                    logger.info(f"📊 Inferred config: max_drawdown_pct={inferred['max_drawdown_pct']}")
                    self.config.max_drawdown_pct = inferred['max_drawdown_pct']
                    # Save for future runs
                    with open(config_path, 'wb') as f:
                        pickle.dump(self.config, f)
                    logger.info(f"💾 Saved inferred config to {config_path}")
                else:
                    logger.error("❌ Cannot infer config from model - observation space mismatch likely")
            # Update n_envs from loaded/updated config
            n_envs = self.config.n_envs

        if continue_training and scaler_path.exists():
            logger.info(f"📂 Loading existing scaler for incremental training: {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            features_scaled = self.scaler.transform(features)
        else:
            # Fit new scaler
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)

        # Factory function for creating environments
        def make_env(rank: int):
            def _init():
                return GateThresholdEnv(
                    prices=prices,
                    features=features_scaled,
                    ensemble_preds=ensemble_preds,
                    config=self.config,
                )
            return _init

        # Create vectorized environment (parallel if n_envs > 1)
        if n_envs > 1:
            vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        else:
            vec_env = DummyVecEnv([make_env(0)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # Check for incremental learning - load existing model if available
        if continue_training and GATE_RL_MODEL_PATH.exists():
            logger.info(f"📂 Loading existing model for incremental training: {GATE_RL_MODEL_PATH}")
            self.model = SAC.load(str(GATE_RL_MODEL_PATH), env=vec_env, device="cpu")
            logger.info("✅ Continuing training from saved model")
        else:
            # Create new SAC model
            # CRITICAL: CPU only to avoid Metal/TensorFlow conflicts
            policy_kwargs = dict(
                net_arch=dict(
                    pi=self.config.net_arch_pi,
                    qf=self.config.net_arch_qf
                )
            )

            self.model = SAC(
                "MlpPolicy",
                vec_env,
                learning_rate=self.config.learning_rate,
                buffer_size=self.config.buffer_size,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                tau=self.config.tau,
                ent_coef=self.config.ent_coef,  # Fixed entropy coefficient
                learning_starts=self.config.learning_starts,  # Random exploration steps
                policy_kwargs=policy_kwargs,
                device="cpu",  # CRITICAL: avoid GPU conflicts
                verbose=verbose,
            )

        # Evaluation environment with Monitor + VecNormalize for proper tracking
        eval_log_dir = Path("trained_data/logs/rl_gates")
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        eval_env = DummyVecEnv([lambda: Monitor(
            GateThresholdEnv(
                prices=prices,
                features=features_scaled,
                ensemble_preds=ensemble_preds,
                config=self.config,
            ),
            str(eval_log_dir / "eval_monitor")
        )])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(GATE_RL_MODEL_PATH.parent),
            log_path=str(Path("trained_data/logs/rl_gates")),
            eval_freq=eval_freq,
            n_eval_episodes=5,
            deterministic=True,
        )

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )

        # Load the BEST model saved by EvalCallback (not the final one)
        best_model_path = GATE_RL_MODEL_PATH.parent / "best_model.zip"
        if best_model_path.exists():
            logger.info(f"📥 Loading best model from {best_model_path}")
            self.model = SAC.load(best_model_path, env=vec_env, device="cpu")

        self._is_trained = True
        self.save()

        # Stats - access underlying environment from vectorized wrapper
        # Note: SubprocVecEnv doesn't expose .envs, only DummyVecEnv does
        try:
            underlying_env = vec_env.venv.envs[0]
            stats = {
                'total_timesteps': total_timesteps,
                'final_equity': underlying_env.equity,
                'max_drawdown': (underlying_env.peak_equity - underlying_env.equity) / underlying_env.peak_equity if underlying_env.peak_equity > 0 else 0.0,
                'total_trades': len(underlying_env.trade_history),
                'win_rate': sum(1 for t in underlying_env.trade_history if t.get('pnl', 0) > 0) / max(len(underlying_env.trade_history), 1),
            }
        except AttributeError:
            # SubprocVecEnv runs envs in separate processes - can't access directly
            logger.info("Using SubprocVecEnv - detailed stats not available (envs in separate processes)")
            stats = {
                'total_timesteps': total_timesteps,
                'final_equity': None,
                'max_drawdown': None,
                'total_trades': None,
                'win_rate': None,
            }

        logger.info(f"✅ Gate Threshold RL training complete: {stats}")
        return stats

    def get_adjusted_thresholds(
        self,
        features: np.ndarray,
        win_rate: float = 0.5,
        drawdown: float = 0.0,
    ) -> Dict[str, float]:
        """
        Get optimal gate thresholds for current market state.

        Args:
            features: Current market features
            win_rate: Recent win rate
            drawdown: Current drawdown

        Returns:
            Dictionary with adjusted thresholds
        """
        if not self._is_trained or self.model is None:
            # Return base thresholds
            return {
                'confidence': self.config.base_confidence,
                'momentum': self.config.base_momentum,
                'risk': self.config.base_risk,
            }

        # Detect regime
        regime_onehot, _ = detect_regime(features)

        # Build observation
        obs = np.array([
            regime_onehot[0],
            regime_onehot[1],
            regime_onehot[2],
            win_rate,
            drawdown,
        ], dtype=np.float32)

        # Get action
        action, _ = self.model.predict(obs, deterministic=True)

        # Scale to actual deltas
        conf_delta = action[0] * self.config.confidence_delta_range * 100
        mom_delta = action[1] * self.config.momentum_delta_range
        risk_delta = action[2] * self.config.risk_delta_range

        # Calculate and clip thresholds
        conf = np.clip(self.config.base_confidence + conf_delta, 30, 70)
        mom = np.clip(self.config.base_momentum + mom_delta, 0.05, 0.40)
        risk = np.clip(self.config.base_risk + risk_delta, 0.01, 0.05)

        return {
            'min_confidence': float(conf),
            'min_momentum': float(mom),
            'max_drawdown_pct': float(risk),
        }

    def save(self):
        """Save model, scaler, and config."""
        GATE_RL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save(str(GATE_RL_MODEL_PATH))
            logger.info(f"💾 Gate RL model saved to {GATE_RL_MODEL_PATH}")

        if self.scaler is not None:
            with open(GATE_RL_SCALER_PATH, "wb") as f:
                pickle.dump(self.scaler, f)

        # Save config for incremental learning compatibility
        config_path = GATE_RL_MODEL_PATH.parent / "gate_rl_config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)
        logger.info(f"💾 Gate RL config saved to {config_path}")

    def load(self, force_direct: bool = False) -> bool:
        """Load model and scaler.

        When TensorFlow is already loaded, uses subprocess to avoid
        TF/PyTorch deadlock on macOS (Intel and Metal).

        Args:
            force_direct: If True, skip the subprocess fallback and load
                directly with SAC.load(). Use when already in a child process
                (e.g. rl_worker) where the TF import is just an SB3 side-effect.
        """
        _ensure_sb3_imported()
        if not SB3_AVAILABLE:
            logger.debug("stable-baselines3 not available")
            return False

        try:
            if GATE_RL_MODEL_PATH.exists():
                import sys
                if not force_direct and 'tensorflow' in sys.modules:
                    self._load_via_subprocess(GATE_RL_MODEL_PATH)
                    # Fallback to direct load if subprocess failed
                    if not self._is_trained:
                        logger.info("Gate RL subprocess failed, falling back to direct load")
                        self.model = SAC.load(str(GATE_RL_MODEL_PATH), device="cpu")
                        self._is_trained = True
                        logger.info(f"📂 Gate RL model loaded from {GATE_RL_MODEL_PATH} (direct fallback)")
                else:
                    self.model = SAC.load(str(GATE_RL_MODEL_PATH), device="cpu")
                    self._is_trained = True
                    logger.info(f"📂 Gate RL model loaded from {GATE_RL_MODEL_PATH}")

            if GATE_RL_SCALER_PATH.exists():
                with open(GATE_RL_SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)

            return self._is_trained
        except Exception as e:
            logger.warning(f"Failed to load Gate RL model: {e}")
            return False

    def _load_via_subprocess(self, model_path: Path) -> None:
        """Load SAC model in a child process to avoid TF/PyTorch deadlock.

        The child process doesn't have TF imported, so SAC.load() works fine.
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
                "    from stable_baselines3 import SAC\n"
                f'    model = SAC.load("{model_path}", device="cpu")\n'
                f'    with open("{tmp_path}", "wb") as f:\n'
                "        cloudpickle.dump(model, f)\n"
                "    sys.exit(0)\n"
                "except Exception as e:\n"
                '    print(f"Gate RL subprocess load failed: {e}", file=sys.stderr)\n'
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
                logger.info(f"📂 Gate RL model loaded from {model_path} (via subprocess)")
            else:
                err = result.stderr.strip() if result.stderr else "unknown error"
                logger.warning(f"Gate RL subprocess load failed: {err}")
        except subprocess.TimeoutExpired:
            logger.warning("Gate RL subprocess load timed out (30s)")
        except Exception as e:
            logger.warning(f"Gate RL subprocess load failed: {e}")
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
