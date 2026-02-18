# Existing RL Components Review

**Document Purpose**: Comprehensive analysis of existing Reinforcement Learning implementations to identify patterns for extension.

**Date**: February 2026  
**Status**: Complete

---

## Table of Contents

1. [RL Algorithms Inventory](#1-rl-algorithms-inventory)
2. [Interface Patterns](#2-interface-patterns)
3. [Data Flow Diagrams](#3-data-flow-diagrams)
4. [Configuration Schema](#4-configuration-schema)
5. [Extensibility Assessment](#5-extensibility-assessment)
6. [Code Patterns](#6-code-patterns)

---

## 1. RL Algorithms Inventory

### 1.1 Implemented Algorithms

| Algorithm | Framework | File | Purpose |
|-----------|-----------|------|---------|
| **PPO** (Proximal Policy Optimization) | stable-baselines3 | [`rl_position_sizing.py`](../rl_position_sizing.py) | Position sizing optimization |
| **SAC** (Soft Actor-Critic) | stable-baselines3 | Referenced in docs | Gate threshold optimization |
| **Incremental Learning** | scikit-learn | [`online_retrainer.py`](../online_retrainer.py) | Gate model retraining |

### 1.2 Framework Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL FRAMEWORK STACK                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Application Layer                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ RLPositionSizer │  │ OnlineRetrainer │  │ Future RL       │  │
│  │                 │  │                 │  │ Components      │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │            │
│  ┌────────┴────────────────────┴────────────────────┴────────┐  │
│  │              RL Algorithm Layer                            │  │
│  │  stable-baselines3 (PPO, SAC)                             │  │
│  │  - MlpPolicy network architecture                        │  │
│  │  - EvalCallback for early stopping                       │  │
│  │  - DummyVecEnv for environment wrapping                  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                           │                                      │
│  ┌────────────────────────┴───────────────────────────────────┐  │
│  │              Environment Layer                              │  │
│  │  gymnasium (gym)                                           │  │
│  │  - Env base class                                          │  │
│  │  - spaces (Box, Discrete)                                  │  │
│  │  - Monitor wrapper                                         │  │
│  └────────────────────────────────────────────────────────────┘  │
│                           │                                      │
│  ┌────────────────────────┴───────────────────────────────────┐  │
│  │              ML Backend Layer                              │  │
│  │  PyTorch (via stable-baselines3)                          │  │
│  │  scikit-learn (StandardScaler, Ridge, RF, XGBoost)        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Algorithm Details

#### PPO (Proximal Policy Optimization)

**Location**: [`rl_position_sizing.py:532`](../rl_position_sizing.py:532)

```python
self.model = PPO(
    "MlpPolicy",           # Standard MLP policy network
    vec_env,               # Vectorized environment
    learning_rate=3e-4,    # Default learning rate
    n_steps=2048,          # Steps per update
    batch_size=64,         # Mini-batch size
    n_epochs=10,           # Epochs per update
    gamma=0.99,            # Discount factor
    gae_lambda=0.95,       # GAE parameter
    verbose=0,             # Suppress SB3 output
    device="cpu",          # Force CPU for TF compatibility
)
```

**Key Characteristics**:
- Uses MlpPolicy (Multi-Layer Perceptron)
- CPU-only to avoid TensorFlow/PyTorch GPU conflicts
- Supports early stopping via EvalCallback
- Discrete action space for position sizing

#### SAC (Soft Actor-Critic)

**Referenced in**: [`docs/RL_TRAINING_IMPROVEMENTS.md`](RL_TRAINING_IMPROVEMENTS.md)

Used for gate threshold optimization with continuous action spaces.

---

## 2. Interface Patterns

### 2.1 Environment Interface (Gymnasium)

The [`TradingEnv`](../rl_position_sizing.py:149) class implements the standard Gymnasium interface:

```python
class TradingEnv(gym.Env):
    """Gymnasium environment for RL-based position sizing."""
    
    # Observation Space: Box with shape (obs_dim,)
    observation_space: spaces.Box
    
    # Action Space: Discrete with n actions
    action_space: spaces.Discrete
    
    # Core methods
    def __init__(self, features, ensemble_predictions, prices, config): ...
    def step(self, action) -> Tuple[obs, reward, terminated, truncated, info]: ...
    def reset(self, seed=None, options=None) -> Tuple[obs, info]: ...
    def render(self, mode="human") -> None: ...
```

### 2.2 Observation Space Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    OBSERVATION VECTOR                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Component 1: Market Features (n_features dimensions)           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ - Technical indicators from feature engineering            │ │
│  │ - Scaled using StandardScaler                              │ │
│  │ - Shape: (n_features,) typically 20-50 features            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Component 2: Ensemble Predictions (2 dimensions)               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ [0] direction_prob: Probability of upward movement         │ │
│  │ [1] confidence: Model confidence in prediction             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Component 3: Account State (4 dimensions)                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ [0] equity_pct: Current equity vs initial (relative)       │ │
│  │ [1] drawdown: Current drawdown from peak equity            │ │
│  │ [2] daily_pnl_pct: Daily P&L as percentage                 │ │
│  │ [3] win_rate: Rolling win rate from last 20 trades         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Total Dimension: n_features + 2 + 4 = n_features + 6           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Action Space Structure

**Discrete Actions** (6 levels):

| Action | Position Size | Description |
|--------|---------------|-------------|
| 0 | 0.0% | No position |
| 1 | 1.0% | Minimum position |
| 2 | 3.0% | Small position |
| 3 | 5.0% | Medium position |
| 4 | 7.0% | Large position |
| 5 | 10.0% | Maximum position |

```python
# From rl_position_sizing.py:111
POSITION_LEVELS = [0.0, 0.01, 0.03, 0.05, 0.07, 0.10]
```

### 2.4 Reward Structure

The reward function is multi-component:

```python
def _calculate_reward(self, pnl: float) -> float:
    """
    Reward = Base_P/L + Sharpe_Adjustment - Drawdown_Penalty + Win_Rate_Bonus
    """
    # 1. Base reward: normalized P/L
    reward = pnl / initial_balance * 100  # Scale to percentage
    
    # 2. Sharpe component: reward consistency
    if len(trade_history) > 5:
        sharpe = mean(returns) / std(returns)
        reward += sharpe_weight * sharpe  # sharpe_weight = 0.1
    
    # 3. Drawdown penalty
    if drawdown > 0.05:
        reward -= drawdown_penalty * drawdown * 100  # penalty = 0.5
    
    # 4. Win rate bonus
    if win_rate > 0.5:
        reward += win_rate_bonus * (win_rate - 0.5) * 10  # bonus = 0.2
    
    return reward
```

### 2.5 Integration Hook Pattern

RL components integrate via post-ensemble hooks:

```python
# From cli/training.py
def _train_rl_position_sizer_if_ready(
    console,
    rl_timesteps: int = 500_000,
    min_samples: int = 500,
    *,
    features: np.ndarray | None = None,
    ensemble_predictions: np.ndarray | None = None,
    prices: np.ndarray | None = None,
) -> bool:
    """
    Train RL position sizer after ensemble training completes.
    
    Integration Pattern: Post-training hook with data passthrough
    """
```

**Called at 4 locations** in training pipeline:
1. After enterprise validation (Line 2407)
2. After XGBoost training (Line 2652)
3. After ensemble completion (Line 4213)
4. With graceful error handling at each point

---

## 3. Data Flow Diagrams

### 3.1 Training Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RL TRAINING DATA FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                                                        │
│  │ Historical Data  │                                                        │
│  │ - OANDA API      │                                                        │
│  │ - CSV files      │                                                        │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐     ┌──────────────────┐                              │
│  │ Feature          │────▶│ Ensemble Models  │                              │
│  │ Engineering      │     │ - Transformer    │                              │
│  │ - 200+ features  │     │ - XGBoost        │                              │
│  │ - Noise reduction│     │ - RF             │                              │
│  └──────────────────┘     │ - Ridge          │                              │
│                           └────────┬─────────┘                              │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    RL TRAINING DATA PREPARATION                       │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  rl_features = feature_df[feature_columns].values[:min_len]          │  │
│  │  rl_predictions = np.column_stack([direction_probs, confidences])    │  │
│  │  rl_prices = feature_df['close'].values[:min_len]                    │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐   │
│  │ StandardScaler   │────▶│ TradingEnv       │────▶│ PPO Training     │   │
│  │ - Fit on train   │     │ - Gymnasium Env  │     │ - 100K timesteps │   │
│  │ - Transform all  │     │ - Observation    │     │ - EvalCallback   │   │
│  └──────────────────┘     │ - Reward calc    │     │ - Early stopping │   │
│                           └──────────────────┘     └────────┬─────────┘   │
│                                                             │              │
│                                                             ▼              │
│                           ┌──────────────────────────────────────────────┐ │
│                           │                    OUTPUT                     │ │
│                           ├──────────────────────────────────────────────┤ │
│                           │ rl_position_sizer.zip (PPO model)            │ │
│                           │ rl_scaler.pkl (StandardScaler)               │ │
│                           └──────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Inference Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RL INFERENCE DATA FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                                                        │
│  │ Live Market Data │                                                        │
│  │ - Real-time bars │                                                        │
│  │ - Account state  │                                                        │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐     ┌──────────────────┐                              │
│  │ Feature          │────▶│ Ensemble Models  │                              │
│  │ Engineering      │     │ - Predictions    │                              │
│  │ - Same pipeline  │     │ - Confidence     │                              │
│  └──────────────────┘     └────────┬─────────┘                              │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    RL POSITION SIZER INFERENCE                        │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  # Scale features                                                    │  │
│  │  features_scaled = scaler.transform(features)                        │  │
│  │                                                                       │  │
│  │  # Construct observation                                             │  │
│  │  obs = concatenate([                                                 │  │
│  │      features_scaled,                                                │  │
│  │      ensemble_prediction,  # [direction_prob, confidence]            │  │
│  │      [0.0, 0.0, 0.0, 0.5]  # Placeholder account state              │  │
│  │  ])                                                                  │  │
│  │                                                                       │  │
│  │  # Get action from model                                             │  │
│  │  action, _ = model.predict(obs, deterministic=True)                  │  │
│  │                                                                       │  │
│  │  # Convert to position size                                          │  │
│  │  position_pct = POSITION_LEVELS[action]                              │  │
│  │  position_size = account_equity * position_pct                       │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│                           ┌──────────────────┐                              │
│                           │ Position Size    │                              │
│                           │ - In dollars     │                              │
│                           │ - Risk-adjusted  │                              │
│                           └──────────────────┘                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Online Retraining Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ONLINE RETRAINING FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐     ┌──────────────────┐                              │
│  │ Drift Detection  │────▶│ Trigger Check    │                              │
│  │ - Feature drift  │     │ - Cooldown OK?   │                              │
│  │ - Accuracy drop  │     │ - Daily limit?   │                              │
│  └──────────────────┘     └────────┬─────────┘                              │
│                                    │                                         │
│                           Yes      │                                         │
│                    ┌───────────────┘                                         │
│                    │                                                         │
│                    ▼                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    LOAD REPLAY DATA                                   │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  Sources:                                                            │  │
│  │  1. trained_data/online_learning/trade_buffer.json                   │  │
│  │  2. trained_data/drift_detection/drift_state.json                    │  │
│  │                                                                       │  │
│  │  Validation:                                                         │  │
│  │  - min_samples_for_retrain: 50                                       │  │
│  │  - Consistent feature dimensions                                     │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    RETRAIN GATE MODELS                                │  │
│  ├──────────────────────────────────────────────────────────────────────┤  │
│  │                                                                       │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │  │
│  │  │ XGBoost         │  │ RandomForest    │  │ Ridge               │  │  │
│  │  │ Momentum Model  │  │ Risk Model      │  │ Confidence Model    │  │  │
│  │  │                 │  │                 │  │                     │  │  │
│  │  │ n_estimators=50 │  │ n_estimators=50 │  │ alpha=1.0           │  │  │
│  │  │ max_depth=4     │  │ max_depth=6     │  │                     │  │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────┐     ┌──────────────────┐                              │
│  │ Save Updated     │────▶│ Update State     │                              │
│  │ Models           │     │ - Retrain count  │                              │
│  │ - .pkl format    │     │ - Timestamp      │                              │
│  └──────────────────┘     │ - History log    │                              │
│                           └──────────────────┘                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Configuration Schema

### 4.1 RLConfig Dataclass

```python
# From rl_position_sizing.py:114-138
@dataclass
class RLConfig:
    """Configuration for RL position sizing."""
    
    # Environment settings
    sequence_length: int = 60        # Observation window
    max_position_pct: float = 0.10   # Max 10% of account per trade
    min_position_pct: float = 0.01   # Min 1% of account per trade
    
    # Reward shaping
    sharpe_weight: float = 0.1       # Weight for Sharpe ratio in reward
    drawdown_penalty: float = 0.5    # Penalty for drawdowns
    win_rate_bonus: float = 0.2      # Bonus for winning trades
    
    # Training hyperparameters
    total_timesteps: int = 100_000   # Total training steps
    learning_rate: float = 3e-4      # Learning rate
    n_steps: int = 2048              # Steps per update
    batch_size: int = 64             # Mini-batch size
    n_epochs: int = 10               # Epochs per update
    gamma: float = 0.99              # Discount factor
    gae_lambda: float = 0.95         # GAE lambda parameter
    
    # Risk limits
    max_drawdown_pct: float = 0.10   # 10% max drawdown triggers reset
    daily_loss_limit_pct: float = 0.03  # 3% daily loss limit
```

### 4.2 RetrainConfig Dataclass

```python
# From online_retrainer.py:33-56
@dataclass
class RetrainConfig:
    """Configuration for online retraining."""
    
    # Cooldown settings
    cooldown_minutes: int = 60       # Minimum time between retrains
    max_retrains_per_day: int = 3    # Daily limit
    
    # Data requirements
    min_samples_for_retrain: int = 50    # Minimum replay samples
    min_accuracy_drop: float = 0.05      # 5% accuracy drop triggers retrain
    
    # Training parameters
    epochs_per_retrain: int = 1      # For TF if extended
    validation_split: float = 0.2    # 20% for validation
    
    # Model selection
    retrain_xgboost: bool = True
    retrain_rf: bool = True
    retrain_ridge: bool = True
    retrain_transformer: bool = False  # Expensive, usually skip
    
    # Persistence
    state_file: str = "trained_data/online_retrain_state.json"
```

### 4.3 YAML Configuration Integration

```yaml
# From config/config_improved_H1.yaml
buddy:
  train_defaults:
    # RL training settings
    rl_timesteps: 500000
    auto_train_rl: true
    
training:
  # RL auto-training after ensemble
  auto_train_rl: true
  
regime:
  # Regime detection for RL context
  use_regime: true
  regime_lookback: 20
  regime_lookahead: 12
```

---

## 5. Extensibility Assessment

### 5.1 Components That Can Be Extended

| Component | Extensibility | Notes |
|-----------|---------------|-------|
| **TradingEnv** | ✅ High | Can subclass for new environments |
| **RLConfig** | ✅ High | Dataclass can be extended |
| **Reward Function** | ✅ High | Modular `_calculate_reward()` method |
| **Observation Space** | ✅ Medium | Requires dimension updates |
| **Action Space** | ⚠️ Medium | Requires POSITION_LEVELS update |
| **PPO Hyperparameters** | ✅ High | All configurable via RLConfig |
| **OnlineRetrainer** | ✅ High | Can add new model types |

### 5.2 Components That Need New Implementation

| Component | Reason | Suggested Approach |
|-----------|--------|-------------------|
| **New RL Algorithm** | Different algorithm (DQN, A2C) | Create new trainer class |
| **Continuous Actions** | Current is discrete | New action space definition |
| **Multi-Agent RL** | Not supported | New architecture required |
| **RL for Feature Selection** | Not implemented | New environment design |
| **RL for Hyperparameter Optimization** | Not implemented | New environment design |

### 5.3 Extension Patterns

#### Pattern A: New Environment Creation

```python
# Recommended pattern for new RL environments
class NewTradingEnv(gym.Env):
    """New RL environment for [purpose]."""
    
    def __init__(
        self,
        features: np.ndarray,
        ensemble_predictions: np.ndarray,
        prices: np.ndarray,
        config: Optional[RLConfig] = None,
    ):
        # 1. Call parent init
        super().__init__()
        
        # 2. Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # 3. Define action space
        self.action_space = spaces.Discrete(n_actions)
        
        # 4. Initialize state
        self._reset_state()
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        # Follow existing pattern
        pass
    
    def _calculate_reward(self, *args) -> float:
        """Calculate shaped reward."""
        # Follow existing pattern
        pass
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        # Follow existing pattern
        pass
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        # Follow existing pattern
        pass
```

#### Pattern B: New RL Trainer Class

```python
# Recommended pattern for new RL trainers
class NewRLTrainer:
    """New RL trainer for [purpose]."""
    
    def __init__(self, config: Optional[NewRLConfig] = None):
        self.config = config or NewRLConfig()
        self.model = None
        self.scaler = None
        self._is_trained = False
    
    def train(
        self,
        features: np.ndarray,
        ensemble_predictions: np.ndarray,
        prices: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train RL model."""
        # 1. Ensure dependencies
        _ensure_sb3_imported()
        
        # 2. Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # 3. Create environment
        env = NewTradingEnv(features_scaled, ensemble_predictions, prices, self.config)
        
        # 4. Create model
        self.model = PPO("MlpPolicy", env, ...)
        
        # 5. Train
        self.model.learn(total_timesteps=self.config.total_timesteps)
        
        # 6. Save
        self.save()
        
        return stats
    
    def predict(self, features: np.ndarray, **kwargs) -> Any:
        """Get prediction from trained model."""
        if not self._is_trained:
            return self._fallback_prediction()
        
        obs = self._construct_observation(features)
        action, _ = self.model.predict(obs, deterministic=True)
        return self._action_to_output(action)
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save model and scaler."""
        pass
    
    def load(self, path: Optional[Path] = None) -> bool:
        """Load model and scaler."""
        pass
```

---

## 6. Code Patterns

### 6.1 Lazy Import Pattern

**Purpose**: Avoid 8+ second startup penalty from PyTorch imports

```python
# From rl_position_sizing.py:49-99

# Optional dependencies - lazy load
GYM_AVAILABLE = None
SB3_AVAILABLE = None
gym = None
spaces = None
PPO = None

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
    global PPO, SB3_AVAILABLE, ...
    if SB3_AVAILABLE is None:
        try:
            from stable_baselines3 import PPO as _PPO
            # ... other imports
            SB3_AVAILABLE = True
        except (ImportError, AttributeError, Exception) as e:
            SB3_AVAILABLE = False
    return SB3_AVAILABLE
```

### 6.2 GPU Conflict Avoidance Pattern

**Purpose**: Prevent TensorFlow/PyTorch GPU conflicts on macOS

```python
# From rl_position_sizing.py:29-33

# CRITICAL: Disable GPU before PyTorch imports
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Disable CUDA GPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Enable MPS CPU fallback
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # Disable MPS memory

# Force CPU in PPO
self.model = PPO(
    "MlpPolicy",
    env,
    device="cpu",  # Force CPU to avoid GPU conflicts
)
```

### 6.3 Subprocess Isolation Pattern

**Purpose**: Load PyTorch models when TensorFlow is already loaded

```python
# From rl_position_sizing.py:710-761

def _load_via_subprocess(self, model_path: Path) -> None:
    """Load PPO model in child process to avoid TF/PyTorch deadlock."""
    
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
    )
    
    if result.returncode == 0:
        import cloudpickle
        with open(tmp_path, 'rb') as f:
            self.model = cloudpickle.load(f)
```

### 6.4 Graceful Fallback Pattern

**Purpose**: System works even when RL dependencies unavailable

```python
# From rl_position_sizing.py:612-632

def get_position_size(
    self,
    features: np.ndarray,
    ensemble_prediction: np.ndarray,
    account_equity: float = 10000.0,
) -> float:
    """Get optimal position size for current market state."""
    
    if not self._is_trained or self.model is None:
        # Fallback to simple 1% rule
        logger.warning("RL model not trained, using default 1% position size")
        return account_equity * 0.01
    
    # ... RL prediction logic
```

### 6.5 Progress Callback Pattern

**Purpose**: Show training progress without SB3 verbosity

```python
# From rl_position_sizing.py:391-458

class RLProgressCallback:
    """Callback that prints periodic RL training progress to console."""
    
    def __new__(cls, *args, **kwargs):
        # Only create if SB3 is available
        if BaseCallback is None:
            return None
        instance = super().__new__(cls)
        return instance
    
    def _make_inner(self):
        """Create the actual SB3 BaseCallback subclass instance."""
        outer = self
        
        class _Inner(BaseCallback):
            def _on_training_start(self_inner):
                outer._console_print("▶ PPO training started...")
            
            def _on_step(self_inner) -> bool:
                step = self_inner.num_timesteps
                if step - outer._last_printed >= outer._print_freq:
                    # Print progress
                    pct = 100 * step / outer._total
                    outer._console_print(f"⏳ {step:,} / {outer._total:,} ({pct:.1f}%)")
                return True
        
        return _Inner()
```

### 6.6 Thread-Safe Retraining Pattern

**Purpose**: Prevent concurrent retraining

```python
# From online_retrainer.py:102-103, 228-233

class OnlineRetrainer:
    def __init__(self, ...):
        # Thread safety
        self._retrain_lock = Lock()
        self._is_retraining = False
    
    def trigger_retrain(self, ..., force: bool = False) -> Dict[str, Any]:
        # Acquire lock
        if not self._retrain_lock.acquire(blocking=False):
            result['status'] = 'blocked'
            result['blocked_reason'] = "Another retrain is in progress"
            return result
        
        self._is_retraining = True
        
        try:
            # ... retraining logic
        finally:
            self._is_retraining = False
            self._retrain_lock.release()
```

---

## Appendix: File Summary

| File | Lines | Purpose | Key Classes |
|------|-------|---------|-------------|
| [`rl_position_sizing.py`](../rl_position_sizing.py) | 841 | PPO position sizer | `TradingEnv`, `RLPositionSizer`, `RLConfig` |
| [`online_retrainer.py`](../online_retrainer.py) | 672 | Incremental retraining | `OnlineRetrainer`, `RetrainConfig` |
| [`cli/training_ops.py`](../cli/training_ops.py) | 1097 | Training operations | `train_rl_sizer`, `train_rl_gates`, `train_rl_exits` |

---

## Summary

The existing RL implementation provides a solid foundation for extension:

1. **PPO Algorithm**: Proven, stable RL algorithm with good documentation
2. **Gymnasium Interface**: Standard environment interface for easy extension
3. **Lazy Imports**: Handles optional dependencies gracefully
4. **Subprocess Isolation**: Solves TF/PyTorch compatibility issues
5. **Configuration Dataclasses**: Type-safe, extensible configuration
6. **Post-Ensemble Integration**: Clean hook pattern for RL training

**Key Extension Points**:
- Create new environments by subclassing `TradingEnv`
- Add new RL trainers following the `RLPositionSizer` pattern
- Extend configuration via dataclass inheritance
- Integrate via post-ensemble hooks in `cli/training.py`

**Protected Components** (do not modify):
- Transformer training pipeline
- Ensemble model architectures
- Feature engineering pipeline
- Continual learning mechanisms (EMA, EWC, Replay Buffer)
