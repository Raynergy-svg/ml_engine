# RL Integration Strategy for ML Engine Trading Bot

> **Status**: Planning Complete | **Date**: 2026-01-28  
> **Based on**: Context7 documentation for stable-baselines3, gymnasium, gym-anytrading

---

## Executive Summary

The ML Engine already has a **production-ready PPO position sizer** in `rl_position_sizing.py`. This plan outlines a phased strategy to:
1. Validate and optimize the existing RL sizer
2. Formalize RL-like heuristics into proper RL frameworks
3. Add new RL modules for gate thresholds and exit timing

**Expected Impact**: +5-15% Sharpe improvement with gate threshold RL; +10-20% on optimal exits.

---

## 1. Current RL Implementation Analysis

### 1.1 Existing RL Position Sizer (`rl_position_sizing.py`)

| Component | Current Implementation |
|-----------|----------------------|
| **Algorithm** | PPO (Proximal Policy Optimization) via stable-baselines3 |
| **State Space** | `Box`: market features + ensemble predictions + account state |
| **Action Space** | `Discrete(6)`: position levels `[0%, 0.5%, 1%, 2%, 3%, 5%]` |
| **Reward** | `pnl_normalized + sharpe_weight*sharpe - drawdown_penalty*dd + win_rate_bonus` |
| **Device** | CPU-only (avoids Metal/TensorFlow conflicts) |

### 1.2 RL-Like Heuristics Already Present

| Location | Pattern | RL Opportunity |
|----------|---------|----------------|
| `src/core/modular_inference.py` | Fixed gate thresholds (50, 0.20, 2.5%) | Learn adaptive thresholds |
| `src/risk/position_sizing.py` | Confidence→multiplier step function | Continuous learned mapping |
| `online_retrainer.py` | Drift detection triggers | Contextual bandit for when to retrain |
| `config_improved_H1.yaml` | Fixed R:R (2.0 TP ratio) | Optimal stopping RL |

---

## 2. Implementation Plan

### Phase 1: Validate Existing RL Sizer (Week 1)

**Goal**: Ensure current PPO sizer outperforms heuristic Kelly/confidence sizing.

```bash
# Train with production data
python main.py train-rl-sizer --timesteps 500000

# Backtest comparison
python main.py backtest --instrument EUR_USD --use-rl-sizer
python main.py backtest --instrument EUR_USD  # heuristic baseline
```

**Add EvalCallback for early stopping** (modify `rl_position_sizing.py`):

```python
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Stop when Sharpe-proxy reward reaches threshold
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1.5, verbose=1)
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path="trained_data/models/",
    callback_on_new_best=callback_on_best,
    eval_freq=5000,
    n_eval_episodes=10,
    deterministic=True
)
model.learn(total_timesteps=timesteps, callback=eval_callback)
```

---

### Phase 2: Gate Threshold RL Environment (Week 2-3)

**Goal**: Replace fixed thresholds with learned adaptive policy.

**New File**: `src/rl/gate_threshold_env.py`

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GateThresholdEnv(gym.Env):
    """
    RL environment for learning optimal gate thresholds.
    
    Observation (5-dim Box):
        - regime_encoding: one-hot [trend, chop, mean_revert] → 3 dims
        - recent_win_rate: rolling 20-trade win rate
        - current_drawdown: % from peak equity
        
    Action (3-dim Box, continuous):
        - Δ confidence_threshold: [-0.1, +0.1] around base 50
        - Δ momentum_threshold: [-0.1, +0.1] around base 0.20
        - Δ risk_threshold: [-0.01, +0.01] around base 0.025
        
    Reward:
        - trade_pnl (normalized)
        - -0.5 * max_drawdown_penalty
        - +0.1 * rolling_sharpe
    """
    
    def __init__(self, prices: np.ndarray, features: np.ndarray, 
                 ensemble_preds: np.ndarray, config: dict):
        super().__init__()
        
        self.prices = prices
        self.features = features
        self.ensemble_preds = ensemble_preds
        self.config = config
        
        # Base thresholds (from config)
        self.base_confidence = 50.0
        self.base_momentum = 0.20
        self.base_risk = 0.025
        
        # Observation: [regime(3), win_rate(1), drawdown(1)] = 5 dims
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 0.5], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: threshold deltas (continuous)
        self.action_space = spaces.Box(
            low=np.array([-0.1, -0.1, -0.01], dtype=np.float32),
            high=np.array([0.1, 0.1, 0.01], dtype=np.float32),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 60  # Start after warmup
        self.equity = 100000.0
        self.peak_equity = self.equity
        self.trade_history = []
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Simplified regime detection from features
        regime = self._detect_regime()
        win_rate = self._calc_win_rate()
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        return np.array([*regime, win_rate, drawdown], dtype=np.float32)
    
    def step(self, action):
        # Apply threshold adjustments
        conf_thresh = self.base_confidence + action[0] * 100  # Scale to 0-100
        mom_thresh = self.base_momentum + action[1]
        risk_thresh = self.base_risk + action[2]
        
        # Simulate trade decision with adjusted thresholds
        trade_result = self._simulate_trade(conf_thresh, mom_thresh, risk_thresh)
        
        # Calculate reward
        reward = self._calculate_reward(trade_result)
        
        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        truncated = (self.peak_equity - self.equity) / self.peak_equity > 0.10
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _calculate_reward(self, trade_result):
        pnl = trade_result.get('pnl', 0)
        reward = pnl / self.equity * 100  # Normalized P/L
        
        # Drawdown penalty
        dd = (self.peak_equity - self.equity) / self.peak_equity
        if dd > 0.03:
            reward -= 0.5 * dd * 100
        
        # Sharpe component (rolling)
        if len(self.trade_history) > 5:
            returns = [t['pnl'] / self.equity for t in self.trade_history[-20:]]
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns)
                reward += 0.1 * sharpe
        
        return reward
```

**Training with SAC** (continuous actions):

```python
from stable_baselines3 import SAC

# SAC for continuous threshold optimization
policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[256, 256]))
model = SAC(
    "MlpPolicy", 
    env, 
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    buffer_size=50000,
    batch_size=256,
    gamma=0.99,
    device="cpu",  # CRITICAL: avoid Metal conflicts
    verbose=1
)
model.learn(total_timesteps=100000)
model.save("trained_data/models/sac_gate_thresholds")
```

---

### Phase 3: Optimal Exit Timing RL (Week 4-5)

**Goal**: Learn when to exit trades instead of fixed R:R ratios.

**New File**: `src/rl/optimal_exit_env.py`

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class OptimalExitEnv(gym.Env):
    """
    RL environment for learning optimal trade exit timing.
    
    Observation (Dict space, flattened for MLP):
        - unrealized_pnl: current P/L in pips
        - time_in_trade: bars since entry
        - momentum_since_entry: price momentum
        - atr_normalized: volatility context
        
    Action (Discrete):
        - 0: HOLD (continue position)
        - 1: EXIT_PROFIT (close with profit target)
        - 2: EXIT_LOSS (close with stop loss)
        
    Reward:
        - Realized P/L at exit
        - Small negative reward per bar held (opportunity cost)
    """
    
    def __init__(self, trade_scenarios: list):
        super().__init__()
        
        self.scenarios = trade_scenarios
        
        # Observation: [pnl, time, momentum, atr] = 4 dims
        self.observation_space = spaces.Box(
            low=np.array([-100, 0, -1, 0], dtype=np.float32),
            high=np.array([100, 100, 1, 5], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: discrete exit decisions
        self.action_space = spaces.Discrete(3)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.scenario_idx = self.np_random.integers(0, len(self.scenarios))
        self.scenario = self.scenarios[self.scenario_idx]
        self.step_in_trade = 0
        self.entry_price = self.scenario['entry_price']
        self.direction = self.scenario['direction']  # 1=long, -1=short
        self.position_open = True
        return self._get_obs(), {}
    
    def _get_obs(self):
        current_price = self.scenario['prices'][self.step_in_trade]
        pnl_pips = (current_price - self.entry_price) * self.direction * 10000
        momentum = self.scenario['momentum'][self.step_in_trade]
        atr = self.scenario['atr'][self.step_in_trade]
        return np.array([pnl_pips, self.step_in_trade, momentum, atr], dtype=np.float32)
    
    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        
        if action == 0:  # HOLD
            reward = -0.01  # Small holding cost
            self.step_in_trade += 1
            
            # Check if scenario ended
            if self.step_in_trade >= len(self.scenario['prices']):
                # Forced exit at end
                current_price = self.scenario['prices'][-1]
                pnl = (current_price - self.entry_price) * self.direction * 10000
                reward = pnl
                terminated = True
                
        else:  # EXIT (action 1 or 2)
            current_price = self.scenario['prices'][self.step_in_trade]
            pnl = (current_price - self.entry_price) * self.direction * 10000
            reward = pnl
            terminated = True
        
        return self._get_obs(), reward, terminated, truncated, {}
```

**Training with PPO** (discrete actions):

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    device="cpu",
    verbose=1
)
model.learn(total_timesteps=200000)
model.save("trained_data/models/ppo_optimal_exit")
```

---

### Phase 4: Integration & CLI (Week 6)

**Add CLI commands** to `main.py`:

```python
@cli.command()
@click.option('--timesteps', default=100000)
def train_rl_gates(timesteps):
    """Train RL gate threshold optimizer."""
    from src.rl.gate_threshold_env import GateThresholdEnv
    # ... training code

@cli.command()  
@click.option('--timesteps', default=200000)
def train_rl_exits(timesteps):
    """Train RL optimal exit timing."""
    from src.rl.optimal_exit_env import OptimalExitEnv
    # ... training code
```

**Add inference flags**:

```python
@click.option('--use-rl-gates/--no-rl-gates', default=False)
@click.option('--use-rl-exits/--no-rl-exits', default=False)
```

---

## 3. Algorithm Selection Guide

| Use Case | Algorithm | Action Space | Reason |
|----------|-----------|--------------|--------|
| Position sizing | **PPO** | Discrete(6) | Existing, works well for discrete choices |
| Gate thresholds | **SAC** | Box(3,) continuous | Needs smooth exploration, entropy bonus |
| Exit timing | **PPO** | Discrete(3) | Simple discrete choices, stable |
| Ensemble weights | **SAC** | Box(3,) continuous | Continuous weight optimization |

### SAC vs PPO Decision Matrix

```
Need continuous actions? ──Yes──> SAC
         │
         No
         │
         ▼
Need entropy-based exploration? ──Yes──> SAC
         │
         No
         │
         ▼
PPO (simpler, more stable)
```

---

## 4. Reward Engineering Best Practices

### Multi-Objective Reward Template

```python
def calculate_reward(self, trade_result: dict) -> float:
    # 1. Base: Normalized P/L
    pnl = trade_result['pnl']
    reward = pnl / self.initial_equity * 100
    
    # 2. Risk-adjusted: Sharpe component
    if len(self.returns) > 5:
        sharpe = np.mean(self.returns) / (np.std(self.returns) + 1e-8)
        reward += self.config.sharpe_weight * np.clip(sharpe, -2, 2)
    
    # 3. Drawdown penalty (progressive)
    dd = self.current_drawdown
    if dd > 0.03:
        reward -= self.config.drawdown_penalty * (dd ** 2) * 100
    
    # 4. Win rate bonus (above 50%)
    if self.win_rate > 0.5:
        reward += self.config.win_rate_bonus * (self.win_rate - 0.5) * 10
    
    # 5. Trade frequency regularization (avoid overtrading)
    if self.trades_today > 5:
        reward -= 0.1 * (self.trades_today - 5)
    
    return float(np.clip(reward, -10, 10))
```

---

## 5. Technical Constraints

### GPU/Metal Conflict Mitigation

```python
# CRITICAL: All RL models MUST use CPU
model = SAC("MlpPolicy", env, device="cpu")
model = PPO("MlpPolicy", env, device="cpu")

# Lazy loading pattern (from rl_position_sizing.py)
def _lazy_load_rl_model(path: str):
    """Lazy load to avoid TF/PyTorch GPU conflicts."""
    from stable_baselines3 import SAC
    return SAC.load(path, device="cpu")
```

### Data Requirements

| Component | Minimum Samples | Notes |
|-----------|----------------|-------|
| Position sizer | 500 trades | Already met with historical data |
| Gate thresholds | 1000 decision points | ~42 days of H1 data |
| Exit timing | 500 trade scenarios | Extract from historical trades |

---

## 6. Expected Performance Gains

| Enhancement | Complexity | Expected Sharpe Δ | Confidence |
|-------------|------------|-------------------|------------|
| RL position sizer (existing) | ✅ Done | +0.1 to +0.3 | High |
| RL gate thresholds | Medium | +0.2 to +0.5 | Medium-High |
| RL optimal exits | Medium | +0.3 to +0.6 | Medium |
| RL ensemble weights | Low | +0.1 to +0.2 | Medium |

**Total potential improvement**: +0.7 to +1.6 Sharpe ratio points

---

## 7. File Structure After Implementation

```
src/
├── rl/
│   ├── __init__.py
│   ├── gate_threshold_env.py      # NEW: SAC environment
│   ├── optimal_exit_env.py        # NEW: PPO environment  
│   ├── callbacks.py               # NEW: Training callbacks
│   └── utils.py                   # NEW: Shared utilities
├── core/
│   └── modular_inference.py       # MODIFY: Add RL gate integration
└── risk/
    └── position_sizing.py         # MODIFY: Add RL exit integration

trained_data/models/
├── rl_position_sizer.zip          # Existing
├── sac_gate_thresholds.zip        # NEW
└── ppo_optimal_exit.zip           # NEW
```

---

## 8. Testing Strategy

```bash
# Unit tests for environments
pytest tests/test_rl_environments.py -v

# Integration test: full training cycle
python -c "
from src.rl.gate_threshold_env import GateThresholdEnv
from stable_baselines3 import SAC
import numpy as np

# Mock data
env = GateThresholdEnv(
    prices=np.random.randn(1000).cumsum() + 100,
    features=np.random.randn(1000, 20),
    ensemble_preds=np.random.rand(1000, 4),
    config={}
)
model = SAC('MlpPolicy', env, device='cpu', verbose=0)
model.learn(total_timesteps=1000)
print('✓ SAC training works')
"
```

---

## 9. Next Steps

1. **Immediate**: Run `python main.py train-rl-sizer --timesteps 500000` to validate existing sizer
2. **Week 1**: Create `src/rl/` directory structure and `gate_threshold_env.py`
3. **Week 2**: Train SAC on gate thresholds, backtest vs fixed thresholds
4. **Week 3**: Create `optimal_exit_env.py`, train PPO
5. **Week 4**: Integrate both into inference pipeline
6. **Week 5**: A/B test on paper trading account
7. **Week 6**: Production deployment with monitoring

---

*Plan created using Context7 documentation for stable-baselines3 and gymnasium.*
