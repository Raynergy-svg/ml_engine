# US-292: Expectancy Tracking System

## Overview
Implemented autonomous expectancy tracking per agent per volatility regime. Tracks E = Win% × AvgWin + Loss% × AvgLoss with rolling windows (default 50 trades) to reframe agent optimization from win rate to risk-adjusted returns.

## Files Created
- `src/scanner/expectancy_tracker.py` — Core tracker class with atomic persistence
- `tests/test_expectancy_tracker.py` — 33 comprehensive unit tests

## Key Features

### 1. Expectancy Calculation
```python
E = Win% × AvgWin + Loss% × AvgLoss
```

Example: 60% win rate, avg win $100, avg loss -$80
```
E = 0.60 × 100 + 0.40 × (-80) = $28 (positive expectancy)
```

### 2. Weight Modifiers
- **Positive expectancy**: weight_modifier = 1.0 (full weight)
- **Negative expectancy**: weight_modifier = 1.0 + penalty (default 0.80 with -0.20 penalty)
- **Insufficient data** (< min_trades_for_calc): weight_modifier = 1.0 (conservative default)

### 3. Per-Regime Isolation
Each agent tracks separate expectancy per volatility regime:
- **LOW** → Uptrend context
- **NORMAL** → Balanced conditions
- **HIGH** → Elevated volatility
- **EXTREME** → Crisis mode

Same agent may be profitable in NORMAL but unprofitable in EXTREME.

### 4. Atomic State Persistence
- Writes to `.tmp` file first, then atomic `os.rename()` to final path
- Validates JSON structure on load (graceful degradation, no crash)
- Checks file freshness (warns if > 1 hour stale)
- Includes version field for forward compatibility

## API Reference

### ExpectancyTracker

```python
from src.scanner.expectancy_tracker import (
    ExpectancyTracker,
    ExpectancyConfig,
    create_default_expectancy_tracker
)

# Create with defaults
tracker = create_default_expectancy_tracker()

# Or customize config
config = ExpectancyConfig(
    window_size=100,           # rolling window depth
    negative_penalty=-0.25,    # stronger penalty
    min_trades_for_calc=10,    # higher threshold
    persistence_path="my/path/expectancy.json"
)
tracker = ExpectancyTracker(config)
```

### Recording Trades

```python
# After a trade closes, record outcome
tracker.record_trade(
    agent_name="trend",
    regime="NORMAL",
    pnl=150.50,        # realized P&L in account currency
    won=True           # True if hit TP, False if hit SL
)
```

### Querying Expectancy

```python
# Get expectancy for single agent/regime
result = tracker.get_expectancy("trend", "NORMAL")
print(f"Expectancy: ${result.expectancy:.2f}")
print(f"Win rate: {result.win_rate * 100:.1f}%")
print(f"Weight modifier: {result.weight_modifier}")
print(f"Is meaningful: {result.is_meaningful}")  # >= min_trades_for_calc

# Get weight modifier only
modifier = tracker.get_weight_modifier("trend", "NORMAL")
# Use in agent weight calculation: agent_weight *= modifier

# Get all expectancies (summary view)
all_exp = tracker.get_all_expectancies()
# Dict[agent_name][regime_name] -> ExpectancyResult

# Get top N agents in a regime (ranked by expectancy)
top = tracker.get_top_agents("NORMAL", n=5)
# Returns List[(agent_name, ExpectancyResult)]
for agent_name, result in top:
    print(f"{agent_name}: E=${result.expectancy:.2f}")
```

### State Management

```python
# Save state (atomic)
tracker.save_state()  # uses config.persistence_path
# or
tracker.save_state("/custom/path/expectancy.json")

# Load state
loaded = tracker.load_state()  # returns True if successful, False if missing/corrupted
if loaded:
    print("State restored")
else:
    print("No prior state, starting fresh")

# Serialize/deserialize
data = tracker.to_dict()  # JSON-serializable dict
tracker2 = ExpectancyTracker.from_dict(data)
```

## Integration with ScannerAgentTeam

In `src/scanner/agents/_team.py`, the expectancy tracker should be injected and used to:

1. **After RL weight update** (in `update_weights_from_outcome()`):
```python
# Record trade outcome to expectancy tracker
tracker.record_trade(
    agent_name=agent_name,
    regime=regime,
    pnl=realized_pl,
    won=trade_won
)

# Apply weight modifier
if tracker is not None:
    modifier = tracker.get_weight_modifier(agent_name, regime)
    current_weight *= modifier
    logger.info(f"Applied expectancy modifier {modifier} to {agent_name}")
```

2. **During execution.sync_closed_trades_rl()**:
```python
for entry in pending:
    # ... determine trade outcome ...

    # Record to expectancy tracker if available
    if expectancy_tracker is not None:
        try:
            tracker.record_trade(
                agent_name=agent_name,
                regime=regime,
                pnl=realized_pl,
                won=trade_won
            )
        except Exception as e:
            logger.debug(f"Expectancy tracking failed: {e}")
```

## Rules Applied

### From `.claude/rules/improvement.md`

✓ **JSON Safety Gates**
- Atomic write: .tmp file then os.rename()
- json.dumps with indent=2 and sort_keys=True
- Version field in persisted state
- Graceful fallback on load (never crash on corrupted file)

✓ **State Persistence Gates**
- Validates state file freshness (warns if > 1 hour old)
- Includes timestamp in persisted state
- load_state() returns False on missing/error (graceful)

✓ **Silent Exception Prevention**
- No bare except clauses
- All exceptions logged with context
- load_state() returns False on error (caller informed)

✓ **Test Coverage Gates**
- 33 unit tests covering all paths
- Edge cases: zero trades, all wins, all losses, single trade
- Persistence round-trip tested
- Mock-based (no external dependencies)

## Testing

Run all tests:
```bash
cd /sessions/clever-peaceful-knuth/mnt/ml_engine
python -m pytest tests/test_expectancy_tracker.py -v
```

All 33 tests pass:
```
PASSED: test_init_creates_all_agent_regime_windows
PASSED: test_record_trade_single
PASSED: test_record_trade_invalid_agent_raises
PASSED: test_record_trade_invalid_regime_raises
PASSED: test_expectancy_calculation_known_values
PASSED: test_weight_modifier_positive_expectancy
PASSED: test_weight_modifier_negative_expectancy
PASSED: test_weight_modifier_insufficient_data
PASSED: test_rolling_window_respects_max_size
PASSED: test_get_all_expectancies
PASSED: test_get_top_agents_ranks_by_expectancy
PASSED: test_get_top_agents_excludes_insufficient_data
PASSED: test_edge_all_wins
PASSED: test_edge_all_losses
PASSED: test_edge_single_trade
PASSED: test_edge_empty_window
PASSED: test_edge_negative_pnl_on_win
PASSED: test_to_dict_serialization
PASSED: test_from_dict_deserialization
PASSED: test_from_dict_invalid_data_raises
PASSED: test_from_dict_skips_unknown_agents
PASSED: test_from_dict_skips_unknown_regimes
PASSED: test_save_state_creates_file
PASSED: test_save_state_atomic_write
PASSED: test_save_state_creates_parent_dirs
PASSED: test_save_state_ioerror_cleans_up
PASSED: test_load_state_missing_file_returns_false
PASSED: test_load_state_corrupted_file_returns_false
PASSED: test_load_state_roundtrip
PASSED: test_create_default_expectancy_tracker
PASSED: test_create_default_with_config_override
PASSED: test_different_regimes_isolated
PASSED: test_same_agent_different_regimes
```

## Example Usage

```python
from src.scanner.expectancy_tracker import (
    ExpectancyTracker,
    ExpectancyConfig,
    create_default_expectancy_tracker
)

# Initialize tracker
tracker = create_default_expectancy_tracker()

# Simulate 100 trades across agents and regimes
import random
random.seed(42)

agents = ["trend", "mean_reversion", "volatility", "risk_sentinel"]
regimes = ["LOW", "NORMAL", "HIGH", "EXTREME"]

for _ in range(100):
    agent = random.choice(agents)
    regime = random.choice(regimes)

    # Bias some agents/regimes to be profitable
    if agent == "trend" and regime == "NORMAL":
        won = random.random() < 0.65  # 65% win rate
        pnl = random.uniform(80, 150) if won else random.uniform(-100, -50)
    else:
        won = random.random() < 0.45  # 45% default
        pnl = random.uniform(50, 100) if won else random.uniform(-80, -30)

    tracker.record_trade(agent, regime, pnl, won)

# Analyze results
print("\n=== Expectancy by Agent/Regime ===")
all_exp = tracker.get_all_expectancies()
for agent in agents:
    print(f"\n{agent}:")
    for regime in regimes:
        result = all_exp[agent][regime]
        if result.is_meaningful:
            print(f"  {regime:8} E=${result.expectancy:7.2f} "
                  f"WR={result.win_rate*100:5.1f}% "
                  f"Mod={result.weight_modifier:.2f}")

# Find best agents per regime
print("\n=== Top Agents per Regime ===")
for regime in regimes:
    top = tracker.get_top_agents(regime, n=2)
    print(f"\n{regime}:")
    for agent, result in top:
        print(f"  {agent}: E=${result.expectancy:.2f}")

# Persist
tracker.save_state("expectancy_snapshot.json")
print("\nState saved to expectancy_snapshot.json")
```

## Notes

- **Window size**: Default 50 trades per agent/regime. Adjust based on trade velocity.
- **min_trades_for_calc**: Default 5. Lower values make modifier unstable; higher values delay learning.
- **negative_penalty**: Default -0.20 (0.80 multiplier). Adjust to control downside of negative expectancy.
- **Not a replacement for RL weights**: Expectancy is a complementary signal. Weight modifiers should be applied *after* RL weight updates, not before.
- **Regime accuracy**: Depends on regime detection in Scanner. If regime is wrong at entry, expectancy will be biased.

## Future Enhancements

- [ ] Regime-specific thresholds for min_trades_for_calc
- [ ] Decay/forgetting older trades (exponential weighting)
- [ ] Correlation of expectancy with entry confidence (joint distribution)
- [ ] Multi-pair expectancy aggregation (which pairs are highest expectancy for which agents)
- [ ] Live dashboard: Top agents by regime, expectancy trends over time
