# RL Weight Persistence Enhancements

This document describes the enhanced RL weight persistence system in `src/scanner/agents.py`.

## Overview

The enhanced system adds four major capabilities to prevent weight overfitting, enable rollback, and ensure robust weight management across sessions:

1. **Time-Based Decay** — Weights drift toward baseline over time
2. **Confidence Scaling** — Blend baseline and learned weights based on sample size
3. **Weight Snapshots** — Save checkpoints every 50 trades for rollback
4. **Corruption Recovery** — Validate and recover from NaN/inf values

## Weight Metadata Format

Enhanced `trained_data/models/agent_weights.json` now includes metadata:

```json
{
  "_global": { "trend": 1.15, "volatility": 1.07, ... },
  "NORMAL": { "trend": 1.12, ... },
  "HIGH": { "trend": 1.10, ... },
  "EXTREME": { "trend": 1.30, ... },
  "_meta": {
    "total_trades": 47,
    "trades_NORMAL": 25,
    "trades_HIGH": 22,
    "last_updated": "2026-03-19T10:00:00Z",
    "min_trades_per_regime": 10
  }
}
```

**Key fields:**
- `total_trades` — Total trades across all regimes (incremented at RL sync)
- `trades_REGIME` — Per-regime trade count (used for confidence thresholds)
- `last_updated` — ISO timestamp of last weight update (used for time decay)

## 1. Time-Based Decay

Prevents indefinite persistence of learned weights by drifting them toward baseline over time.

### When It Runs

- Automatically when `reload_learned_weights()` is called (typically once per session start)
- Logged as: `"Applied XXh time decay to agent weights (factor: YY%)"`

### Formula

```
decay_factor = min(hours_since_last_update / 2400, 0.5)
decayed_weight = learned_weight + (baseline_weight - learned_weight) * decay_factor
```

**Key characteristics:**
- 1% drift per 24 hours toward baseline
- Caps at 50% drift (never fully resets even after 50 days)
- Harmless to run repeatedly (applies only the first time in a session)

### Example

If a weight was 1.5, baseline is 1.15, and 48 hours have passed:
```
decay_factor = 48 / 2400 = 0.02
decayed = 1.5 + (1.15 - 1.5) * 0.02 = 1.493
```

## 2. Confidence Scaling

Blends baseline and learned weights based on trade count to prevent overfitting on small samples.

### When It Runs

- Automatically in `_apply_confidence_scaling()` during session start
- Also applied on-demand in `get_weights_for_regime()` at retrieval time

### Scaling Rules

| Trade Count | Confidence | Formula |
| --- | --- | --- |
| < 10 | 0.2 (20%) | `baseline * 0.8 + learned * 0.2` |
| 10-50 | Linear | `baseline * (1-n/50) + learned * (n/50)` |
| ≥ 50 | 1.0 (100%) | Use learned weights directly |

### Example

With 10 trades, 30% confidence:
```
confidence = 10 / 50 = 0.2
final_weight = baseline * 0.8 + learned * 0.2
```

With 60 trades, 100% confidence:
```
confidence = min(60 / 50, 1.0) = 1.0
final_weight = learned (no blending)
```

## 3. Weight Snapshots

Save a checkpoint of all weights every 50 trades, enabling rollback to specific points in training.

### Where Snapshots Live

```
trained_data/models/weight_snapshots/
├── 50.json
├── 100.json
├── 150.json
└── ...
```

Each snapshot contains:
```json
{
  "trade_count": 50,
  "timestamp": "2026-03-19T10:30:00Z",
  "weights": { ... full weight state ... }
}
```

### Automatic Cleanup

- Keeps the last 10 snapshots
- Older snapshots are automatically deleted
- This prevents disk bloat while preserving recent checkpoints

### Manual Usage

```python
from src.scanner.agents import ScannerAgentTeam

team = ScannerAgentTeam(config)

# List available snapshots
snapshots = team.list_weight_snapshots()
# Output: [(50, '2026-03-19T10:30:00Z'), (100, '2026-03-19T12:00:00Z'), ...]

# Rollback to trade count 50
team.load_weight_snapshot(50)

# Verify by checking weights
print(team._learned_weights["_meta"]["total_trades"])  # Should be 50
```

## 4. Corruption Recovery

Automatically detects and recovers from corrupt weight files.

### What Gets Recovered

- **NaN values** — Reset to baseline weight
- **Inf values** — Reset to baseline weight
- **Out-of-range** — Clamp to valid range `[0.05, 10.0]`
- **Non-numeric** — Reset to baseline weight
- **JSON parse errors** — Fall back to baseline weights

### Logging

All recoveries are logged:
```
ERROR: JSON parse error in agent_weights.json: ... Using baseline weights.
WARNING: Invalid value (inf) for volatility in HIGH. Resetting to baseline.
WARNING: Out-of-range value (50.0) for momentum in _global. Clamping to valid range.
```

## Integration Points

### 1. Session Startup

In your scanner initialization, weights are automatically loaded and decayed:

```python
from src.scanner.agents import ScannerAgentTeam

team = ScannerAgentTeam(config)
team.reload_learned_weights()  # Applies time decay + confidence scaling
```

### 2. RL Weight Updates

After every trade closes, weights are updated and snapshots saved:

```python
# In execution.py's RL sync loop:
team.update_weights_from_outcome(
    agent_verdicts=analysis.agent_verdicts,
    trade_won=(outcome == "TP"),
    regime=regime,
)
# ^ Automatically:
#   - Updates weights
#   - Increments total_trades
#   - Saves snapshot if trade_count % 50 == 0
```

### 3. Continuous Scanner

The watch loop calls weight decay each cycle:

```python
# In continuous.py's _run_smart_loop():
team.apply_weight_decay(decay_rate=0.02)  # 2% per scan cycle
```

## Monitoring & Debugging

### Check Current Weight Maturity

```python
meta = team._learned_weights.get("_meta", {})
total = meta.get("total_trades", 0)
confidence = min(total / 50.0, 1.0)
print(f"Training maturity: {int(confidence * 100)}% ({int(total)} trades)")
```

### View Weight History

```python
snapshots = team.list_weight_snapshots()
for trade_count, timestamp in snapshots:
    print(f"  {trade_count} trades @ {timestamp}")
```

### Validate Current Weights

```python
validated = team._validate_weights(team._learned_weights)
if validated:
    print("✓ Weights valid")
else:
    print("✗ Corruption detected and recovered")
```

### Check Decay Applied

The log will show:
```
INFO: Applied 48h time decay to agent weights (factor: 2%)
```

## Rules & Constraints

Per trading rules:

- Time decay is **non-negotiable** — prevents indefinite persistence
- Snapshots enable **safe rollback** to known-good checkpoints
- Confidence scaling **prevents overfitting** on small samples
- Corruption recovery is **graceful** — always falls back to baseline

## Performance Notes

- Time decay: `O(n)` where n = number of agents (~12-15)
- Confidence scaling: `O(n)` per regime retrieval
- Snapshots: `O(1)` save, cleanup once per 50 trades
- Validation: `O(n)` on load (only once per session)

All operations complete in <10ms.

## Troubleshooting

### Weights Not Updating

Check that `total_trades` is incrementing:
```python
meta = team._learned_weights["_meta"]
print(f"Total trades: {meta.get('total_trades')}")
```

If stuck at 0, ensure `update_weights_from_outcome()` is called after trades close.

### Snapshots Not Saving

Check directory permissions:
```bash
ls -la trained_data/models/weight_snapshots/
```

Must be writable by the trading bot process.

### Time Decay Not Applied

Ensure `reload_learned_weights()` is called at session start, not just `__init__`:
```python
team = ScannerAgentTeam(config)
team.reload_learned_weights()  # <- Required for time decay
```

### High Variance in Early Phase

This is expected. With <10 trades, learned weights are heavily blended with baseline (80/20 baseline/learned). As you accumulate trades, confidence increases and learned weights are favored. This is intentional to prevent overfitting.

