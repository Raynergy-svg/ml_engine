# PairModelSelector — Quick Reference

## Import

```python
from src.scanner.automation import PairModelSelector
```

## Initialize

```python
selector = PairModelSelector(
    registry_path="trained_data/models/pair_model_registry.json",
    selections_log_path="trained_data/models/pair_model_selections.jsonl",
    models_dir="trained_data/models",
    rolling_window=30,      # EMA window
    min_trades=30,          # Min trades before recommending switch
    switch_threshold=0.02,  # 2% improvement to switch
)
```

## Core Methods

### Record Prediction

```python
selector.record_prediction(
    pair="EUR_USD",
    model_type="joint",           # or "per_pair"
    predicted_direction="LONG",   # Predicted: LONG/SHORT/HOLD
    actual_direction="LONG",      # Actual: LONG/SHORT/HOLD
)
```

### Check for Switch

```python
new_model = selector.check_switch("EUR_USD")
# Returns: "joint" | "per_pair" | None

if new_model:
    selector.execute_switch("EUR_USD", new_model)
```

### Get Status

```python
# All pairs
status = selector.get_status()
# Returns: {"EUR_USD": {record_dict}, "GBP_USD": {...}, ...}

# Summary table
print(selector.get_summary())

# Single pair details
stats = selector.get_pair_stats("EUR_USD")
print(f"Active: {stats['active_model']}")
print(f"Recommendation: {stats['recommended_model']}")
```

### Reset Pair

```python
selector.reset_pair("EUR_USD")  # Start fresh after retraining
```

## Switch Criteria

| Condition | Result |
|-----------|--------|
| `joint_trades < min_trades` | No switch |
| `per_pair_trades < min_trades` | No switch |
| Current: `joint`, Per-pair > joint + threshold | **Switch to per_pair** |
| Current: `per_pair`, Joint > per_pair + threshold | **Switch to joint** |
| Improvement < threshold | No switch |

**Example (default: min_trades=30, threshold=0.02):**
- Joint 55%, Per-pair 58% → **No switch** (only 3%, but need more data)
- Joint 55%, Per-pair 60% → **Switch to per-pair** (5% gain > 2% threshold)

## Data Files

| File | Purpose | Format |
|------|---------|--------|
| `pair_model_registry.json` | Current state (all pairs) | JSON |
| `pair_model_selections.jsonl` | Audit trail | JSONL (append-only) |
| `models/{PAIR}/` | Per-pair model files | .keras or .pkl |

## Continuous Scanner Integration

In `src/scanner/automation/continuous.py` (Step 5h):

```python
from src.scanner.automation import PairModelSelector

pms = PairModelSelector()
for pair in traded_pairs:
    new_model = pms.check_switch(pair)
    if new_model:
        pms.execute_switch(pair, new_model)
        # Log: "Switched {pair} from X to Y"
```

## Common Patterns

### Monitor All Pairs

```python
selector = PairModelSelector()
status = selector.get_status()

print("Pair Model Status:")
for pair, rec in status.items():
    print(f"  {pair:12} | {rec['active_model']:10} | "
          f"J:{rec['joint_accuracy']:.1%} P:{rec['per_pair_accuracy']:.1%}")
```

### Check Specific Pair Recommendation

```python
stats = selector.get_pair_stats("EUR_USD")
if stats['recommended_model']:
    print(f"Recommend switching {stats['pair']} to {stats['recommended_model']}")
else:
    print(f"{stats['pair']} should stay on {stats['active_model']}")
```

### View Switch History

```python
import json
with open("trained_data/models/pair_model_selections.jsonl") as f:
    for line in f:
        event = json.loads(line)
        print(f"{event['timestamp']} | {event['pair']} | "
              f"{event['old_model']} → {event['new_model']}")
```

### Reset After Retraining

```python
selector = PairModelSelector()

# After retraining EUR_USD per-pair model
selector.reset_pair("EUR_USD")
print("Reset EUR_USD — will evaluate from scratch")
```

## Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `rolling_window` | 30 | EMA lookback (smaller = faster adaptation) |
| `min_trades` | 30 | Min evaluations before recommending switch |
| `switch_threshold` | 0.02 | Min improvement ratio to switch (0-1) |

**Adjust for:**
- **Aggressive**: `min_trades=15, switch_threshold=0.01` (switch faster)
- **Conservative**: `min_trades=50, switch_threshold=0.05` (more evidence needed)

## Accuracy Calculation

Uses **exponential moving average** (EMA):

```
accuracy = alpha * current_result + (1 - alpha) * previous_accuracy
alpha = 2 / (rolling_window + 1)
```

- More recent predictions weighted higher
- Model recovery possible within rolling_window trades
- Smooth degradation (no cliff-edge blocking)

## File Safety

- ✅ File-locked writes (thread-safe)
- ✅ Graceful handling of missing/corrupted JSON
- ✅ Try/except on all I/O operations
- ✅ Never crashes the scanner

## Debugging

```python
# Check if per-pair model exists on disk
has_model = selector.has_per_pair_model("EUR_USD")

# Get detailed stats
stats = selector.get_pair_stats("EUR_USD")
print(f"Has per-pair model on disk: {stats['has_per_pair_model_on_disk']}")

# View full registry
import json
with open("trained_data/models/pair_model_registry.json") as f:
    registry = json.load(f)
    print(json.dumps(registry, indent=2))
```

## Test Coverage

Full test suite in: `tests/test_pair_model_selector.py`

Run tests:
```bash
python3 -m pytest tests/test_pair_model_selector.py -v
```

Test categories:
- Unit tests for all methods
- Persistence (JSON save/load)
- Concurrent access (threading)
- Integration workflows
- Edge cases (invalid inputs, missing files)

## Limitations

- Static global threshold (per-pair thresholds planned)
- No confidence weighting (planned)
- Slow adaptation during regime shifts (planned)
- No feature importance analysis (planned)

## Related Systems

- **AccuracyGate**: `src/scanner/automation/accuracy_gate.py` — blocks low-accuracy pairs
- **ModelManager**: `src/scanner/automation/model_manager.py` — A/B testing & promotion
- **LearningEngine**: `src/scanner/automation/learning_engine.py` — trade analysis
- **ContinuousScanner**: `src/scanner/automation/continuous.py` — main scan loop
