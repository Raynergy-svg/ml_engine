# Per-Pair Model Selector Guide

## Overview

The **PairModelSelector** is an autonomous system that automatically switches between joint and per-pair fine-tuned models based on rolling accuracy comparison. It enables adaptive model selection where each currency pair can run the best-performing model for that pair's unique market characteristics.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           Continuous Scanner (Scan Loop)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├─→ Record Predictions
                     │   (joint vs per_pair)
                     │
                     └─→ Learning Loop (5h)
                         │
                         ├─ Analyze trades
                         ├─ Update RLS weights
                         │
                         └─→ PairModelSelector.check_switch()
                             │
                             ├─ Compare rolling accuracy
                             ├─ Check min_trades threshold
                             │
                             └─→ execute_switch() if recommended
                                 │
                                 ├─ Update registry (JSON)
                                 ├─ Log switch event (JSONL)
                                 └─ Update config
```

## Core Components

### 1. PairModelRecord

Tracks accuracy metrics for a single pair:

```python
@dataclass
class PairModelRecord:
    pair: str                      # "EUR_USD"
    active_model: str              # "joint" or "per_pair"
    joint_accuracy: float          # Rolling accuracy of joint model
    per_pair_accuracy: float       # Rolling accuracy of per-pair model
    joint_trades: int              # Trades evaluated on joint
    per_pair_trades: int           # Trades evaluated on per-pair
    last_switch: str               # ISO timestamp
    reason: str                    # Why current model is active
```

### 2. PairModelSelector

Manages model selection for all pairs:

```python
selector = PairModelSelector(
    registry_path="trained_data/models/pair_model_registry.json",
    selections_log_path="trained_data/models/pair_model_selections.jsonl",
    models_dir="trained_data/models",
    rolling_window=30,             # EMA window
    min_trades=30,                 # Min trades before recommending switch
    switch_threshold=0.02,         # 2% improvement to switch
)
```

## Key Operations

### Recording Predictions

Record every prediction outcome to track accuracy:

```python
selector.record_prediction(
    pair="EUR_USD",
    model_type="joint",            # or "per_pair"
    predicted_direction="LONG",    # "LONG", "SHORT", "HOLD"
    actual_direction="LONG",       # What actually happened
)
```

**When to call:**
- After every model prediction
- Pass actual direction from trade outcome
- In learning loop: after trade closes

### Checking for Switch

Recommend a model switch when one outperforms the other:

```python
new_model = selector.check_switch("EUR_USD")
if new_model:  # Returns "joint" or "per_pair", or None
    selector.execute_switch("EUR_USD", new_model)
```

**Switch criteria:**
- Both models must have `min_trades` evaluations (default: 30)
- Winning model must exceed incumbent by `switch_threshold` (default: 2%)
- Example: joint=55% accuracy, per_pair=57% → no switch (only 2%, at threshold)
- Example: joint=55% accuracy, per_pair=60% → **switch to per_pair** (5% gain)

### Getting Status

View current model selections:

```python
status = selector.get_status()
# Returns: {"EUR_USD": {active_model: "joint", ...}, ...}

summary = selector.get_summary()
# Human-readable table

stats = selector.get_pair_stats("EUR_USD")
# Detailed stats including recommendation
```

## Integration with Scanner

The system is integrated into `ContinuousScanner._run_learning_loop()`:

```python
# 5h. Per-pair model selector: check for model switches
pms = PairModelSelector()
for pair in traded_pairs:
    new_model = pms.check_switch(pair)
    if new_model:
        pms.execute_switch(pair, new_model)
        # Log or notify user
```

### Workflow in Continuous.py

1. **Scanning phase**: Each scan records predictions
   - Per pair, per model (if using true A/B testing)
   - Stored in PairAnalysis

2. **Execution phase**: Trades are executed and outcomes recorded

3. **Learning phase** (Step 5h):
   - Load closed trades from journal
   - Extract pairs with outcomes
   - For each pair, check if model switch recommended
   - Execute switch if threshold met
   - Log switch event

## Rolling Accuracy (EMA)

The system uses **exponential moving average** to weight recent performance more heavily:

```
accuracy_new = alpha * current_result + (1 - alpha) * accuracy_old
alpha = 2 / (rolling_window + 1)
```

**Default (rolling_window=30):**
- `alpha = 2 / 31 ≈ 0.0645`
- Recent trades weighted 6.5x more than old trades
- Allows model recovery within ~30 recent trades

**Why EMA over simple rolling window?**
- Smooth degradation when model performance changes
- Avoids cliff-edge blocking (e.g., pair goes from 60% to 45% accuracy)
- Recent trades influence decision more than historical performance
- Retraining can unlock a blocked pair within window size

## Data Persistence

### Registry (JSON)

**File:** `trained_data/models/pair_model_registry.json`

```json
{
  "EUR_USD": {
    "pair": "EUR_USD",
    "active_model": "joint",
    "joint_accuracy": 0.5823,
    "per_pair_accuracy": 0.5501,
    "joint_trades": 45,
    "per_pair_trades": 42,
    "last_switch": "2026-03-19T10:15:00Z",
    "reason": "Switched from per_pair to joint: joint=0.582 per_pair=0.550"
  },
  "GBP_USD": { ... }
}
```

- Updated after every `record_prediction()`
- File-locked writes (safe for concurrent access)
- Survives session restarts

### Selections Log (JSONL)

**File:** `trained_data/models/pair_model_selections.jsonl`

Append-only audit trail:

```jsonl
{"timestamp": "2026-03-19T10:15:00Z", "pair": "EUR_USD", "old_model": "per_pair", "new_model": "joint", "joint_accuracy": 0.582, "per_pair_accuracy": 0.550, "joint_trades": 45, "per_pair_trades": 42, "reason": "Switched from per_pair to joint: ..."}
{"timestamp": "2026-03-19T10:45:00Z", "pair": "GBP_USD", "old_model": "joint", "new_model": "per_pair", ...}
```

Each line documents:
- When the switch occurred
- Which pair switched
- Accuracy values that triggered the decision
- Trade counts used in decision

### Per-Pair Models (Directory)

**Structure:** `trained_data/models/{PAIR}/`

```
trained_data/models/
├── joint/
│   ├── buddy_tf_joint.keras
│   └── buddy_tf_joint_metadata.json
├── EUR_USD/
│   ├── buddy_tf_EUR_USD.keras     (fine-tuned for EUR_USD)
│   └── metadata.json
├── GBP_USD/
│   └── buddy_tf_GBP_USD.keras
└── pair_model_registry.json
```

The selector checks for `{pair_dir}/*.keras` to determine if a per-pair model exists.

## Configuration

### Default Parameters

```python
rolling_window=30          # EMA lookback period
min_trades=30             # Min evals before recommending switch
switch_threshold=0.02     # 2% improvement threshold
```

### Customization

```python
# Conservative: require more evidence before switching
selector = PairModelSelector(
    min_trades=50,           # More trades required
    switch_threshold=0.05,   # 5% improvement needed
    rolling_window=20,       # Shorter window = faster adaptation
)

# Aggressive: switch more frequently
selector = PairModelSelector(
    min_trades=15,
    switch_threshold=0.01,   # 1% is enough
    rolling_window=50,       # Longer window = slower adaptation
)
```

## Practical Scenarios

### Scenario 1: New Pair with Per-Pair Model

1. Pair starts with `active_model="joint"` (default)
2. Both models are evaluated in parallel (via shadow testing or separate evaluation)
3. After 30+ trades:
   - If per_pair > joint + 2% → switch to per_pair
   - If per_pair better, remains better → switch confirmed
4. Subsequent trades use per_pair until accuracy degrades

### Scenario 2: Model Degradation

1. Pair running per_pair with 60% accuracy
2. Market regime shifts, per_pair accuracy drops to 45%
3. Joint model still at 55%
4. After 30+ recent trades showing this trend:
   - 55% > 45% + 2% → switch back to joint
   - Scanner uses joint for subsequent EUR_USD predictions

### Scenario 3: Retraining Unlock

1. Pair EUR_USD blocked (joint: 48%, per_pair: 47%)
2. Re-trained per_pair model uploaded to disk
3. Selector recommends per_pair, but accuracy still low
4. If per_pair retrained well:
   - New trades evaluated with retrained per_pair
   - If per_pair > 60%, > joint + 2% → switch
   - Previous low accuracy is weighted via EMA (not ignored, but recent wins matter more)

## Monitoring and Debugging

### Check Current Status

```python
selector = PairModelSelector()

# All pairs summary
print(selector.get_summary())

# Specific pair
stats = selector.get_pair_stats("EUR_USD")
print(f"Active: {stats['active_model']}")
print(f"Recommendation: {stats['recommended_model']}")
print(f"Joint accuracy: {stats['joint_accuracy']:.3f}")
print(f"Per-pair accuracy: {stats['per_pair_accuracy']:.3f}")
print(f"Trades: J={stats['joint_trades']}, P={stats['per_pair_trades']}")
```

### Reset a Pair

```python
# After retraining, start accuracy evaluation fresh
selector.reset_pair("EUR_USD")
# Pair reverts to active_model="joint", trades=0
```

### Audit Trail

```python
# Check when and why switches happened
import json
with open("trained_data/models/pair_model_selections.jsonl") as f:
    for line in f:
        event = json.loads(line)
        print(f"{event['timestamp']}: {event['pair']} "
              f"{event['old_model']} → {event['new_model']}")
```

## Limitations & Future Enhancements

### Current Limitations

1. **Static threshold**: Switch threshold is global (2% by default)
   - Future: Per-pair thresholds based on pair volatility

2. **No confidence weighting**: All predictions weighted equally
   - Future: Weight by model confidence score

3. **Blind to regime shifts**: EMA adapts slowly during regime changes
   - Future: Detect regime and reset evaluation

4. **No per-pair feature importance**: Unknown why per-pair is better
   - Future: Analyze feature attribution per pair

### Planned Enhancements

- [ ] Regime-aware threshold adjustment
- [ ] Confidence-weighted accuracy tracking
- [ ] Per-pair model feature importance analysis
- [ ] Automatic per-pair model retraining trigger
- [ ] Pair clustering (similar pairs → similar model performance)
- [ ] Multi-model ensemble selection (joint + per-pair + custom)

## Related Files

- **Scanner**: `src/scanner/engine.py` — loads active model per pair
- **Continuous Loop**: `src/scanner/automation/continuous.py` — calls selector in learning loop
- **Test Suite**: `tests/test_pair_model_selector.py` — comprehensive test coverage
- **Accuracy Gate**: `src/scanner/automation/accuracy_gate.py` — blocks low-accuracy pairs
- **Model Manager**: `src/scanner/automation/model_manager.py` — A/B testing and promotion

## Examples

### Basic Usage

```python
from src.scanner.automation import PairModelSelector

selector = PairModelSelector()

# After a trade closes, record outcome
selector.record_prediction(
    pair="EUR_USD",
    model_type="joint",
    predicted_direction="LONG",
    actual_direction="LONG",  # Correct!
)

# Periodically check for switches
new_model = selector.check_switch("EUR_USD")
if new_model:
    selector.execute_switch("EUR_USD", new_model)
    print(f"Switched EUR_USD to {new_model}")
```

### Integration in Scanner

```python
# In continuous.py learning loop
from src.scanner.automation import PairModelSelector

pms = PairModelSelector()
for pair in ["EUR_USD", "GBP_USD", "USD_JPY"]:
    # Record outcomes from closed trades
    pms.record_prediction(pair, "joint", pred, actual)

    # Check and execute switches
    new_model = pms.check_switch(pair)
    if new_model:
        pms.execute_switch(pair, new_model)
```

### Monitoring

```python
# Get status for all pairs
status = selector.get_status()
for pair, record in status.items():
    print(f"{pair}: {record['active_model']} "
          f"(J:{record['joint_accuracy']:.1%} "
          f"P:{record['per_pair_accuracy']:.1%})")
```

## Code Quality & Safety

- ✅ File-locked JSON writes (no corruption in concurrent access)
- ✅ Graceful handling of missing/corrupted registry
- ✅ Try/except wrapping on all I/O (never crashes scanner)
- ✅ Lazy import of dependencies (no circular imports)
- ✅ Comprehensive test coverage
- ✅ Full docstrings on all public methods
