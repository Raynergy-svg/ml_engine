# Drift Detection and Retrain Trigger System

## Overview

The drift detection system automatically monitors model prediction accuracy during live trading and triggers retraining when performance degrades below acceptable thresholds. This prevents trading with deteriorated models and ensures continuous improvement.

## Architecture

```
ContinuousScanner (every 50 scans)
    └─> RetrainTrigger
            ├─> Records prediction outcomes from closed trades
            ├─> Calculates rolling accuracy per pair
            ├─> Detects per-pair and global drift
            └─> Writes retrain_request.json → scheduled_retrain.py
```

## Components

### 1. RetrainTrigger (`src/scanner/automation/retrain_trigger.py`)

Core drift detection engine.

**Key Methods:**

- `record_prediction(pair, correct)` — Record a trade outcome for accuracy tracking
- `get_pair_accuracy(pair)` — Get rolling accuracy for a specific pair (returns None if < 10 samples)
- `get_global_accuracy()` — Get overall accuracy across all pairs (returns None if < 20 samples)
- `check_drift()` — Check if drift detected, trigger retrain if needed
- `get_stats()` — Return current monitoring statistics

**Configuration:**

```python
trigger = RetrainTrigger(
    per_pair_threshold=0.52,      # Block individual pairs below this
    global_threshold=0.53,         # Retrain all on global drift
    rolling_window=50,             # Keep last 50 predictions per pair
    cooldown_hours=24,             # Wait 24h after retrain before checking again
)
```

**Thresholds:**

- `per_pair_threshold=0.52` — If a pair's rolling accuracy drops below 52%, it's added to retrain request
- `global_threshold=0.53` — If overall accuracy drops below 53%, ALL pairs are added to retrain request
- These thresholds are intentionally tight (barely above 50% random baseline) to catch degradation quickly

### 2. Integration in ContinuousScanner (`src/scanner/automation/continuous.py`)

Added to the learning loop (step 5h) after accuracy gate merging.

**Triggers every 50 scan cycles:**

```python
if self._scan_count % 50 == 0:
    drift_request = trigger.check_drift()
    if drift_request:
        # Write retrain_request.json for scheduled_retrain.py to pick up
        # Log observation to ObservationLog
```

**Minimum sample sizes:**

- Per-pair accuracy: Requires 10+ predictions before evaluation
- Global accuracy: Requires 20+ predictions before evaluation
- This prevents false positives early in trading

### 3. RetrainRequest (`src/scanner/automation/retrain_trigger.py`)

Data class capturing drift event:

```python
@dataclass
class RetrainRequest:
    pairs: list                    # Pairs to retrain
    reason: str                    # "Drift detected: 3 pair(s) below accuracy threshold"
    accuracy_snapshot: dict        # Current accuracy per pair
    triggered_at: str              # ISO timestamp
    priority: str                  # "urgent" or "normal"
```

### 4. Integration in scheduled_retrain.py (`scripts/scheduled_retrain.py`)

**New function:** `check_drift_retrain_request()`

On startup, checks if `trained_data/retrain_request.json` exists:
- If found, prioritizes those pairs for retraining
- Logs which pairs came from drift trigger vs defaults vs explicit args
- Command-line `--pairs` argument always takes highest priority

**After successful retrain:**

```python
if success:
    # Clear the drift request so next trigger can create a new one
    request_path.unlink()
```

## Workflow

### Normal Operation

1. **Trading Loop** — ContinuousScanner runs scans and executes trades
2. **Closed Trades** — After each trade closes, ExecutionManager syncs outcomes
3. **Accuracy Recording** (every scan) — RetrainTrigger.record_prediction() logs if direction was correct
4. **Drift Check** (every 50 scans) — RetrainTrigger.check_drift() evaluates rolling accuracy
5. **Trigger Alert** — If drift detected:
   - Write `retrain_request.json` with priority and affected pairs
   - Log observation to `observations.jsonl`
   - Display console alert: `🔄 RETRAIN TRIGGER: X pair(s) below accuracy threshold (priority=urgent)`

### Retraining Pipeline

1. **Scheduled Job** — `scheduled_retrain.py` runs (e.g., Mon/Wed/Fri 10 AM UTC)
2. **Check for Request** — `check_drift_retrain_request()` looks for `retrain_request.json`
3. **Pair Selection**:
   - If drift request exists → use those pairs
   - Else if `--pairs` argument provided → use those
   - Else → use defaults
4. **Joint Training** — Train ensemble models for selected pairs
5. **Validation** — Verify required models exist
6. **Clear Request** — Delete `retrain_request.json` on success

## File Locations

| File | Purpose |
|------|---------|
| `src/scanner/automation/retrain_trigger.py` | Core drift detection engine |
| `trained_data/retrain_request.json` | Drift request (created by trigger, read by scheduled_retrain.py) |
| `trained_data/retrain_all_summary.json` | Retrain status (used for cooldown check) |
| `trained_data/trade_journal_rl.json` | Source of prediction outcomes |
| `trained_data/observations.jsonl` | Observation log (includes drift triggers) |

## Cooldown Period

After retraining completes, a cooldown period prevents redundant triggers:

- **Default:** 24 hours
- **Checked at:** Every 50-scan drift detection
- **Determined by:** Timestamp in `retrain_all_summary.json`

This prevents thrashing if models briefly dip after retraining.

## Accuracy Calculation

**Rolling Window (What gates use):**
- Last 50 predictions per pair
- Minimum 10 samples before evaluation
- Once sufficient data exists, new retraining unlocks a pair within rolling_window trades

**All-Time (For reference only):**
- All historical predictions
- Shown in reports but not used for gating

## Thresholds vs Accuracy Gate

| System | Purpose | Threshold | Action |
|--------|---------|-----------|--------|
| **Accuracy Gate** (`accuracy_gate.py`) | Prevent trading with bad models | 0.55 (55%) | Blocks pair from trading |
| **Drift Trigger** (`retrain_trigger.py`) | Detect degradation | 0.52 (52%) | Triggers retraining |

Drift trigger is intentionally lower (52%) to detect problems before trading is blocked.

## Observation Logging

When drift is detected, it's logged to `observations.jsonl`:

```json
{
  "timestamp": "2026-03-19T10:30:00+00:00",
  "pair": "_ALL",
  "category": "retrain_trigger",
  "description": "Drift detected: 3 pair(s) below accuracy threshold",
  "metadata": {
    "EUR_USD": {"accuracy": 0.51, "trades": 45},
    "GBP_USD": {"accuracy": 0.48, "trades": 32},
    "_global": {"accuracy": 0.50}
  }
}
```

## Example: Drift Trigger in Action

**Scan #245 (at 5-minute intervals, this is ~20 hours in):**

```
Accuracy snapshot after 200+ trades:
  EUR_USD: 48 correct / 102 trades = 47.1% (below 52% threshold)
  GBP_USD: 51 correct / 98 trades = 52.0% (at threshold, OK)
  USD_JPY: 45 correct / 92 trades = 48.9% (below threshold)
  Global: 165 correct / 350 trades = 47.1% (below 53% threshold)

Actions:
  ✓ Drift detected for EUR_USD, USD_JPY
  ✓ Global accuracy below threshold
  ✓ Trigger priority: "urgent" (global drift)
  ✓ Write retrain_request.json with pairs=[EUR_USD, GBP_USD, USD_JPY, ...]
  ✓ Log observation to observations.jsonl
  ✓ Console: "🔄 RETRAIN TRIGGER: Drift detected: 15 pair(s) below accuracy threshold (priority=urgent)"

Next scheduled retrain (48 hours later):
  ✓ check_drift_retrain_request() finds request
  ✓ Retrains 15 pairs (from urgent request)
  ✓ Validates models
  ✓ Deletes retrain_request.json on success
  ✓ Cooldown begins (24 hours)
```

## Testing

### Unit Test

```python
from src.scanner.automation.retrain_trigger import RetrainTrigger

trigger = RetrainTrigger()

# Simulate 15 trades: 7 correct, 8 wrong (47% accuracy)
for _ in range(7):
    trigger.record_prediction("EUR_USD", True)
for _ in range(8):
    trigger.record_prediction("EUR_USD", False)

acc = trigger.get_pair_accuracy("EUR_USD")
assert acc == 0.4667
assert acc < trigger.per_pair_threshold  # Should trigger
```

### Integration Test

Monitor the learning loop output during watch mode:

```
Scan #250:
  ✓ Accuracy gate: EUR_USD blocked, GBP_USD allowed
  [magenta]Learning: 5 insights captured, 0 rules promoted[/magenta]
  [red]🔄 RETRAIN TRIGGER: Drift detected: 3 pair(s) below accuracy threshold (priority=urgent)[/red]
```

## Best Practices

1. **Monitor per-pair accuracy** — Use `accuracy_gate.get_report()` to track trends
2. **Check observation log** — Grep for `"category": "retrain_trigger"` to see historical drift events
3. **Review scheduled retrain logs** — Check logs/retrain_*.log for success/failure details
4. **Verify models after retrain** — Ensure new models are loading in scanner (check for "model loaded")
5. **Set appropriate thresholds** — 52%/53% works for most FX pairs; adjust if trading systematically better/worse

## Troubleshooting

**Q: Drift never triggers even though accuracy is low?**
- Check minimum sample count: Need 10+ samples per pair or 20+ globally
- Check cooldown: If recently retrained, drift check is skipped
- Verify prediction outcomes are being recorded in trade_journal_rl.json

**Q: Retrain runs but doesn't clear the request?**
- Retrain must complete successfully (status="success")
- Check logs/retrain_*.log for error details
- Verify write permissions on trained_data/

**Q: Same pairs keep getting triggered over and over?**
- May indicate fundamental model issues (not just drift)
- Check if new data has structural changes (volatility, correlations)
- Consider manual inspection of recent closed trades for patterns

## References

- **Accuracy Gate:** `docs/ACCURACY_GATE_GUIDE.md`
- **Learning Engine:** `docs/LEARNING_ENGINE.md` (if available)
- **Trading Rules:** `.claude/rules/trading.md`
