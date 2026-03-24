# Accuracy Gate User Guide

## Overview

The **AccuracyGate** system automatically tracks directional prediction accuracy for each currency pair during live trading. It blocks pairs whose prediction accuracy drops below a configurable threshold, preventing the system from trading deteriorated pairs.

## How It Works

### 1. Outcome Recording
Every time a trade closes, the learning engine automatically records:
- **Pair** (e.g., EUR_USD)
- **Predicted direction** (LONG or SHORT)
- **Actual outcome** (win/loss)
- **Confidence score** (0.0 to 1.0)
- **Timestamp** (ISO format)

This happens automatically in `LearningEngine.analyze_trade()` → AccuracyGate.record_outcome().

### 2. Accuracy Calculation
For each pair, accuracy is calculated as:
```
Accuracy = Wins / Total Trades
```

Example: EUR_USD has 6 wins and 4 losses across 10 trades = 60% accuracy

### 3. Auto-Blocking
Pairs are blocked when **both** conditions are met:
1. **Minimum trades threshold reached**: Pair has at least `min_trades` outcomes recorded (default: 5)
2. **Accuracy below threshold**: `accuracy < min_accuracy` (default: 55%)

Rationale: 55% is a conservative floor (better than 50% random chance). If a pair drops below this during live trading, it indicates the model has degraded.

### 4. Integration Points

#### In Learning Engine
```python
# File: src/scanner/automation/learning_engine.py
# After analyzing a closed trade, record outcome in accuracy gate
if self.accuracy_gate and pair and direction:
    self.accuracy_gate.record_outcome(
        pair=pair,
        predicted_direction=direction,
        actual_outcome=trade_won,
        confidence=confidence,
    )
```

#### In Continuous Scanner
```python
# File: src/scanner/automation/continuous.py (in _run_learning_loop)
# Merge accuracy-gated blocked pairs into scanner config every scan cycle
blocked_by_accuracy = ag.get_blocked_pairs()
if blocked_by_accuracy:
    original_blocked = set(self.scanner.config.blocked_pairs or [])
    new_blocked = set(blocked_by_accuracy)
    merged = list(original_blocked | new_blocked)
    self.scanner.config.blocked_pairs = merged
```

## Configuration

### Parameters
- **min_accuracy** (default: 0.55)
  - Minimum directional accuracy to remain unblocked
  - 0.55 = 55% (better than random 50%)
  - Adjust lower for volatile pairs, higher for strict filtering

- **min_trades** (default: 5)
  - Minimum completed trades before a pair can be blocked
  - Prevents blocking after a single loss streak
  - Adjust higher for conservative evaluation, lower for quick adaptation

- **data_path** (default: "trained_data/pair_accuracy.json")
  - Where to persist accuracy data
  - Human-readable JSON format with indent=2

### How to Adjust

Edit in `LearningEngine.__init__()`:
```python
self.accuracy_gate = AccuracyGate(min_accuracy=0.60, min_trades=10)
```

Or in `ContinuousScanner._run_learning_loop()`:
```python
ag = AccuracyGate(min_accuracy=0.60, min_trades=10)
```

## Usage Examples

### Check Pair Status

```python
from src.scanner.automation.accuracy_gate import AccuracyGate

ag = AccuracyGate()
is_allowed, accuracy, reason = ag.check_pair("EUR_USD")

if is_allowed:
    print(f"✓ EUR_USD can trade: {reason}")
else:
    print(f"✗ EUR_USD is blocked: {reason}")
```

### Get All Blocked Pairs

```python
blocked = ag.get_blocked_pairs()
print(f"Currently blocked: {blocked}")
# Output: Currently blocked: ['GBP_USD', 'AUD_USD']
```

### View Full Report

```python
report = ag.get_report()
print(report)

# Output:
# AccuracyGate Report (min_accuracy=55.0%, min_trades=5)
# ================================================================================
# EUR_USD      ✓ ALLOWED       62.0% ( 31/ 50) — accuracy 62.0% >= 55.0%
# GBP_USD      ✗ BLOCKED       42.0% ( 21/ 50) — accuracy 42.0% < 55.0% after 50 trades
# AUD_USD      ✗ BLOCKED       48.0% ( 24/ 50) — accuracy 48.0% < 55.0% after 50 trades
# ================================================================================
# Blocked pairs: 2 of 3 (67%)
#   GBP_USD, AUD_USD
```

### Get Pair Statistics

```python
stats = ag.get_pair_stats("EUR_USD")
if stats:
    print(f"EUR_USD: {stats['wins']}/{stats['total_trades']} = {stats['accuracy']:.1%}")
    print(f"Recent trades: {stats['recent_trades'][-5:]}")
```

### Reset a Pair (After Model Retraining)

```python
# After retraining EUR_USD model, clear old accuracy history
ag.reset_pair("EUR_USD")
print("EUR_USD accuracy history reset")
```

## Persistence

Accuracy data is stored in JSON format:

```json
{
  "EUR_USD": {
    "total": 50,
    "wins": 31,
    "accuracy": 0.62,
    "trades": [
      {
        "direction": "LONG",
        "outcome": true,
        "confidence": 0.75,
        "timestamp": "2026-03-18T09:30:00Z"
      },
      ...
    ]
  },
  "GBP_USD": {
    "total": 50,
    "wins": 21,
    "accuracy": 0.42,
    "trades": [...]
  }
}
```

## Error Handling

All AccuracyGate operations are wrapped in try/except blocks to never crash the scan loop:

- Invalid pair names are logged and skipped
- JSON load/save failures log warnings but don't halt operation
- Recording outcomes with invalid data is gracefully handled

## Integration with Existing Systems

### Blocked Pairs Flow
```
Trade closes → analyze_trade() → accuracy_gate.record_outcome()
    ↓
Every scan cycle → _run_learning_loop() → accuracy_gate.get_blocked_pairs()
    ↓
Merge with ScannerConfig.blocked_pairs
    ↓
scanner.scan() respects merged blocked_pairs list
```

### Console Output
When a pair is blocked:
```
[yellow]Accuracy gate blocking 2 pair(s): GBP_USD, AUD_USD[/yellow]
```

## Best Practices

1. **Don't block too early**: min_trades=5 is conservative. For more stable evaluation, use min_trades=10-20.

2. **Threshold selection**:
   - 0.55 = strict (requires ~55% directional accuracy)
   - 0.50 = very permissive (only blocks clearly broken pairs)
   - Recommendation: start at 0.55, adjust based on pair volatility

3. **Monitor blocked pairs**: Review why pairs are blocked. If accurate, it indicates model degradation. If spurious, increase min_trades.

4. **Reset after retraining**: If you retrain a pair's model, call `reset_pair()` to start fresh evaluation.

5. **Don't bypass**: If a pair is blocked by accuracy, there's usually a good reason. Allow it to rebuild confidence or retrain the model.

## Troubleshooting

**Q: A pair is blocked but I think it's working fine**
- Check the trade history: `ag.get_pair_stats("EUR_USD")`
- The accuracy calculation is unbiased. If blocked, accuracy is genuinely low.
- Consider increasing `min_trades` to let it build more history before blocking.

**Q: A pair isn't blocked but has bad accuracy**
- It may not have enough trades yet. Check `total_trades` vs `min_trades`.
- Pair may be above threshold by a small margin. Monitor closely.

**Q: Data file got corrupted**
- Delete `trained_data/pair_accuracy.json` to start fresh.
- The system will rebuild accuracy on next closed trades.

**Q: Blocked pairs aren't being skipped in scans**
- Verify `continuous.py` is running the learning loop (should be automatic).
- Check that `self.scanner.config.blocked_pairs` is being used in `engine.py`.

## Future Enhancements

1. **Pair-specific thresholds**: Different min_accuracy for different pair groups
2. **Rolling window accuracy**: Last N trades instead of all-time
3. **Confidence weighting**: Higher-confidence trades count more toward accuracy
4. **Auto-recovery**: Pairs unblock after N correct predictions
5. **Accuracy trends**: Alert on downward accuracy trends before hitting threshold

## Related Files

- `src/scanner/automation/accuracy_gate.py` — Core module
- `src/scanner/automation/learning_engine.py` — Integration point 1
- `src/scanner/automation/continuous.py` — Integration point 2
- `.claude/rules/trading.md` — Promotion rules (rule 4: "Higher weighted_vote_score correlates with better outcomes")
- `trained_data/pair_accuracy.json` — Persisted data (auto-created)
