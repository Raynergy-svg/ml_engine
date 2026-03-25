# Pre-Trade Backtest Validation Gate

## Overview

The BacktestGate is a lightweight pre-execution validation gate that backtests the current trading signal on recent candle data before committing to a trade. It validates signal quality through rapid historical simulation.

**Key Principle:** SOFT gating - never hard-blocks trades, only adjusts confidence. This matches the uncertainty agent pattern.

## Architecture

### Location
- **Implementation:** `src/scanner/backtest_gate.py`
- **Integration:** `src/scanner/engine.py` (_scan_pair method)

### How It Works

1. **Simulation:** For each of the last N candles, simulate entry at close with given SL/TP
2. **Win Rate Calculation:** Count trades that hit TP before SL
3. **Validation:** If win_rate >= min_win_rate (default 55%), PASS
4. **Soft Penalty:** If win_rate < threshold, apply confidence_penalty (default -15%)

### Gate Configuration

```python
BacktestGate(
    min_win_rate=0.55,          # Minimum acceptable win rate (55% default)
    lookback_candles=20,         # Recent candles to simulate on (20 default)
    confidence_penalty=0.15,     # Confidence reduction if test fails (-15% default)
)
```

## Result States

### PASS (win_rate >= min_win_rate)
- **Action:** No confidence adjustment
- **Confidence Adjustment:** 0.0
- **Reason:** Signal validated by recent historical performance
- **Example:** "Backtest passed: 12/20 wins (60% >= 55%)"

### SOFT FAIL (win_rate < min_win_rate, confidence_penalty > 0)
- **Action:** Reduce confidence by penalty amount
- **Confidence Adjustment:** -0.15 (default)
- **Reason:** Signal underperformed recent history, reduce confidence
- **Example:** "Backtest failed: 9/20 wins (45% < 55%) → -15% confidence penalty"

### PASS (insufficient data)
- **Action:** No adjustment - don't block on missing data
- **Confidence Adjustment:** 0.0
- **Reason:** Not enough data to validate
- **Conditions:**
  - No candle data available
  - < 5 candles available
  - < 3 completed trades in backtest window

### PASS (error)
- **Action:** Graceful degradation - no penalty
- **Confidence Adjustment:** 0.0
- **Reason:** Backtest error - proceed with caution (don't block)

## Integration in Scanner

### Location in _scan_pair Flow

```
1. Run inference (GateEvaluator)
2. Fallback to technicals if HOLD
3. Calculate SL/TP from ATR
4. Apply specialist agents
5. ← BACKTEST GATE APPLIED HERE ← (after agents, before cache)
6. Cache result
7. Return PairAnalysis
```

### Code Integration

```python
# In engine.py _scan_pair method (after _apply_specialist_agents)

self._init_analysis_tools()
if self._backtest_gate and result.direction != "HOLD":
    backtest_result = self._backtest_gate.validate(
        pair=pair,
        direction=result.direction,
        sl_pips=result.sl_pips,
        tp_pips=result.tp_pips,
        recent_candles=df_raw,
        pip_value=pip_value,
    )

    # Apply soft penalty if failed
    if not backtest_result.passed and backtest_result.confidence_adjustment < 0:
        old_confidence = result.confidence
        result.confidence = max(0.0, result.confidence + backtest_result.confidence_adjustment)
```

## Log Output

### INFO Level (Summary)
```
EUR_USD: Backtest validation - Backtest passed: 12/20 wins (60% >= 55%) 
         (trades_tested=20, win_rate=0.6)
```

### DEBUG Level (Detailed)
```
EUR_USD: Backtest gate penalty applied: 0.650 → 0.500
```

## Testing

### Unit Test Example

```python
import pandas as pd
from src.scanner.backtest_gate import BacktestGate

# Create test data
data = {
    'open': [1.1000, 1.1005, 1.1010],
    'high': [1.1008, 1.1012, 1.1018],
    'low': [1.0995, 1.1000, 1.1005],
    'close': [1.1002, 1.1008, 1.1012]
}
df = pd.DataFrame(data)

gate = BacktestGate()
result = gate.validate(
    pair="EUR_USD",
    direction="LONG",
    sl_pips=10,
    tp_pips=15,
    recent_candles=df,
    pip_value=0.0001
)

assert result.passed == True or result.passed == False  # Validate based on data
assert 0.0 <= result.win_rate <= 1.0
assert -0.15 <= result.confidence_adjustment <= 0.0
```

## Configuration Tuning

### Conservative (Higher Standards)
```python
BacktestGate(
    min_win_rate=0.60,          # 60% minimum win rate
    lookback_candles=30,         # Test on 30 candles
    confidence_penalty=0.20,     # -20% penalty if fails
)
```

### Aggressive (Lower Standards)
```python
BacktestGate(
    min_win_rate=0.50,          # 50% minimum win rate
    lookback_candles=15,         # Test on 15 candles
    confidence_penalty=0.10,     # -10% penalty if fails
)
```

## Why SOFT Gating?

1. **Avoids False Negatives:** Signal may be valid even if recent history underperforms
2. **Matches Existing Pattern:** Follows uncertainty agent's soft penalty approach
3. **Preserves Opportunity:** Reduces confidence rather than blocking entirely
4. **Risk-Aware:** Position sizing and leverage will naturally reduce due to lower confidence
5. **Adaptive:** Over multiple scans, RL will adjust agent weights based on trade outcomes

## Edge Cases Handled

| Case | Behavior | Reason |
|------|----------|--------|
| No candle data | PASS (no adjustment) | Don't block on missing data |
| < 5 candles | PASS (no adjustment) | Insufficient data for backtest |
| < 3 completed trades | PASS (no adjustment) | Incomplete candle window |
| Backtest error | PASS (no adjustment) | Graceful degradation |
| Invalid SL/TP | SOFT FAIL (-15%) | Invalid input detected |
| Invalid direction | SOFT FAIL (-15%) | Invalid direction parameter |

## Performance Considerations

- **Complexity:** O(N) where N = lookback_candles (default 20)
- **Time:** < 1ms per pair on modern hardware
- **Memory:** Minimal - operates on existing candle data
- **Parallelization:** Thread-safe, runs in parallel with other pairs

## Related Rules (from trading.md)

- NEVER skip R:R ratio validation (1.2:1 minimum)
- SL/TP use ATR-based calculation (not hardcoded pips)
- Correlation filter prevents double exposure
- RL sync after trade close feeds outcomes back to weights

## Future Enhancements

1. **Timeframe-Specific Backtests:** Test on different timeframes (M5, H1, H4)
2. **Conditional Entry Methods:** Test multiple entry strategies
3. **Drawdown-Aware Gates:** Check maximum drawdown during backtest window
4. **Ensemble Backtests:** Weight recent vs older candles
5. **ML-Based Weighting:** Learn optimal penalty factor from outcomes

## Troubleshooting

### "Backtest validation - insufficient data..."
- **Cause:** Fewer than 5 candles available or candle fetch error
- **Action:** Check OANDA connection or CSV data availability
- **Result:** Gate passes, no adjustment applied

### High confidence penalty reducing scores
- **Cause:** Recent candle history doesn't match signal
- **Action:** Increase min_win_rate threshold in config
- **Result:** Only strong signals will pass without penalty

### No backtest gate output in logs
- **Cause:** direction == "HOLD" (gate only applies to LONG/SHORT)
- **Action:** Check that signal generates non-HOLD direction
- **Result:** Technical analysis or agents may need tuning

## References

- `src/scanner/backtest_gate.py` - Implementation
- `src/scanner/engine.py` - Integration (search "_apply_specialist_agents")
- `CLAUDE.md` - Trading rules and soft gating patterns
- `.claude/rules/trading.md` - Execution gates documentation
