# RF Streak Risk Threshold Implementation

## Overview

This implementation resolves the issue where the RF (Random Forest) model's streak risk gate was blocking valid trades due to an overly restrictive threshold.

## Problem Statement

The RF model was predicting a streak probability of 0.93, which exceeded the previous threshold of 0.6 (60%), causing the risk gate to fail and block otherwise valid trade opportunities. This resulted in missed trading opportunities despite having strong signals from other models.

## Solution

### 1. Increased Default Threshold

**Changed:** `max_streak_prob` from `0.6` to `0.95`

**Location:** `src/core/modular_inference.py` line 195

**Rationale:**
- Aligns with the scanner's approach (which uses 0.98)
- Allows trades with streak probability ≤ 95% to pass
- Still maintains risk control by blocking extreme values (> 95%)
- User's specific case (0.93) now passes the gate

### 2. Added Configuration Support

**New Config Section:** `config/config_improved_H1.yaml`

```yaml
inference:
  # Gate 4: RandomForest risk (drawdown & streak)
  max_drawdown_pct: 0.025  # 2.5% max expected drawdown
  max_streak_prob: 0.95    # 95% max streak continuation probability
  
  # ... other gate thresholds
```

**Benefits:**
- Users can customize thresholds without modifying code
- Allows per-deployment tuning
- Easy to experiment with different values

### 3. CLI Integration

**Updated:** `cli/commands.py` - `buddy` command

The CLI now automatically loads inference configuration from the YAML file:

```python
# Load inference configuration from config file
if cfg and 'inference' in cfg:
    inf_cfg = cfg['inference']
    inference_config = InferenceConfig(
        max_streak_prob=inf_cfg.get('max_streak_prob', 0.95),
        # ... other parameters
    )
```

## Testing

### Test Coverage

Created comprehensive test suite: `tests/test_inference_config.py`

**11 tests covering:**
- Default values validation
- YAML config loading
- Streak risk gate behavior
- Old vs new threshold comparison
- Permissive mode flags
- All 8 gate thresholds

**Results:** All 11 tests passing ✅

### Specific Test Case

```python
# User's problematic case
streak_prob = 0.93

# Old threshold (0.6)
0.93 <= 0.6 = False  # ❌ BLOCKED

# New threshold (0.95)
0.93 <= 0.95 = True  # ✅ ALLOWED
```

## Impact Analysis

### What Changed

| Aspect | Before | After |
|--------|--------|-------|
| **Default Threshold** | 0.6 (60%) | 0.95 (95%) |
| **User's Case (0.93)** | ❌ BLOCKED | ✅ ALLOWED |
| **Config Override** | Not available | Available via YAML |
| **Test Coverage** | None | 11 tests |

### What Stayed the Same

- Scanner still uses permissive threshold (0.98)
- Extreme values (>0.95) still blocked
- All other gates unchanged
- No breaking changes to API

### Trade-offs

**Pros:**
- More trades can pass the risk gate
- Reduces false negatives
- Configurable per deployment
- Better alignment with scanner

**Cons:**
- Slightly higher risk tolerance
- May allow some marginally riskier trades

**Mitigation:**
- Still blocks extreme values (>0.95)
- Other 7 gates still provide protection
- Users can adjust via config if too permissive

## Usage

### Default Behavior

No action required - the new threshold (0.95) is now the default.

### Customizing Thresholds

Edit `config/config_improved_H1.yaml`:

```yaml
inference:
  max_streak_prob: 0.90  # Use 90% if you want to be more conservative
  # OR
  max_streak_prob: 0.98  # Use 98% for maximum permissiveness (like scanner)
```

Then run:

```bash
./bin/Buddy EUR_USD
```

The CLI will automatically load your custom threshold.

### Verifying Configuration

Run the test suite to verify everything works:

```bash
pytest tests/test_inference_config.py -v
```

## Documentation Updates

1. **`.github/copilot-instructions.md`**
   - Updated Gate 4 documentation
   - Split into 4a (drawdown) and 4b (streak)
   - Updated threshold table

2. **This Document**
   - Complete implementation summary
   - Usage instructions
   - Impact analysis

## Future Considerations

### Potential Improvements

1. **Dynamic Thresholds**
   - Adjust threshold based on market conditions
   - Use volatility-adjusted thresholds
   - Implement adaptive risk management

2. **Threshold Optimization**
   - Backtest different threshold values
   - Find optimal threshold per pair
   - Use ML to learn optimal thresholds

3. **Enhanced Monitoring**
   - Track how often streak gate blocks trades
   - Log streak probability distribution
   - Alert on suspicious patterns

### Monitoring Recommendations

Keep an eye on:
- Trade frequency changes
- Win rate impact
- Drawdown patterns
- Streak probability distribution

If you see adverse effects, you can:
1. Lower the threshold in the config file
2. Enable additional risk controls
3. Adjust other gate thresholds

## Summary

This implementation successfully resolves the RF streak risk blocking issue by:

1. ✅ Increasing default threshold from 0.6 to 0.95
2. ✅ Adding configuration support via YAML
3. ✅ Integrating with CLI for automatic loading
4. ✅ Providing comprehensive test coverage
5. ✅ Updating all documentation

The user's specific case (streak_prob = 0.93) now passes the risk gate, while still maintaining appropriate risk controls for extreme values.

---

**Version:** 1.0  
**Date:** 2026-02-07  
**Status:** Implemented and Tested ✅
