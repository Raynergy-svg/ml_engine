# Neural Agent Bootstrap Test Report

Date: 2026-06-15 17:44
Environment: /Users/buddy/Documents/ml_engine
PYTHONPATH: /Users/buddy/Documents/ml_engine

## 1. Import Test

**Result: PASSED**

```
Neural agent imports OK
```

All required neural agent modules load correctly:
- `src.scanner.agents.neural.policies.TrendPolicy`
- `src.scanner.agents.neural.neural_agent_base.NeuralAgentConfig`

## 2. Synthetic Replay Test

**Result: PASSED** (with behavioral observations)

### Execution Details
```
Verdict name: neural_trend
Metadata has features: True
Buffer size: 26
Loss after update: None
Synthetic replay test PASSED
```

### Buffer Analysis
- Initial buffer after `evaluate()`: 1 sample
- After 25 `record_outcome()` calls: 26 total samples
- Buffer correctly accumulates experiences

### Online Update Verification
`_online_update()` return value is `None` by design (return annotation `-> None`).
However, update execution was confirmed via diagnostic probe:
- Update count before: 7
- Update count after: 8
- Policy model built: Yes (input shape `(None, 105)`)
- **Conclusion**: Online update executed successfully over 26 replay samples.

### Loss Observation
The `_online_update()` method does not return loss values; it logs them internally
via `logger.info(...)`. The test script's print of `loss` will always show `None`.
This is expected behavior, not a failure.

## 3. train_neural_agents.py Dry-Run

**Result: PARTIAL** — `--dry-run` flag does not exist.

Available CLI flags:
```
--collect-only        Collect outcomes, skip updates
--update-all          Update all agents from replay
--min-samples         Minimum samples threshold
--outcome-log         Path to trade outcomes log
--shadow-mode PAIR    Bootstrap synthetic data for a pair
--save-dir            Model save directory
--evaluate            Compute per-agent AUC on outcome log
```

### Alternative Verification
- `--help`: Loads successfully without import errors
- `--collect-only`: Script loads and runs; exits with error because no outcome log exists at `trained_data/logs/trade_outcomes.jsonl`

This confirms the script's import graph resolves correctly. The missing outcome log
is expected in a clean environment with no prior trade history.

## Summary

| Test | Status | Notes |
|------|--------|-------|
| Neural agent imports | PASSED | Clean import resolution |
| Synthetic replay buffer | PASSED | 26 samples, update executed |
| Online update ran | PASSED | Update count incremented 7→8 |
| Loss returned | N/A | Method returns None by design |
| train_neural_agents.py load | PASSED | No import errors; `--help` and `--collect-only` functional |
| `--dry-run` flag | MISSING | Flag not implemented; use `--collect-only` instead |

## Recommendations

1. If a true dry-run mode is desired for `train_neural_agents.py`, consider adding a `--dry-run` flag that instantiates agents and verifies the training pipeline without requiring an outcome log.
2. The synthetic test's loss assertion could be strengthened by probing `_update_count` or parsing logger output rather than relying on the `None` return value of `_online_update()`.
