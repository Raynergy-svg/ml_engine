# Full Pipeline Integration Test — Quick Reference

## What This Test Does

Validates all **11 automation modules** in ContinuousScanner._run_learning_loop() work together without crashes, cascade failures, or data corruption.

**Test File:** `testing/test_full_pipeline_integration.py`
**Status:** ✓ All 5 scenarios pass

---

## Quick Run

```bash
cd /sessions/magical-compassionate-einstein/mnt/ml_engine
python testing/test_full_pipeline_integration.py
```

**Expected:** All 5 tests pass in ~0.05 seconds

---

## The 11-Module Pipeline (What Gets Tested)

```
Input: closed_trade ──→
  │
  1. LearningEngine.analyze_trade(entry)
  │  └─→ Extract insights from outcome
  │
  2. AccuracyGate.record_outcome(pair, won)
  │  └─→ Track per-pair prediction accuracy
  │
  3. LearningEngine.update_pair_sl_tp(entry)
  │  └─→ Adapt SL/TP per pair
  │
  4. LearningEngine.check_promotions()
  │  └─→ Promote patterns to rules (if 3+ repetitions)
  │
  5. ConfigTuner.apply_to_config(config)
  │  └─→ Apply promoted rules to live config
  │
  6. ImprovementTracker.record_session(trades, learnings, rules)
  │  └─→ Log session metrics
  │
  7. StateEngine.save_state(goal, status, done, next)
  │  └─→ Persist session state
  │
  8. ObservationLog.log_observation(pair, category, description)
  │  └─→ Record market patterns
  │
  9. ModelManager.record_shadow_result(pair, incumbent, candidate, outcome)
  │  └─→ Record A/B test comparison
  │
 10. AlertManager.check_alerts(nav, peak_nav, trades, weights)
  │  └─→ Fire if thresholds breached
  │
 11. PortfolioOptimizer.rank_pairs(journal)
  │  └─→ Rank pairs by Sharpe, manage rotation
  │
  └──→ Output: config adjusted, state saved, alerts fired
```

---

## The 5 Test Scenarios

### Scenario A: Winning Trade
**Purpose:** Validate happy path (profitable trade flows through all modules)

**Setup:**
- 1 NZD_USD SHORT trade
- +$125 profit (50 pips)
- Consensus: 0.72 (high)

**Key Validations:**
✓ LearningEngine extracts "high_consensus_works" insight
✓ AccuracyGate records 100% accuracy
✓ ImprovementTracker logs +$125 P/L
✓ AlertManager fires 0 alerts (healthy)
✓ PortfolioOptimizer ranks the pair

---

### Scenario B: Losing Trades (Triggers Alerts)
**Purpose:** Validate alert firing on consecutive losses

**Setup:**
- 4 consecutive EUR_USD losses
- Total: -$275 (4 × -~$70)
- Accuracy: 0% (0/4 correct)

**Key Validations:**
✓ LearningEngine flags uncertainty warnings
✓ AccuracyGate blocks EUR_USD (0% accuracy)
✓ AlertManager FIRES: "consecutive_losses: 4 losses (threshold=3)"
✓ ImprovementTracker logs -$275 session
✓ No cascade failures despite losses

---

### Scenario C: Module Isolation
**Purpose:** Verify one module's failure doesn't crash the pipeline

**Setup:**
- Write corrupt JSON to pair_accuracy.json
- Try to load and use it

**Key Validations:**
✓ AccuracyGate logs warning but recovers
✓ All other modules initialize successfully
✓ Next record_outcome() works with fresh data
✓ No cascade failures

---

### Scenario D: First-Run (Empty State)
**Purpose:** Validate system handles missing/empty files on first run

**Setup:**
- All temp files missing
- Fresh initialization

**Key Validations:**
✓ All modules initialize with sensible defaults
✓ StateEngine loads default state
✓ AccuracyGate creates new data on first trade
✓ ObservationLog returns empty list
✓ No crashes or errors

---

### Scenario E: Portfolio Rotation (Sharpe-Based)
**Purpose:** Validate pair ranking and rotation logic

**Setup:**
- 51 trades across 4 pairs
- Varied P/L per pair

**Key Validations:**
✓ PortfolioOptimizer ranks pairs by Sharpe
✓ USD exposure constraint enforced (max 40%)
✓ AUD_USD demoted to observe-only
✓ EUR_USD, GBP_USD, NZD_USD remain active
✓ Correct constraint handling

---

## What Gets Tested For Each Module

| Module | Tested | Result |
|--------|--------|--------|
| LearningEngine | analyze_trade, update_pair_sl_tp, check_promotions | ✓ |
| AccuracyGate | record_outcome, check_pair, get_blocked_pairs | ✓ |
| ConfigTuner | apply_to_config | ✓ |
| ImprovementTracker | record_session | ✓ |
| StateEngine | save_state, load_state | ✓ |
| ObservationLog | log_observation, get_recent | ✓ |
| ModelManager | record_shadow_result | ✓ |
| AlertManager | check_all (fires consecutive_losses) | ✓ |
| PortfolioOptimizer | rank_pairs, get_active_pairs, get_observe_only_pairs | ✓ |

---

## Key Test Insights

### JSON Parsing Resilience
✓ AccuracyGate recovers from corrupt JSON without crash
```
[WARNING] Failed to load accuracy data: ... Starting fresh.
```

### Error Handling
✓ All modules have try/except guards
✓ No module crashes the pipeline
✓ Graceful degradation on missing files

### Production Readiness
✓ All return types match production (PairRanking objects, not dicts)
✓ All method signatures are correct
✓ No API mismatches

### Alert System
✓ Consecutive_losses fires at threshold=3 (4 losses trigger alert)
✓ Alert severity: WARNING
✓ Cooldown mechanism prevents spam

---

## File Structure

```
testing/
├── test_full_pipeline_integration.py      (Main test — 660 lines)
├── FULL_PIPELINE_INTEGRATION_TEST_REPORT.md  (Detailed findings)
└── FULL_PIPELINE_TEST_GUIDE.md            (This file)
```

---

## Important Notes

### No Production Code Modified
This test is **read-only**. It:
- Uses tempfile for all file I/O
- Never touches production .claude/ or trained_data/
- Never modifies live models or configs
- Pure validation (no side effects)

### Temp Files Cleaned Up
Each scenario creates its own temp directory which is cleaned up after the test.

### No Dependencies on Real Data
Mock trade entries use realistic structure but synthetic data.

---

## Running Individual Scenarios

To run a specific scenario:

```python
import unittest
from testing.test_full_pipeline_integration import TestFullPipelineIntegration

# Run one scenario
suite = unittest.TestLoader().loadTestsFromName(
    'test_full_pipeline_integration.TestFullPipelineIntegration.test_scenario_a_winning_trade_flow'
)
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
```

---

## Interpreting Output

### Success
```
ok
Ran 5 tests in 0.052s
OK

✓✓✓ ALL TESTS PASSED ✓✓✓
```

### Failure
If any module signature changes:
```
FAIL: test_scenario_x ...
TypeError: [ModuleName].__init__() got unexpected keyword argument 'xxx'
```
→ Update test with correct API

---

## Maintenance

When adding a new automation module:

1. Add import to test file
2. Create step in scenario (e.g., "Step 12: NewModule.do_thing()")
3. Add to pipeline diagram comment
4. Add entry to summary table
5. Run test → should PASS

---

## Performance

- **Runtime:** ~0.05 seconds for all 5 scenarios
- **Memory:** Minimal (tempfile cleanup)
- **File I/O:** All in /tmp (fast)

---

## See Also

- `FULL_PIPELINE_INTEGRATION_TEST_REPORT.md` — Detailed findings
- `.claude/rules/improvement.md` — Code quality gates used
- `src/scanner/automation/continuous.py` — Actual _run_learning_loop() implementation
