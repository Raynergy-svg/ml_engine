# ML Engine Learning Loop Integration Test Suite

## Overview

This comprehensive test suite validates the entire ML Engine learning loop pipeline — from trade outcome analysis through rule promotion to config adaptation. The test suite is **production-safe**: it uses `tempfile` for all file I/O and never touches production `.claude/` or `trained_data/` directories.

## Test Structure

```
test_learning_loop_integration.py (36 tests, 8 test classes)
├── TestImports (7 tests)
│   └── Verify all automation modules import without errors
├── TestLearningEngine (5 tests)
│   └── Trade analysis, learning extraction, file I/O
├── TestAccuracyGate (5 tests)
│   └── Per-pair accuracy tracking and auto-blocking
├── TestConfigTuner (5 tests)
│   └── Rule-based config adjustments with bounds enforcement
├── TestStateEngine (5 tests)
│   └── Cross-session state persistence and recovery
├── TestImprovementTracker (4 tests)
│   └── Session metrics and trend analysis
├── TestObservationLog (4 tests)
│   └── Market observation logging and filtering
└── TestIntegration (1 test)
    └── End-to-end learning loop pipeline validation
```

## Running the Tests

### All Tests
```bash
cd /sessions/magical-compassionate-einstein/mnt/ml_engine
python testing/test_learning_loop_integration.py
```

### Expected Output
```
Ran 36 tests in ~0.03s

OK

======================================================================
TEST SUMMARY
======================================================================
Tests run: 36
Successes: 36
Failures: 0
Errors: 0
======================================================================
```

## Module Coverage

### 1. LearningEngine
**File:** `src/scanner/automation/learning_engine.py`

**What it does:**
- Analyzes closed trades and extracts actionable learning patterns
- Promotes recurring patterns (3+ occurrences) to trading rules
- Records insights in `.claude/learnings.md`
- Integrates with AccuracyGate for per-pair accuracy tracking

**Tests verify:**
- ✅ Winning trade analysis (high consensus detection)
- ✅ Losing trade analysis (uncertainty warnings)
- ✅ Malformed data handling (None agents dict)
- ✅ Open trade handling (no outcome)
- ✅ File I/O and persistence

### 2. AccuracyGate
**File:** `src/scanner/automation/accuracy_gate.py`

**What it does:**
- Tracks directional prediction accuracy per currency pair
- Auto-blocks pairs whose accuracy falls below threshold (default 55%)
- Prevents premature blocking with min_trades safeguard
- Persists accuracy data to JSON

**Tests verify:**
- ✅ Outcome recording and persistence
- ✅ Below-min-trades protection
- ✅ High-accuracy pair allowance (100%)
- ✅ Low-accuracy pair blocking (0%)
- ✅ Blocked pairs list generation

**Key API:**
```python
gate = AccuracyGate(min_accuracy=0.55, min_trades=5)
gate.record_outcome(pair="EUR_USD", predicted_direction="LONG", actual_outcome=True)
is_allowed, accuracy, reason = gate.check_pair("EUR_USD")
blocked_pairs = gate.get_blocked_pairs()
```

### 3. ConfigTuner
**File:** `src/scanner/automation/config_tuner.py`

**What it does:**
- Reads promoted rules from `.claude/rules/trading.md`
- Applies adjustments to ScannerConfig fields
- Enforces bounds to prevent runaway tuning
- Deduplicates rules via MD5 hashing

**Tests verify:**
- ✅ Rule loading from empty/non-existent file
- ✅ ATR SL multiplier adjustment (+0.1)
- ✅ Uncertainty threshold adjustment (-0.02)
- ✅ Bounds enforcement (min/max clamping)
- ✅ Deduplication within session

**Key API:**
```python
tuner = ConfigTuner()
rules = tuner.load_rules()  # Parse rules from file
adjustments = tuner.apply_to_config(config)  # Apply to ScannerConfig
```

**Bounds Enforcement:**
```python
BOUNDS = {
    "atr_sl_multiplier": (0.5, 2.0),
    "atr_tp_multiplier": (0.8, 3.0),
    "max_uncertainty_score": (0.30, 0.95),
    # ... more fields
}
```

### 4. StateEngine
**File:** `src/scanner/automation/state_engine.py`

**What it does:**
- Persists trading state to `.claude/state.json` for cross-session continuity
- Tracks goal, status, completed tasks, next action
- Stores portfolio snapshot (NAV, open trades, P/L)
- Increments scan cycle counter

**Tests verify:**
- ✅ Load missing state (returns defaults)
- ✅ Save and load round-trip
- ✅ Corrupted JSON handling (exception caught, defaults applied)
- ✅ Partial state merging with defaults
- ✅ Scan cycle counter incrementation

**Key API:**
```python
engine = StateEngine()
state = engine.load_state()  # Load with fallback to defaults
engine.save_state(
    goal="Improve accuracy",
    status="in_progress",
    done=["task1", "task2"],
    next_action="Continue testing"
)
cycle_count = engine.increment_scan_cycle()
```

### 5. ImprovementTracker
**File:** `src/scanner/automation/improvement_tracker.py`

**What it does:**
- Records session-level metrics to JSONL log
- Calculates win rates, P/L, learning velocity
- Detects trends (stable/improving/declining)
- Generates improvement reports

**Tests verify:**
- ✅ Session recording with 3 closed trades
- ✅ Trend detection with insufficient data
- ✅ Trend calculation (stable performance)
- ✅ Report generation

**Key API:**
```python
tracker = ImprovementTracker()
tracker.record_session(trades=[...], learnings_added=2, rules_promoted=1)
trend = tracker.get_trend(window=10)  # {'win_rate_trend': 'stable', ...}
report = tracker.generate_report()  # Multi-line string
```

### 6. ObservationLog
**File:** `src/scanner/automation/observation_log.py`

**What it does:**
- Logs interesting market observations (even without trades)
- Captures regime changes, disagreement, near-misses
- Enables pattern discovery and anomaly detection
- Supports filtering by pair and category

**Tests verify:**
- ✅ Single observation logging
- ✅ Retrieval of all recent observations
- ✅ Filtering by pair
- ✅ Filtering by category

**Key API:**
```python
log = ObservationLog()
log.log_observation(
    pair="EUR_USD",
    category="regime_change",
    description="Volatility spike",
    metadata={"volatility": 0.25}
)
recent = log.get_recent(pair="EUR_USD", category="regime_change", limit=20)
```

### 7. ScannerConfig
**File:** `src/scanner/config.py`

**What it does:**
- Master configuration dataclass for the FX Scanner
- Defines all gate thresholds, position sizing, agent toggles
- Supports multiple scan profiles (balanced, conservative, aggressive, smart)
- Provides pip values for all currency pairs

**Tests verify:**
- ✅ Import without errors
- ✅ Used by ConfigTuner for adjustments

## Key Findings

### Robustness
| Scenario | Result |
|----------|--------|
| Missing files | ✅ Graceful defaults |
| Corrupted JSON | ✅ Exception caught, defaults applied |
| Malformed data | ✅ Type-checked, safe fallbacks |
| Bounds violations | ✅ Clamped to min/max |
| Duplicate rules | ✅ Deduplication via hash |

### Integration Points
```
LearningEngine
    ↓ (records outcomes)
AccuracyGate
    ↓ (appends learnings)
learnings.md
    ↓ (check for promotions)
rules/trading.md
    ↓ (load and parse rules)
ConfigTuner
    ↓ (apply to config)
ScannerConfig
    ↓ (save state)
StateEngine
    ↓ (record session)
ImprovementTracker
    ↓ (log observations)
ObservationLog
```

## Test Data

All tests use mock data with realistic but synthetic values:

### Example Trade Entry
```python
trade = {
    "pair": "EUR_USD",
    "trade_id": "trade_001",
    "direction": "LONG",
    "confidence": 0.75,
    "sl_pips": 15,
    "tp_pips": 30,
    "atr_pips": 20,
    "outcome": {
        "trade_won": True,
        "pnl_pips": 25,
        "realized_pl": 250.0,
    },
    "agents": {
        "agent_reasons": [
            {"name": "trend", "score": 0.8},
            {"name": "uncertainty", "score": 0.2},
        ]
    },
    "weighted_vote_score": 0.75,
}
```

### Example Learning Entry
```python
entry = LearningEntry(
    date="2026-03-18",
    category="sl_tp",
    insight="sl_too_tight for EUR_USD: lost 25.0p > 2x ATR",
    action="Increase atr_sl_multiplier",
    source_trade_id="trade_001",
)
```

## Important Notes

### File I/O Safety
All tests use `tempfile.TemporaryDirectory()` — **zero production files are touched**.

### Return Type Warning
`AccuracyGate.check_pair()` returns a **tuple**, not a boolean:
```python
is_allowed, accuracy, reason = gate.check_pair("EUR_USD")
# NOT: if gate.check_pair("EUR_USD"): ...
```

### Bounds Are Hard-Coded
ConfigTuner bounds are defined in `BOUNDS` dict. Adding new tunable fields requires manual bounds entry.

## Extending the Tests

To add tests for a new module:

1. Create a new `TestNewModule(unittest.TestCase)` class
2. Implement `setUp()` (create temp dir, instantiate module)
3. Implement `tearDown()` (cleanup temp dir)
4. Add test methods: `test_feature_1()`, `test_feature_2()`, etc.
5. Add to `run_tests()` function
6. Run: `python testing/test_learning_loop_integration.py`

Example:
```python
class TestNewModule(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        from src.scanner.automation.new_module import NewModule
        self.module = NewModule(path=self.temp_path / "data.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_feature(self):
        result = self.module.do_something()
        self.assertEqual(result, expected)
        logger.info("✓ Feature works")
```

## Troubleshooting

### "No module named 'src'"
The test file adds PROJECT_ROOT to sys.path. Run from project root:
```bash
cd /sessions/magical-compassionate-einstein/mnt/ml_engine
python testing/test_learning_loop_integration.py
```

### "Failed to load accuracy data"
This is a DEBUG log — expected when starting fresh. Not an error.

### AttributeError in AccuracyGate
Ensure you're unpacking the tuple:
```python
# Wrong:
if gate.check_pair("EUR_USD"):  # Returns tuple!

# Right:
is_allowed, _, _ = gate.check_pair("EUR_USD")
if is_allowed: ...
```

## Performance

- **Test execution:** ~31ms for all 36 tests
- **Slowest module:** StateEngine (JSON I/O), still <5ms per test
- **Memory overhead:** Negligible (mock data only)

## Continuous Integration

Add to CI/CD pipeline:
```bash
python testing/test_learning_loop_integration.py
```

Exit code: 0 if all tests pass, 1 if any fail.
