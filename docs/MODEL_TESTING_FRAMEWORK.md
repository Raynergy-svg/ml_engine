# Model A/B Testing and Version Control Framework

## Overview

The Model Manager provides safe, data-driven model promotion through shadow testing. Instead of deploying new models directly to production, the system runs them in parallel with the incumbent model, collecting accuracy metrics on live scan results, and automatically promoting when the candidate proves superior.

**Key Features:**
- Version registration with metadata tracking
- Shadow testing (compare candidate vs production on same data)
- Automatic promotion when candidate meets improvement threshold
- Safe rollback to previous versions
- Corruption-resistant JSONL logging with file locking
- CLI for manual model management

## Architecture

```
Scanner Scan Results
       ↓
   [Shadow Test]
   Incumbent vs Candidate
       ↓
   Record Predictions
   to version_history.jsonl
       ↓
   Every 50 scans:
   Check Promotion Criteria
       ↓
   If Improvement > Threshold:
   [PROMOTE] → Copy to production
   Backup old production
```

## Core Concepts

### ModelVersion
Represents a registered model with tracking metadata:
- `version_tag`: Unique identifier (e.g., "v1.0.0", "v1.1.0-candidate")
- `model_path`: Path to .keras or model directory
- `accuracy`: Validation accuracy from training (informational)
- `shadow_scans`: Number of live comparisons completed
- `shadow_correct`: Correct predictions during shadow testing
- `shadow_accuracy`: Actual accuracy on live data
- `is_production`: Currently in use
- `is_candidate`: Being shadow-tested against production

### ShadowTestResult
Single comparison point between models on one pair:
- Records both models' predictions
- Logs actual outcome after N candles
- Tracks which model was correct

## Workflows

### 1. Register a New Candidate Model

```bash
python scripts/model_cli.py register v1.1.0 \
  trained_data/models/buddy_tf_candidate.keras \
  --accuracy 0.85 \
  --candidate
```

**What happens:**
1. ModelManager validates that the model file exists
2. Creates a ModelVersion record with metadata
3. Marks as candidate (is_candidate=True)
4. Appends registration to version_history.jsonl

**Result:** Candidate is ready for shadow testing.

### 2. Shadow Testing Runs Automatically

When `ContinuousScanner` runs, it:
1. Executes a normal scan with production model
2. Calls `_run_shadow_tests(result)` after each scan
3. Records predictions from both models
4. Every 50 scans, calls `_check_model_promotion()`

**What's tracked:**
- Each pair prediction from both models
- Actual price movement (outcome)
- Which model was correct
- Accuracy trend over time

**Progress visualization:**
```
  [dim]Model A/B: v1.1.0 42% complete (21/50 scans) improvement: +0.0234[/dim]
```

### 3. Automatic Promotion

When candidate meets criteria (every 50 scans):
```
If shadow_scans >= min_shadow_scans (50)
  AND shadow_accuracy - incumbent_accuracy > min_improvement (0.02)
    → Promote candidate to production
```

**Promotion process:**
1. Backup current production to `trained_data/models/archive/`
2. Copy candidate to `trained_data/models/buddy_tf.keras`
3. Update version records (is_production=True, is_candidate=False)
4. Log promotion to version_history.jsonl

**Output:**
```
✓ MODEL PROMOTED: v1.1.0
  Production accuracy: 0.6234
  Improvement: +0.0234
```

### 4. Monitoring Status

```bash
python scripts/model_cli.py status
```

**Output shows:**
- Current production model and accuracy
- Candidate model (if active) and progress
- Shadow test completion percentage
- Improvement vs incumbent
- Promotion readiness

### 5. Manual Rollback

If production model degrades:
```bash
python scripts/model_cli.py rollback v1.0.0
```

**What happens:**
1. Backup current production
2. Copy v1.0.0 from history
3. Update version records
4. Log rollback to history

## Configuration

### ModelManager Parameters

```python
ModelManager(
    models_dir="trained_data/models",           # Where models live
    version_history_path="trained_data/models/version_history.jsonl",
    min_shadow_scans=50,                        # Min scans before promotion decision
    min_improvement=0.02,                       # Min accuracy improvement (2%)
)
```

**Tuning:**
- Increase `min_shadow_scans` (e.g., 100) for more confidence before promotion
- Increase `min_improvement` (e.g., 0.05) to require 5% improvement
- Decrease to promote faster but with less certainty

### ContinuousScanner Integration

Shadow testing runs automatically every scan:
```python
try:
    self._run_shadow_tests(result)
    # Every 50th scan: check for promotion
    if self._scan_count % 50 == 0:
        self._check_model_promotion()
except Exception as shadow_err:
    logger.debug(f"Shadow test error: {shadow_err}")
```

**No configuration needed** — safe defaults apply.

## File Locations

### Core Files
- `src/scanner/automation/model_manager.py` — ModelManager class
- `src/scanner/automation/continuous.py` — Integration into scan loop
- `scripts/model_cli.py` — CLI for manual management

### Data Files
- `trained_data/models/buddy_tf.keras` — Current production model
- `trained_data/models/buddy_tf_candidate.keras` — Candidate under test
- `trained_data/models/version_history.jsonl` — All version events (append-only)
- `trained_data/models/archive/` — Backup copies of previous versions

## API Reference

### ModelManager

```python
from src.scanner.automation.model_manager import ModelManager

mm = ModelManager()

# Register a version
version = mm.register_version(
    version_tag="v1.1.0",
    model_path="trained_data/models/buddy_tf_candidate.keras",
    accuracy=0.85,
    is_candidate=True
)

# Record a shadow test result
mm.record_shadow_result(
    pair="EUR_USD",
    incumbent_direction="LONG",
    candidate_direction="SHORT",
    actual_movement=0.8  # 0=down, 1=up
)

# Check if promotion is warranted
promotion_version = mm.check_promotion()  # Returns version_tag or None

# Promote a candidate
if promotion_version:
    mm.promote(promotion_version)

# Get status summary
status = mm.get_status()
print(f"Production: {status['production']['version_tag']}")
print(f"Candidate: {status['candidate']['version_tag']}")
print(f"Progress: {status['shadow_test_progress']['progress_pct']}%")

# Rollback to previous version
mm.rollback("v1.0.0")
```

## Safety Guarantees

### Corruption Resistance
- JSONL logging with file locking (fcntl) prevents corruption
- Each record is self-contained (not dependent on previous records)
- Missing/corrupted lines are skipped with warnings
- In-memory state is the source of truth for decisions

### Non-Breaking Integration
- Shadow testing is silent by default (logged at DEBUG level)
- Does not affect existing scan logic
- Wraps in try/except — no exception escapes to scanner
- Can be disabled by not calling `_run_shadow_tests()`

### Conservative Promotion
- Requires 50+ successful comparisons (default)
- Requires 2% accuracy improvement (default)
- Incumbent must have baseline accuracy tracked
- Old production is always backed up before overwrite

## Monitoring and Alerts

### What to Watch

**Version History Growth:**
```bash
wc -l trained_data/models/version_history.jsonl
# Archive if exceeds 100k lines (consolidate to summary)
```

**Production Model Stability:**
```python
status = mm.get_status()
if status['production']['shadow_accuracy'] < 0.50:
    print("WARNING: Production accuracy below threshold")
```

**Candidate Drift Detection:**
```python
# If candidate hasn't improved after 200+ scans, retire it
if candidate.shadow_scans > 200 and improvement < 0.01:
    print("Candidate not improving; recommend rollback or reset")
```

## Troubleshooting

### "No production model found"
**Problem:** First deployment — no incumbent to compare against.
**Solution:** Register v1.0.0 as production without candidate:
```bash
python scripts/model_cli.py register v1.0.0 \
  trained_data/models/buddy_tf.keras \
  --accuracy 0.55
```

### Candidate stuck at same accuracy
**Problem:** Two models are equally good, improvement is borderline.
**Solution:**
1. Increase scans: `min_shadow_scans = 100`
2. Check if candidate is actually different (reload from disk)
3. Retire candidate and try a new one

### "Model path does not exist"
**Problem:** Path to model file is incorrect.
**Solution:** Use absolute paths:
```bash
python scripts/model_cli.py register v1.1.0 \
  /absolute/path/to/trained_data/models/buddy_tf_candidate.keras \
  --candidate
```

### Promotion happened but accuracy didn't improve
**Problem:** Shadow test results were misleading (noisy sample).
**Solution:** Rollback and increase `min_improvement` threshold.

## Learning Loop Integration

Shadow testing results feed into the learning system:

```python
# In learning loop (continuous.py)
if promoted:
    le = LearningEngine()
    le.log_learning({
        "date": datetime.now().isoformat(),
        "event": "model_promotion",
        "from_version": incumbent.version_tag,
        "to_version": promoted_version,
        "improvement": improvement,
        "scans": candidate.shadow_scans,
    })
```

This allows the system to:
- Track which model versions worked best
- Correlate promotions with performance changes
- Build historical record of model evolution

## Next Steps

### Extend Shadow Testing
Current MVP logs predictions only. Future enhancements:

1. **Actual Outcome Tracking:** After N candles, record true price movement
2. **Per-Pair Metrics:** Track accuracy by pair (e.g., EUR_USD: 62%, GBP_USD: 58%)
3. **Latency Comparison:** Measure prediction speed (inference time)
4. **Feature Importance:** Log which features drove each prediction
5. **Automated Testing:** Integration with test suite (pytest)

### Advanced Workflows
- Multi-candidate tournament (test 3+ models in parallel)
- Canary deployment (50% traffic on candidate, 50% on incumbent)
- Blue-green switchover (instant cutover with quick rollback)
- Performance profiling (FLOPs, memory, latency per model)

## References

- `trained_data/models/joint/joint_training_meta.json` — Model training metadata
- `trained_data/trade_journal_rl.json` — Trade outcomes (can be used to evaluate models in hindsight)
- `.claude/learnings.md` — System learnings (model decisions, outcomes)
- `.claude/rules/trading.md` — Execution rules (promotion gates)
