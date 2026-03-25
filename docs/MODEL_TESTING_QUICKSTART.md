# Model A/B Testing — Quick Start

## 5-Minute Setup

### Step 1: Register Production Model (Baseline)

If you don't have a production baseline yet:

```bash
python scripts/model_cli.py register v1.0.0 \
  trained_data/models/buddy_tf.keras \
  --accuracy 0.55
```

This marks the current model as the baseline for shadow testing.

### Step 2: Register Candidate Model

When you have a new model to test:

```bash
python scripts/model_cli.py register v1.1.0 \
  trained_data/models/buddy_tf_candidate.keras \
  --accuracy 0.58 \
  --candidate
```

The `--candidate` flag marks it for shadow testing against production.

### Step 3: Start Continuous Scanner

The shadow testing runs automatically:

```bash
python buddy_scanner.py watch --interval 5
```

Or via ContinuousScanner in code:

```python
from src.scanner.engine import Scanner
from src.scanner.automation.continuous import ContinuousScanner

scanner = Scanner()
continuous = ContinuousScanner(scanner)
continuous.run(interval_minutes=5, auto_execute=False)
```

The scanner will:
1. Run scans every 5 minutes (default)
2. Automatically compare candidate vs production
3. Log shadow test results
4. Every 50 scans: check if promotion is ready

### Step 4: Monitor Progress

```bash
python scripts/model_cli.py status
```

**Output example:**
```
MODEL A/B TEST STATUS
======================================================================

Total Versions Registered: 2

[PRODUCTION]
  Version: v1.0.0
  Validation Accuracy: 0.5500
  Shadow Test Accuracy: 0.5524
  Shadow Tests: 25
  Registered: 2026-03-18T10:00:00

[CANDIDATE]
  Version: v1.1.0
  Validation Accuracy: 0.5800
  Shadow Test Accuracy: 0.5689
  Shadow Tests: 25
  Registered: 2026-03-18T10:05:00

[SHADOW TEST PROGRESS]
  Scans Completed: 25/50
  Progress: 50.0%
  Improvement vs Incumbent: +0.0165
  Improvement Threshold: 0.0200
  Ready for Promotion: NO
```

### Step 5: Automatic Promotion

When the candidate exceeds the improvement threshold, it's automatically promoted:

```
✓ MODEL PROMOTED: v1.1.0
  Production accuracy: 0.5689
  Improvement: +0.0165
```

The old production is backed up automatically.

## Common Tasks

### Check If Promotion Happened

```bash
python scripts/model_cli.py history --limit 5
```

Look for lines with `[PROMOTION]`.

### Rollback to Previous Model

If the new production model degrades:

```bash
python scripts/model_cli.py rollback v1.0.0
```

### View Full History

```bash
python scripts/model_cli.py history --limit 100
```

### Check Model Accuracy During Training

Before registering:
```python
from src.scanner.automation.model_manager import ModelManager

# After training your model
mm = ModelManager()
mm.register_version(
    version_tag="v1.2.0",
    model_path="/path/to/new_model.keras",
    accuracy=0.87,  # Your validation accuracy
    is_candidate=True
)
```

## Key Settings

### Sensitivity to Improvement

Default: 2% minimum improvement required

**To require less improvement (promote faster):**
```python
ModelManager(min_improvement=0.01)  # 1%
```

**To require more improvement (promote slower):**
```python
ModelManager(min_improvement=0.05)  # 5%
```

### Number of Test Scans

Default: 50 scans before promotion decision

**To decide faster:**
```python
ModelManager(min_shadow_scans=25)
```

**To be more confident:**
```python
ModelManager(min_shadow_scans=100)
```

## Data Flow

```
Scanner Runs Every 5 Minutes
    ↓
Executes normal scan
    ↓
_run_shadow_tests() called
    ↓
Records predictions from both models
    ↓
Every 50th scan:
    ↓
_check_model_promotion() called
    ↓
If improvement > threshold:
    → Copy candidate to production/backup old version
    → Log promotion to history
    ↓
Display status to console
    ↓
Loop continues...
```

## Troubleshooting

### Q: Shadow tests aren't running
**A:** Check that ContinuousScanner is calling `_run_shadow_tests()` and `_check_model_promotion()`. Both are in the scan loop (line ~189 in continuous.py).

### Q: Candidate never gets promoted
**Options:**
1. Candidate isn't actually better (increase training iterations)
2. Threshold is too high (reduce `min_improvement`)
3. Not enough scans (reduce `min_shadow_scans`)
4. Model loading issue (check version_history.jsonl for errors)

### Q: Where are the model files?
```
trained_data/models/
├── buddy_tf.keras                 ← Current production
├── buddy_tf_candidate.keras       ← Under test
├── version_history.jsonl          ← All events logged here
└── archive/
    └── buddy_tf_backup_v1.0.0_*   ← Previous versions
```

### Q: Can I test multiple candidates at once?
**Current:** Only one candidate at a time (one in production, one under test).

**Future enhancement:** Modify `_get_candidate()` to support multiple candidates.

## API Examples

### In Python Code

```python
from src.scanner.automation.model_manager import ModelManager

mm = ModelManager()

# Register a version
mm.register_version("v1.2.0", "path/to/model.keras", accuracy=0.87, is_candidate=True)

# Check current status
status = mm.get_status()
print(f"Production: {status['production']['version_tag']}")
print(f"Candidate progress: {status['shadow_test_progress']['progress_pct']}%")

# Manually promote (normally automatic)
if mm.check_promotion():
    mm.promote(mm._get_candidate().version_tag)

# View versions
for tag, version in mm._versions.items():
    print(f"{tag}: {version.accuracy:.4f}")
```

### Command Line

```bash
# Register
python scripts/model_cli.py register v1.3.0 path/to/model.keras --candidate

# Status
python scripts/model_cli.py status

# Promote (manual override, normally automatic)
python scripts/model_cli.py promote v1.3.0

# Rollback
python scripts/model_cli.py rollback v1.2.0

# History
python scripts/model_cli.py history
```

## Next: Advanced Features

- **Per-pair accuracy:** Track which pairs the model excels at
- **Latency profiling:** Measure inference speed
- **Canary deployment:** Route 10% of traffic to candidate, 90% to production
- **Multi-candidate tournament:** Test 3+ models in parallel
- **Integration with test suite:** Automated acceptance tests before promotion

See `docs/MODEL_TESTING_FRAMEWORK.md` for full documentation.
