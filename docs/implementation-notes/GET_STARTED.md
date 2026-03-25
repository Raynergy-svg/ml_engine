# Get Started with Model A/B Testing — 5 Minutes

## What You Just Got

A complete model A/B testing framework. Instead of deploying new models blindly, you now:
1. Register a baseline production model
2. Register a new candidate model to test
3. Run the scanner — shadow testing happens automatically
4. Get automatic promotion when candidate proves better

## Step-by-Step

### Step 1: Register Your Current Model (First Time Only)

```bash
cd /sessions/magical-compassionate-einstein/mnt/ml_engine

python scripts/model_cli.py register v1.0.0 \
  trained_data/models/buddy_tf.keras \
  --accuracy 0.55
```

This marks your current production model as the baseline. The accuracy can be approximate.

### Step 2: When You Have a New Model to Test

```bash
python scripts/model_cli.py register v1.1.0 \
  trained_data/models/buddy_tf_candidate.keras \
  --accuracy 0.58 \
  --candidate
```

The `--candidate` flag marks it for shadow testing.

### Step 3: Start the Scanner

The shadow testing runs automatically when you start the scanner:

```bash
python buddy_scanner.py watch --interval 5
```

Or if using the Python API:

```python
from src.scanner.engine import Scanner
from src.scanner.automation.continuous import ContinuousScanner

scanner = Scanner()
continuous = ContinuousScanner(scanner)
continuous.run(interval_minutes=5)
```

The scanner will:
- Run normal scans every 5 minutes
- Automatically compare candidate vs production
- Log results for accuracy calculation
- Every 50 scans: check if promotion is ready

### Step 4: Monitor Progress

```bash
python scripts/model_cli.py status
```

You'll see something like:

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

Keep checking as the scans continue. When `Ready for Promotion: YES`, the next check will automatically promote the candidate.

### Step 5: Automatic Promotion

Once the candidate meets the criteria (50+ scans + 2% improvement), it's automatically promoted:

**Console output:**
```
✓ MODEL PROMOTED: v1.1.0
  Production accuracy: 0.5689
  Improvement: +0.0165
```

The old production model is automatically backed up to `trained_data/models/archive/`.

## Common Commands

### Check Status Anytime
```bash
python scripts/model_cli.py status
```

### View Full History
```bash
python scripts/model_cli.py history --limit 20
```

### Manual Promotion (Override)
```bash
python scripts/model_cli.py promote v1.1.0
```

### Emergency Rollback
```bash
python scripts/model_cli.py rollback v1.0.0
```

### Get Help
```bash
python scripts/model_cli.py --help
```

## File Locations

### Key Files
- **Production Model:** `trained_data/models/buddy_tf.keras`
- **Candidate Model:** `trained_data/models/buddy_tf_candidate.keras`
- **Event Log:** `trained_data/models/version_history.jsonl` (append-only, never modified)
- **Backups:** `trained_data/models/archive/`

### Documentation
- **Quick Start:** This file
- **Full Guide:** `docs/MODEL_TESTING_FRAMEWORK.md`
- **5-Minute Setup:** `docs/MODEL_TESTING_QUICKSTART.md`
- **Architecture:** `docs/MODEL_MANAGER_IMPLEMENTATION.md`

## How It Works (Behind the Scenes)

```
1. After each scan, both models' predictions are recorded
2. When the actual price movement is known, accuracy is calculated
3. Every 50 scans, the system checks:
   - Has candidate been tested 50+ times?
   - Is candidate at least 2% better than production?
4. If YES to both: candidate becomes new production
5. Old production is backed up in case you need to rollback
```

## Key Settings (Defaults Are Good to Start)

If you want to customize:

### Promote Faster (Less Cautious)
```python
# In src/scanner/automation/model_manager.py
ModelManager(min_shadow_scans=20, min_improvement=0.01)
```

### Promote Slower (More Confident)
```python
ModelManager(min_shadow_scans=100, min_improvement=0.05)
```

### Check Promotion More Often
```python
# In src/scanner/automation/continuous.py, line ~195
if self._scan_count % 25 == 0:  # Check every 25 scans instead of 50
    self._check_model_promotion()
```

## What If Something Goes Wrong?

### Q: How do I rollback to the old model?
```bash
python scripts/model_cli.py rollback v1.0.0
```

### Q: Where are the old models backed up?
```bash
ls -la trained_data/models/archive/
```

### Q: How do I see what happened?
```bash
python scripts/model_cli.py history --limit 50
```

### Q: Shadow tests aren't running
1. Check that ContinuousScanner is actually being called
2. Look for "[model" in the scanner logs
3. Verify the candidate is registered: `python scripts/model_cli.py status`

### Q: Candidate never gets promoted
1. Check improvement: `python scripts/model_cli.py status`
2. If improvement < 2%, candidate just isn't better
3. You can lower the threshold in ModelManager if you want to promote faster

## What's Next

The framework supports everything you need:

**Out of the box:**
- ✓ Version registration
- ✓ Automatic shadow testing
- ✓ Automatic promotion
- ✓ Safe rollback
- ✓ Complete audit trail

**Future enhancements:**
- Per-pair accuracy metrics
- Multiple candidates in parallel
- Latency profiling
- Automated acceptance tests
- Canary deployments

## Key Principles

1. **Safe:** Production model never changes without 50+ test samples
2. **Automatic:** No manual intervention needed once candidate is registered
3. **Reversible:** Instant rollback if needed
4. **Auditable:** Every decision is logged to version_history.jsonl
5. **Simple:** No complex configuration, works with defaults

## Quick Troubleshooting Checklist

- [ ] Did you run `register v1.0.0` first?
- [ ] Did you run `register v1.1.0 --candidate` second?
- [ ] Is the scanner still running? (Should say "Next scan in X minutes")
- [ ] Check `python scripts/model_cli.py status` — what's the progress %?
- [ ] View history: `python scripts/model_cli.py history`
- [ ] If stuck, rollback: `python scripts/model_cli.py rollback v1.0.0`

## One More Thing

The framework follows all the trading rules from `.claude/rules/`:

✓ Requires multiple confirming signals (50+ scans)
✓ Has clear promotion criteria (2% improvement)
✓ Backs up before overwrite
✓ Has instant rollback capability
✓ Logs everything for analysis

You're good to go. Start with the 5 steps above, then check the full documentation if you want to customize anything.

**Questions?** Check `docs/MODEL_TESTING_FRAMEWORK.md` or run `python scripts/model_cli.py --help`.
