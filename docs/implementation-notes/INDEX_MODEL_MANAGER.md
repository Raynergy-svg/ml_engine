# Model A/B Testing Framework — Complete Index

## Quick Navigation

**Just want to get started?** → Read [`GET_STARTED.md`](GET_STARTED.md) (5 minutes)

**Need quick reference?** → See [`docs/MODEL_TESTING_QUICKSTART.md`](docs/MODEL_TESTING_QUICKSTART.md)

**Want full details?** → Check [`docs/MODEL_TESTING_FRAMEWORK.md`](docs/MODEL_TESTING_FRAMEWORK.md)

**Implementation details?** → See [`docs/MODEL_MANAGER_IMPLEMENTATION.md`](docs/MODEL_MANAGER_IMPLEMENTATION.md)

---

## Files at a Glance

### Getting Started
- **[GET_STARTED.md](GET_STARTED.md)** — 5-minute quick start (READ THIS FIRST)
- **[MODEL_MANAGER_SUMMARY.md](MODEL_MANAGER_SUMMARY.md)** — Delivery summary

### Documentation
- **[docs/MODEL_TESTING_QUICKSTART.md](docs/MODEL_TESTING_QUICKSTART.md)** — 5-minute setup guide
- **[docs/MODEL_TESTING_FRAMEWORK.md](docs/MODEL_TESTING_FRAMEWORK.md)** — Complete technical reference
- **[docs/MODEL_MANAGER_IMPLEMENTATION.md](docs/MODEL_MANAGER_IMPLEMENTATION.md)** — Architecture & integration

### Implementation
- **[src/scanner/automation/model_manager.py](src/scanner/automation/model_manager.py)** — Core framework (564 lines)
- **[src/scanner/automation/continuous.py](src/scanner/automation/continuous.py)** — Integrated into scanner
- **[scripts/model_cli.py](scripts/model_cli.py)** — Command-line tool (262 lines)

### Testing
- **[tests/test_model_manager.py](tests/test_model_manager.py)** — Full test suite (440 lines)

---

## What This Framework Does

Enables safe model deployment through shadow testing:

1. **Register** your current production model
2. **Register** a new candidate model to test
3. **Start** the scanner — shadow testing runs automatically
4. **Monitor** progress with `python scripts/model_cli.py status`
5. **Automatic promotion** when candidate proves superior (50+ scans, 2% improvement)
6. **Instant rollback** if needed

---

## Commands Cheat Sheet

```bash
# First time: register baseline
python scripts/model_cli.py register v1.0.0 \
  trained_data/models/buddy_tf.keras

# Register candidate to test
python scripts/model_cli.py register v1.1.0 \
  trained_data/models/buddy_tf_candidate.keras \
  --candidate

# Start scanner (shadow testing runs automatically)
python buddy_scanner.py watch --interval 5

# Check progress
python scripts/model_cli.py status

# View all events
python scripts/model_cli.py history

# Manual promotion (overrides automation)
python scripts/model_cli.py promote v1.1.0

# Emergency rollback
python scripts/model_cli.py rollback v1.0.0
```

---

## Key Files

### Data Files (Auto-managed)
- `trained_data/models/buddy_tf.keras` — Current production model
- `trained_data/models/buddy_tf_candidate.keras` — Candidate under test
- `trained_data/models/version_history.jsonl` — Event log (append-only)
- `trained_data/models/archive/` — Backup directory

### Config
- `src/scanner/config.py` — Scanner configuration (existing file, not modified)

### Rules
- `.claude/rules/trading.md` — Promotes only with 50+ samples + 2% improvement
- `.claude/rules/improvement.md` — Code quality gates

---

## How It Works (Quick Overview)

```
Scanner Runs Every 5 Minutes
    ↓
Execute normal scan with production model
    ↓
_run_shadow_tests() records predictions from both models
    ↓
Every 50th scan:
    ↓
_check_model_promotion() evaluates:
  • Has candidate completed 50+ scans?
  • Is candidate 2% better than production?
    ↓
If YES to both:
  • Backup current production to archive/
  • Copy candidate to production path
  • Log promotion to version_history.jsonl
    ↓
All subsequent scans use new production model
```

---

## Integration Points

**Location:** `src/scanner/automation/continuous.py`, line ~189

After each scan completes, this code runs:

```python
try:
    self._run_shadow_tests(result)
    if self._scan_count % 50 == 0:
        self._check_model_promotion()
except Exception as shadow_err:
    logger.debug(f"Shadow test error: {shadow_err}")
```

**Non-breaking:** Wrapped in try/except, won't crash scanner if anything fails.

---

## Safety & Guarantees

✓ **Conservative:** Requires 50+ samples + 2% improvement before promotion
✓ **Reversible:** Production backup created, instant rollback available
✓ **Auditable:** Every decision logged to version_history.jsonl
✓ **Non-breaking:** Won't crash scanner, silent by default
✓ **Robust:** File locking (fcntl), corruption-resistant JSONL

---

## Next Steps

1. **Read** [`GET_STARTED.md`](GET_STARTED.md) (5 minutes)
2. **Register** baseline model: `python scripts/model_cli.py register v1.0.0 ...`
3. **Register** candidate: `python scripts/model_cli.py register v1.1.0 ... --candidate`
4. **Start** scanner: `python buddy_scanner.py watch --interval 5`
5. **Monitor** with: `python scripts/model_cli.py status`

---

## FAQ

**Q: How do I rollback to a previous model?**
```bash
python scripts/model_cli.py rollback v1.0.0
```

**Q: Where are the old models backed up?**
```bash
ls -la trained_data/models/archive/
```

**Q: How do I see what happened?**
```bash
python scripts/model_cli.py history --limit 50
```

**Q: Can I promote faster?**
Yes, change settings in `model_manager.py`:
```python
ModelManager(min_shadow_scans=20, min_improvement=0.01)
```

---

## Code Statistics

- **model_manager.py:** 564 lines (core framework)
- **model_cli.py:** 262 lines (CLI tool)
- **test_manager.py:** 440 lines (tests)
- **Documentation:** 1400+ lines
- **Total:** 2600+ lines
- **Dependencies:** None (stdlib only)

---

## Support

- **Quick start?** → [`GET_STARTED.md`](GET_STARTED.md)
- **Setup guide?** → [`docs/MODEL_TESTING_QUICKSTART.md`](docs/MODEL_TESTING_QUICKSTART.md)
- **Full reference?** → [`docs/MODEL_TESTING_FRAMEWORK.md`](docs/MODEL_TESTING_FRAMEWORK.md)
- **Architecture?** → [`docs/MODEL_MANAGER_IMPLEMENTATION.md`](docs/MODEL_MANAGER_IMPLEMENTATION.md)
- **CLI help?** → `python scripts/model_cli.py --help`

---

**Ready to go?** Start with [`GET_STARTED.md`](GET_STARTED.md) now!
