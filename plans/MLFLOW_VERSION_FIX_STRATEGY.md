# MLflow Version Fix Strategy

**Date**: 2026-02-20
**Status**: Proposed
**Priority**: High - Blocking training pipeline execution

---

## Executive Summary

An MLflow version mismatch between the installed version (2.10.0) and requirements.txt (3.8.1) is causing Alembic migration errors that block the training pipeline. This document analyzes fix options and recommends **upgrading MLflow to 3.8.1** as the optimal solution.

---

## Problem Analysis

### Current State

| Component | Value | Notes |
|-----------|-------|-------|
| Installed MLflow | 2.10.0 | In `intel` conda environment |
| requirements.txt | 3.8.1 | Specified but not installed |
| DB Revision | `1bd49d398cd23` | MLflow 3.x format |
| Valid in 2.10.0 | `451aebb31d03` | Head revision for 2.10.0 |

### Affected Databases

| Database | Size | Last Modified | Contents |
|----------|------|---------------|----------|
| `trained_data/mlruns/mlflow.db` | 1.15 MB | Feb 18 22:21 | Recent experiments |
| `mlruns/mlflow.db` | 528 KB | Jan 30 20:34 | 14 experiment directories |

### Root Cause

The databases were created/modified by MLflow 3.x (creating revision `1bd49d398cd23`), but the current environment has MLflow 2.10.0 installed, which only recognizes revision `451aebb31d03`. This causes:

```
alembic.util.exc.CommandError: Can't locate revision identified by '1bd49d398cd23'
```

---

## Dependency Analysis

### MLflow 3.8.1 Compatibility

| Package | Current Version | MLflow 3.8.1 Req | Conflict? |
|---------|-----------------|------------------|-----------|
| TensorFlow | 2.16.x - 2.18.x | No direct dep | ✅ No |
| SQLAlchemy | 2.0.45 | >= 1.4 | ✅ No |
| protobuf | 4.23.0 - 5.0.0 | No direct dep | ✅ No |
| numpy | 1.26.4 | Compatible | ✅ No |
| pandas | 2.3.3 | Compatible | ✅ No |
| OpenTelemetry | 1.27.0 | Included in reqs | ✅ No |

**Conclusion**: No dependency conflicts detected. MLflow 3.8.1 should install cleanly.

### M1 Metal Compatibility

- MLflow has no platform-specific code for M1
- TensorFlow Metal is independent of MLflow
- No additional considerations needed

---

## Data Preservation Assessment

### MLflow Data Redundancy with W&B

The codebase implements **dual logging** to both MLflow and W&B:

| Feature | MLflow | W&B | Redundant? |
|---------|--------|-----|------------|
| Loss metrics | ✅ | ✅ | Yes |
| Hyperparameters | ✅ | ✅ | Yes |
| Model checkpoints | ✅ | ✅ | Yes |
| System metrics | ✅ | ✅ | Yes |
| Gradient histograms | ❌ | ✅ | No - W&B only |
| Model graph | ❌ | ✅ | No - W&B only |

**Key Files**:
- [`src/training/enterprise_training.py`](src/training/enterprise_training.py): Lines 60-63, 379-381, 511-517
- [`src/training/wandb_keras_callback.py`](src/training/wandb_keras_callback.py): Full W&B integration
- [`src/training/wandb_observatory.py`](src/training/wandb_observatory.py): Enterprise-grade W&B tracking

**Conclusion**: All critical experiment data is preserved in W&B. MLflow data loss is acceptable if necessary.

---

## Fix Options Evaluation

### Option A: Upgrade MLflow to 3.8.1 ⭐ RECOMMENDED

```bash
pip install mlflow==3.8.1
```

| Aspect | Rating | Notes |
|--------|--------|-------|
| Data Preservation | ✅ Excellent | All existing data retained |
| Risk | 🟢 Low | No dependency conflicts |
| Effort | 🟢 Minimal | Single command |
| Future-proof | ✅ Yes | Matches requirements.txt |

**Pros**:
- Preserves all existing experiment data
- Aligns with requirements.txt specification
- Database schema will be compatible
- No code changes required

**Cons**:
- Requires pip install in active environment
- May need to re-run if environment is recreated

---

### Option B: Reset Databases, Keep MLflow 2.10.0

```bash
rm trained_data/mlruns/mlflow.db mlruns/mlflow.db
```

| Aspect | Rating | Notes |
|--------|--------|-------|
| Data Preservation | ❌ None | All MLflow history lost |
| Risk | 🟡 Medium | Version mismatch persists |
| Effort | 🟢 Minimal | Single command |
| Future-proof | ❌ No | Still mismatched with requirements.txt |

**Pros**:
- Immediate fix with no version changes
- Clean slate for new experiments

**Cons**:
- Loses all MLflow experiment history
- Version mismatch with requirements.txt persists
- Will recur if `pip install -r requirements.txt` is run

---

### Option C: Manual Alembic Revision Downgrade

```sql
-- Update alembic_version table
UPDATE alembic_version SET version_num = '451aebb31d03';
```

| Aspect | Rating | Notes |
|--------|--------|-------|
| Data Preservation | 🟡 Partial | Metadata kept, schema may mismatch |
| Risk | 🔴 High | Schema incompatibility likely |
| Effort | 🔴 High | Manual SQL + verification |
| Future-proof | ❌ No | Hack, not proper solution |

**Pros**:
- Preserves some data
- No version change needed

**Cons**:
- High risk of schema incompatibility
- Complex and error-prone
- Not a supported migration path
- May cause silent data corruption

---

## Recommendation

### Selected Option: A - Upgrade MLflow to 3.8.1

**Rationale**:
1. **No data loss**: All existing experiments preserved
2. **No dependency conflicts**: Analysis shows clean install
3. **Aligns with requirements.txt**: Consistent with project specification
4. **W&B backup exists**: Even if upgrade fails, data exists in W&B
5. **Minimal effort**: Single pip command

---

## Implementation Plan

### Phase 1: Pre-Upgrade Backup

```bash
# Step 1: Backup existing databases
cp trained_data/mlruns/mlflow.db trained_data/mlruns/mlflow.db.backup
cp mlruns/mlflow.db mlruns/mlflow.db.backup

# Step 2: Verify W&B has recent run data
# Check that run lmuw1b78 exists in W&B dashboard
```

### Phase 2: Upgrade MLflow

```bash
# Step 3: Upgrade MLflow
pip install mlflow==3.8.1

# Step 4: Verify installation
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
# Expected output: MLflow version: 3.8.1
```

### Phase 3: Verification

```bash
# Step 5: Test MLflow connection
python scripts/diag_mlflow_alembic.py

# Step 6: Verify database access
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///trained_data/mlruns/mlflow.db')
experiments = mlflow.search_experiments()
print(f'Found {len(list(experiments))} experiments')
"
```

### Phase 4: Cleanup (After Successful Verification)

```bash
# Step 7: Remove backups after confirming success
rm trained_data/mlruns/mlflow.db.backup
rm mlruns/mlflow.db.backup
```

---

## Rollback Strategy

If upgrade fails or causes issues:

```bash
# Step 1: Revert to previous MLflow version
pip install mlflow==2.10.0

# Step 2: Restore database backups
cp trained_data/mlruns/mlflow.db.backup trained_data/mlruns/mlflow.db
cp mlruns/mlflow.db.backup mlruns/mlflow.db

# Step 3: Fall back to Option B (reset databases)
rm trained_data/mlruns/mlflow.db mlruns/mlflow.db
```

---

## Preventive Measures

### 1. Environment Consistency

Add to environment setup scripts:

```bash
# In scripts/install_dep.sh or similar
pip install mlflow==3.8.1  # Pin to specific version
```

### 2. requirements.txt Verification

Add CI check to verify installed versions match requirements.txt:

```python
# Add to scripts/validate_fixes.py
import pkg_resources
import yaml

def check_mlflow_version():
    installed = pkg_resources.get_distribution('mlflow').version
    with open('requirements.txt') as f:
        for line in f:
            if line.startswith('mlflow=='):
                required = line.split('==')[1].strip()
                assert installed == required, f"MLflow version mismatch: {installed} vs {required}"
```

### 3. Database Backup Schedule

Add periodic backup of MLflow databases:

```bash
# Add to crontab or scheduled task
0 0 * * * cp /path/to/mlruns/mlflow.db /path/to/backups/mlflow-$(date +\%Y\%m\%d).db
```

### 4. Documentation Update

Update [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) to include MLflow version verification step.

---

## Decision Matrix

| Criteria | Option A | Option B | Option C |
|----------|----------|----------|----------|
| Data Preservation | ✅ Full | ❌ None | 🟡 Partial |
| Implementation Effort | 🟢 Low | 🟢 Low | 🔴 High |
| Risk Level | 🟢 Low | 🟡 Medium | 🔴 High |
| Future Compatibility | ✅ Yes | ❌ No | ❌ No |
| Recommended | ⭐ YES | ❌ NO | ❌ NO |

---

## Appendix: Technical Details

### MLflow Version History

- **2.10.0**: Released ~Feb 2024, uses alembic revision `451aebb31d03`
- **3.8.1**: Latest stable as of requirements.txt update, uses revision `1bd49d398cd23`

### Database Schema Changes

The revision `1bd49d398cd23` indicates schema changes between 2.x and 3.x series. MLflow 3.x databases are not backward compatible with 2.x.

### Related Files

- [`scripts/diag_mlflow_alembic.py`](scripts/diag_mlflow_alembic.py): Diagnostic script
- [`src/training/enterprise_training.py`](src/training/enterprise_training.py): MLflow integration
- [`src/utils/model_registry.py`](src/utils/model_registry.py): MLflow model registry
- [`requirements.txt`](requirements.txt): Line 62 - `mlflow==3.8.1`

---

## Approval

| Role | Name | Date | Decision |
|------|------|------|----------|
| Architect | Kilo Code | 2026-02-20 | Recommended: Option A |
| Reviewer | _pending_ | _pending_ | _pending_ |

---

**Next Steps**: Switch to Code mode to execute the implementation plan.
