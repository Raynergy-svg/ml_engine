# Walk-Forward Cross-Validation Implementation Guide

## Overview

This document describes the walk-forward cross-validation (WF-CV) system implemented for the ML Engine trading bot. WF-CV provides robust time-series validation that prevents look-ahead bias and gives realistic out-of-sample performance estimates.

## The Problem

Traditional model training can suffer from several issues:

1. **Look-ahead bias**: Models trained on full dataset can "see" future information
2. **Overfitting to recent data**: Models may not generalize to changing market conditions
3. **Optimistic performance estimates**: Standard train/val split doesn't account for temporal dependencies
4. **Concept drift**: Markets change over time, but models are frozen

## The Solution

Walk-forward cross-validation addresses these issues by:

1. **Temporal ordering**: Train/val/test splits maintain chronological order
2. **Sliding windows**: Training window moves forward in time (rolling mode)
3. **Per-fold retraining**: Model is retrained for each time period
4. **Purged gaps**: Samples near split boundaries are removed to prevent leakage

## Architecture

### Components

```
config/config_improved_H1.yaml
    └── walkforward: {...}                              # Configuration

src/training/walkforward_validation.py
    ├── WalkForwardConfig                               # Configuration dataclass
    ├── WalkForwardValidator                            # Splitting logic
    ├── purged_kfold_split()                           # Purged CV
    └── train_direction_with_walkforward()             # Training with WF-CV

src/training/buddy_training_helpers.py
    └── train_with_walkforward_validation()            # Main wrapper function

tests/test_walkforward_config.py
    └── Comprehensive test suite                       # Validation tests
```

### Walk-Forward Modes

#### Rolling (Sliding) Window Mode (Recommended)

```
Data: |========================================| (15,000 candles)

Fold 1: |---TRAIN---|gap|VAL|TEST|
Fold 2:      |---TRAIN---|gap|VAL|TEST|
Fold 3:           |---TRAIN---|gap|VAL|TEST|
Fold 4:                |---TRAIN---|gap|VAL|TEST|
Fold 5:                     |---TRAIN---|gap|VAL|TEST|
```

- Training window size stays constant (60% of data)
- Window slides forward in time
- **Benefit**: Focuses on recent market behavior
- **Best for**: FX trading where recent data is most relevant

#### Expanding Window Mode

```
Data: |========================================| (15,000 candles)

Fold 1: |-TRAIN-|gap|VAL|TEST|
Fold 2: |-----TRAIN-----|gap|VAL|TEST|
Fold 3: |---------TRAIN---------|gap|VAL|TEST|
Fold 4: |-------------TRAIN-------------|gap|VAL|TEST|
```

- Training window grows with each fold
- Uses all historical data
- **Benefit**: More training data in later folds
- **Best for**: When historical patterns are stable

## Configuration

### YAML Configuration (config/config_improved_H1.yaml)

```yaml
# ----- WALK-FORWARD CROSS-VALIDATION -----
walkforward:
  enabled: true                    # Enable walk-forward validation
  mode: "rolling"                  # "rolling" (sliding window) or "expanding"
  n_splits: 5                      # Number of folds
  train_size: 0.60                 # Training window size (60% of data)
  val_size: 0.10                   # Validation size (10%)
  test_size: 0.10                  # Test/holdout size (10%)
  gap: 24                          # Gap between train/val (24 H1 bars = 1 day)
  min_train_size: 2000             # Minimum training samples per fold
  
  # Purged K-Fold settings (advanced)
  use_purged_kfold: true           # Use purged cross-validation
  purge_gap: 24                    # Purge samples near test (24 H1 bars)
  embargo_gap: 12                  # Embargo after train (12 H1 bars)
  
  # Model retraining per fold
  retrain_per_fold: true           # Retrain model for each fold (recommended)
  aggregate_method: "best"         # "best", "average", or "ensemble"
  
  # Regime-aware validation (experimental)
  ensure_regime_balance: false     # Ensure each fold has all regimes
  min_samples_per_regime: 50       # Min samples per regime in fold
```

### Parameter Guide

| Parameter | Description | Recommended Value (H1) |
|-----------|-------------|------------------------|
| `enabled` | Enable walk-forward validation | `true` |
| `mode` | Window type | `"rolling"` (sliding window) |
| `n_splits` | Number of folds | `5` (trade-off between speed and robustness) |
| `train_size` | Training window fraction | `0.60` (60% of data) |
| `val_size` | Validation fraction | `0.10` (10% of data) |
| `test_size` | Test/holdout fraction | `0.10` (10% of data) |
| `gap` | Samples between train/val | `24` (1 day for H1 bars) |
| `min_train_size` | Minimum train samples | `2000` (safety check) |
| `purge_gap` | Samples to purge near test | `24` (1 day) |
| `embargo_gap` | Embargo after training | `12` (12 hours) |
| `retrain_per_fold` | Retrain for each fold | `true` (recommended) |
| `aggregate_method` | How to combine folds | `"best"` (select best fold) |

## Usage

### Method 1: Via Configuration (Automatic)

Enable walk-forward in your config file:

```yaml
walkforward:
  enabled: true
  mode: "rolling"
  n_splits: 5
  retrain_per_fold: true
```

Then use the wrapper function:

```python
from src.training.buddy_training_helpers import train_with_walkforward_validation
from src.training.modular_trainers import TransformerDirectionTrainer, TrainerConfig

# Load config
import yaml
with open('config/config_improved_H1.yaml') as f:
    config = yaml.safe_load(f)

# Train with walk-forward validation
trainer, metrics = train_with_walkforward_validation(
    trainer_class=TransformerDirectionTrainer,
    trainer_config=TrainerConfig(),
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    feature_names=feature_names,
    instrument="EUR_USD",
    wf_config=config.get('walkforward'),  # Pass walk-forward config
    console=console,
)
```

### Method 2: Programmatic Configuration

```python
from src.training.walkforward_validation import WalkForwardConfig, WalkForwardValidator
from src.training.modular_trainers import TransformerDirectionTrainer, TrainerConfig

# Create walk-forward config
wf_config = WalkForwardConfig(
    enabled=True,
    mode="rolling",
    n_splits=5,
    train_size=0.60,
    gap=24,
    retrain_per_fold=True,
    aggregate_method="best",
)

# Create validator
validator = WalkForwardValidator(
    n_splits=wf_config.n_splits,
    train_size=wf_config.train_size,
    val_size=wf_config.val_size,
    gap=wf_config.gap,
    mode=wf_config.mode,
)

# Combine train and val data
import numpy as np
X_combined = np.concatenate([X_train, X_val], axis=0)
y_combined = np.concatenate([y_train, y_val], axis=0)

# Train on each fold
fold_results = []
for fold_idx, (train_idx, val_idx, test_idx) in enumerate(validator.split(X_combined)):
    print(f"Fold {fold_idx + 1}: Train={len(train_idx)}, Val={len(val_idx)}")
    
    # Create fresh trainer for this fold
    trainer = TransformerDirectionTrainer(TrainerConfig())
    
    # Train on this fold
    metrics = trainer.train(
        X_combined[train_idx], y_combined[train_idx],
        X_combined[val_idx], y_combined[val_idx],
        instrument=f"EUR_USD_fold_{fold_idx}",
    )
    
    fold_results.append((trainer, metrics))

# Select best fold
best_trainer, best_metrics = max(fold_results, key=lambda x: x[1]['val_accuracy'])
print(f"Best fold: val_accuracy={best_metrics['val_accuracy']:.1%}")
```

### Method 3: Disable Walk-Forward (Standard Training)

Set `enabled: false` in config or pass `None` for `wf_config`:

```python
trainer, metrics = train_with_walkforward_validation(
    trainer_class=TransformerDirectionTrainer,
    trainer_config=TrainerConfig(),
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    wf_config=None,  # Disable walk-forward, use standard training
)
```

## Aggregation Strategies

### "best" (Recommended)

Selects the fold with highest validation accuracy:

```python
wf_config = {'aggregate_method': 'best', ...}
```

**Pros:**
- Simple and effective
- Best out-of-sample performance
- No ensemble complexity

**Cons:**
- Single model (no diversity)
- May overfit to validation set

### "average"

Returns last fold's trainer but with averaged metrics:

```python
wf_config = {'aggregate_method': 'average', ...}
```

**Pros:**
- Robust metric estimates
- Includes standard deviation
- Good for benchmarking

**Cons:**
- Uses last fold's model (may not be best)
- Metrics don't match returned model

### "ensemble"

Returns last fold's trainer:

```python
wf_config = {'aggregate_method': 'ensemble', ...}
```

**Note:** Full ensemble support (combining predictions from all folds) is not yet implemented. This currently behaves like returning the last fold.

## Purged K-Fold

Purged K-Fold adds additional safeguards to prevent information leakage:

```
Without Purging:
|---TRAIN---|TEST---|

With Purging (purge_gap=24):
|---TRAIN---[PURGE]|TEST|[EMBARGO]---|
```

- **Purge Gap**: Removes training samples too close to test period
- **Embargo Gap**: Removes test samples immediately after training period
- **Based on**: "Advances in Financial Machine Learning" by Marcos Lopez de Prado

Enable in config:

```yaml
walkforward:
  use_purged_kfold: true
  purge_gap: 24      # 1 day for H1 bars
  embargo_gap: 12    # 12 hours for H1 bars
```

## Best Practices

### For H1 (Hourly) Timeframe

1. **Use rolling mode**: Markets change, recent data is most relevant
2. **Gap = 24 bars**: Prevents look-ahead bias (1 day gap)
3. **5 folds**: Good balance between validation robustness and training time
4. **Train size = 60%**: Adequate history without being too stale
5. **Retrain per fold**: Essential for realistic performance estimates

### For Different Timeframes

| Timeframe | Recommended Gap | Train Size | N Splits |
|-----------|----------------|------------|----------|
| M5 (5min) | 288 (1 day) | 0.70 | 7 |
| M15 (15min) | 96 (1 day) | 0.65 | 6 |
| H1 (1hour) | 24 (1 day) | 0.60 | 5 |
| H4 (4hour) | 6 (1 day) | 0.60 | 5 |
| D1 (daily) | 5 (1 week) | 0.50 | 4 |

### Performance Considerations

- **Training time**: ~5x longer than standard training (5 folds)
- **Memory**: Minimal overhead (same data, different splits)
- **Recommendations**:
  - Start with `n_splits=3` for quick iterations
  - Use `n_splits=5` for production
  - Consider `n_splits=7-10` for final validation

## Interpretation of Results

### Metrics to Watch

```
Fold Results:
Fold 1: val_accuracy=0.58
Fold 2: val_accuracy=0.62
Fold 3: val_accuracy=0.56
Fold 4: val_accuracy=0.61
Fold 5: val_accuracy=0.59

Mean: 0.592 ± 0.023 (std)
```

**Good signs:**
- ✅ Consistent performance across folds (low std)
- ✅ Mean > 0.55 for direction prediction
- ✅ No degrading trend (stable over time)

**Warning signs:**
- ⚠️ High variance (std > 0.05): Unstable model
- ⚠️ Degrading trend: Concept drift
- ⚠️ One fold much better: Possible overfitting

### Comparing with Standard Training

```
Standard Training: val_accuracy=0.65
Walk-Forward Mean: val_accuracy=0.59 ± 0.02
```

- **Gap is expected**: Walk-forward is more conservative
- **Typical gap**: 2-8% lower with walk-forward
- **Large gap (>10%)**: May indicate overfitting in standard approach

## Troubleshooting

### "Insufficient data for N folds"

**Problem**: Not enough samples for requested folds

**Solutions:**
1. Reduce `n_splits` (try 3 instead of 5)
2. Reduce `train_size` (try 0.50 instead of 0.60)
3. Fetch more data (increase `candles` in config)

### "Gap too large for dataset"

**Problem**: Gap exceeds available samples

**Solutions:**
1. Reduce `gap` (try 12 instead of 24)
2. Increase dataset size
3. Use smaller `val_size` and `test_size`

### Poor performance on all folds

**Problem**: Model doesn't generalize

**Solutions:**
1. Check feature quality and relevance
2. Try simpler model (reduce layers/units)
3. Increase regularization
4. Use more training data

### High variance across folds

**Problem**: Inconsistent performance

**Solutions:**
1. Check for regime changes (use regime-aware validation)
2. Increase `train_size` for more stable estimates
3. Use ensemble aggregation instead of single best
4. Investigate specific folds with poor performance

## Future Enhancements

Planned improvements:

1. **True ensemble aggregation**: Combine predictions from all folds
2. **Regime-aware splitting**: Ensure each fold has representative regime distribution
3. **Online learning integration**: Incremental updates between folds
4. **Drift detection**: Automatic detection of performance degradation
5. **Parallel fold training**: Train multiple folds simultaneously
6. **Advanced purging**: Label-based purging for triple barrier labeling
7. **Combinatorial purged CV**: Test multiple train/test combinations

## References

1. "Advances in Financial Machine Learning" by Marcos Lopez de Prado
   - Chapter 7: Cross-Validation in Finance
   - Purged K-Fold implementation

2. "Machine Learning for Algorithmic Trading" by Stefan Jansen
   - Walk-forward optimization techniques

3. TimeSeriesSplit (scikit-learn)
   - Standard time-series cross-validation

## Testing

Run tests to verify walk-forward implementation:

```bash
pytest tests/test_walkforward_config.py -v
```

Test coverage:
- ✅ Config creation and loading
- ✅ Rolling vs expanding modes
- ✅ Temporal ordering guarantees
- ✅ Purged K-fold splits
- ✅ No look-ahead bias
- ✅ YAML configuration parsing

## Summary

Walk-forward cross-validation is a critical component for robust model evaluation in time-series trading. By maintaining temporal ordering, using sliding windows, and retraining per fold, we ensure that performance estimates are realistic and account for market dynamics.

**Key takeaways:**
- Use rolling mode for FX trading (recent data matters most)
- Retrain per fold for realistic estimates
- Expect 2-8% lower performance vs standard training (this is normal)
- Monitor fold variance to detect instability
- Start with 3-5 folds for balance of speed and robustness

