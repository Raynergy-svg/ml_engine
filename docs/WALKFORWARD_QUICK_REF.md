# Walk-Forward Validation Quick Reference

## Enable Walk-Forward Validation

### In Config (config/config_improved_H1.yaml)

```yaml
walkforward:
  enabled: true
  mode: "rolling"        # Use sliding window
  n_splits: 5           # 5 folds
  retrain_per_fold: true
```

### In Code

```python
from src.training.buddy_training_helpers import train_with_walkforward_validation

trainer, metrics = train_with_walkforward_validation(
    trainer_class=TransformerDirectionTrainer,
    trainer_config=config,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    wf_config=config.get('walkforward'),
)
```

## Key Parameters

| Parameter | H1 Value | M5 Value | Purpose |
|-----------|----------|----------|---------|
| `gap` | 24 | 288 | Prevent look-ahead (1 day) |
| `train_size` | 0.60 | 0.70 | Training window % |
| `n_splits` | 5 | 7 | Number of folds |
| `mode` | "rolling" | "rolling" | Sliding window |

## Visual Guide

### Rolling Mode (Recommended for FX)
```
|---TRAIN---|gap|VAL|TEST|
     |---TRAIN---|gap|VAL|TEST|
          |---TRAIN---|gap|VAL|TEST|
```

### Expanding Mode (More data per fold)
```
|-TRAIN-|gap|VAL|TEST|
|----TRAIN----|gap|VAL|TEST|
|--------TRAIN--------|gap|VAL|TEST|
```

## Aggregation Methods

- `"best"`: Use fold with highest validation accuracy (recommended)
- `"average"`: Average metrics across folds (good for benchmarking)
- `"ensemble"`: Use last fold (placeholder for future ensemble)

## Expected Performance

- Walk-forward typically 2-8% lower than standard training
- This is **normal** and represents realistic out-of-sample performance
- High variance (std > 0.05) indicates model instability

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Insufficient data" | Reduce `n_splits` to 3 |
| "Gap too large" | Reduce `gap` to 12 |
| Poor all folds | Check features, add data, simplify model |
| High variance | Increase `train_size`, check regime changes |

## File Locations

- **Config**: `config/config_improved_H1.yaml`
- **Implementation**: `src/training/walkforward_validation.py`
- **Wrapper**: `src/training/buddy_training_helpers.py::train_with_walkforward_validation()`
- **Tests**: `tests/test_walkforward_config.py`
- **Documentation**: `docs/WALKFORWARD_VALIDATION_GUIDE.md`
