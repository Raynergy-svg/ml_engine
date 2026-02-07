# Walk-Forward Validation Examples

## Integration into main.py

To integrate walk-forward validation into the Buddy training pipeline, modify `main.py` around line 2934:

### Current Code
```python
dir_trainer = TransformerDirectionTrainer(trainer_config)
dir_metrics = dir_trainer.train(
    dir_data['X_train'], dir_data['y_train'],
    dir_data['X_val'], dir_data['y_val'],
    feature_names=dir_data['feature_names'],
    w_train=dir_data.get('w_train'),
    w_val=dir_data.get('w_val'),
    warm_start_path=str(warm_start_path) if warm_start_path else None,
    instrument=training_instrument,
)
```

### Modified Code (with walk-forward support)
```python
from src.training.buddy_training_helpers import train_with_walkforward_validation

wf_config = cfg.get('walkforward')

dir_trainer, dir_metrics = train_with_walkforward_validation(
    trainer_class=TransformerDirectionTrainer,
    trainer_config=trainer_config,
    X_train=dir_data['X_train'],
    y_train=dir_data['y_train'],
    X_val=dir_data['X_val'],
    y_val=dir_data['y_val'],
    feature_names=dir_data['feature_names'],
    w_train=dir_data.get('w_train'),
    w_val=dir_data.get('w_val'),
    warm_start_path=str(warm_start_path) if warm_start_path else None,
    instrument=training_instrument,
    wf_config=wf_config,  # Pass walk-forward config
    console=console,
)
```

## Configuration

Enable walk-forward in `config/config_improved_H1.yaml`:

```yaml
walkforward:
  enabled: true
  mode: "rolling"
  n_splits: 5
  retrain_per_fold: true
```

## See Also

- [Complete Guide](../docs/WALKFORWARD_VALIDATION_GUIDE.md): Comprehensive documentation
- [Quick Reference](../docs/WALKFORWARD_QUICK_REF.md): Essential commands and parameters
- [Tests](../tests/test_walkforward_config.py): Test suite and usage examples
